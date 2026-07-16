from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from fractions import Fraction
from typing import Final, TypedDict

from sommelier.errors import EvaluationError
from sommelier.evaluation.metrics import METRIC_NAMES, ScoredRecord, metric_components

PAIRED_BOOTSTRAP_VERSION: Final = "sommelier.paired_bootstrap.v1"
EXACT_MCNEMAR_VERSION: Final = "sommelier.exact_mcnemar.v1"
DEFAULT_RESAMPLES: Final = 2000
DEFAULT_CONFIDENCE_LEVEL: Final = 0.95


class ConfidenceInterval(TypedDict):
    lower: float
    upper: float


class PairedBootstrapResult(TypedDict):
    method: str
    seed: int
    confidence_level: float
    resamples: int
    intervals: dict[str, ConfidenceInterval]


class McNemarDiscordantCounts(TypedDict):
    reference_correct_candidate_incorrect: int
    reference_incorrect_candidate_correct: int


class ExactMcNemarResult(TypedDict):
    method: str
    metric: str
    alternative: str
    pairs: int
    discordant_pairs: int
    discordant_counts: McNemarDiscordantCounts
    p_value: float


def stable_bootstrap_seed(base_seed: int, comparison_name: str) -> int:
    """Derives a stable independent stream from the experiment seed."""
    digest = hashlib.sha256(comparison_name.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_paired_identities(
    reference: Sequence[ScoredRecord],
    candidate: Sequence[ScoredRecord],
    *,
    method: str,
) -> None:
    if len(reference) != len(candidate) or not reference:
        raise EvaluationError(
            f"{method} requires equally sized non-empty cohorts",
            hint="Join evaluations by example identity before computing paired evidence.",
        )
    reference_ids = [record["example_id"] for record in reference]
    candidate_ids = [record["example_id"] for record in candidate]
    if any(not example_id for example_id in (*reference_ids, *candidate_ids)):
        raise EvaluationError(
            f"{method} requires non-empty example identities",
            hint="Assign every evaluation row a stable example identity.",
        )
    if reference_ids != candidate_ids:
        raise EvaluationError(
            f"{method} example identities differ",
            hint="Order both cohorts by the same canonical pair identity.",
        )
    if len(set(reference_ids)) != len(reference_ids):
        raise EvaluationError(
            f"{method} requires unique example identities",
            hint="Keep exactly one scored row per example in each cohort.",
        )


def _exact_two_sided_binomial_p_value(first: int, second: int) -> float:
    """Returns ``2 * P[X <= min(first, second)]`` for X ~ Bin(n, 0.5).

    Integer binomial coefficients and :class:`Fraction` preserve the exact
    finite-sample calculation until the final JSON-compatible float cast.
    """
    discordant = first + second
    if discordant == 0:
        return 1.0
    lower = min(first, second)
    coefficient = 1
    tail_numerator = coefficient
    for successes in range(1, lower + 1):
        coefficient = coefficient * (discordant - successes + 1) // successes
        tail_numerator += coefficient
    probability = Fraction(2 * tail_numerator, 1 << discordant)
    return float(min(probability, Fraction(1, 1)))


def exact_mcnemar_full_call(
    reference: Sequence[ScoredRecord],
    candidate: Sequence[ScoredRecord],
) -> ExactMcNemarResult:
    """Exact paired McNemar evidence for full-call exact-match outcomes.

    The two-sided p-value is the doubled lower binomial tail over discordant
    pairs, capped at one. It is supporting significance evidence; experiment
    claim gates remain defined by their predeclared paired-bootstrap bounds.
    """
    _validate_paired_identities(reference, candidate, method="exact McNemar")
    reference_only = 0
    candidate_only = 0
    for reference_record, candidate_record in zip(reference, candidate, strict=True):
        reference_correct = bool(
            metric_components(reference_record)["full_call_exact_match"]["numerator"]
        )
        candidate_correct = bool(
            metric_components(candidate_record)["full_call_exact_match"]["numerator"]
        )
        reference_only += int(reference_correct and not candidate_correct)
        candidate_only += int(candidate_correct and not reference_correct)

    return ExactMcNemarResult(
        method=EXACT_MCNEMAR_VERSION,
        metric="full_call_exact_match",
        alternative="two-sided",
        pairs=len(reference),
        discordant_pairs=reference_only + candidate_only,
        discordant_counts=McNemarDiscordantCounts(
            reference_correct_candidate_incorrect=reference_only,
            reference_incorrect_candidate_correct=candidate_only,
        ),
        p_value=_exact_two_sided_binomial_p_value(reference_only, candidate_only),
    )


def paired_bootstrap_intervals(
    reference: Sequence[ScoredRecord],
    candidate: Sequence[ScoredRecord],
    *,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PairedBootstrapResult:
    """Percentile CIs for ``candidate - reference`` on identical examples.

    Resampling is over example pairs, not independent language/model rows. The
    method is deterministic for a recorded seed and uses the metrics' exact
    additive numerator/denominator semantics, including micro argument F1.
    """
    if len(reference) != len(candidate) or not reference:
        raise EvaluationError(
            "paired bootstrap requires equally sized non-empty cohorts",
            hint="Join evaluations by example identity before computing uncertainty.",
        )
    reference_ids = [record["example_id"] for record in reference]
    candidate_ids = [record["example_id"] for record in candidate]
    if reference_ids != candidate_ids:
        raise EvaluationError(
            "paired bootstrap example identities differ",
            hint="Order both cohorts by the same canonical pair identity.",
        )
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    reference_components = [metric_components(record) for record in reference]
    candidate_components = [metric_components(record) for record in candidate]
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    rng = random.Random(seed)
    size = len(reference)
    for _ in range(resamples):
        reference_totals = {name: [0, 0] for name in METRIC_NAMES}
        candidate_totals = {name: [0, 0] for name in METRIC_NAMES}
        for _ in range(size):
            index = rng.randrange(size)
            for name in METRIC_NAMES:
                reference_component = reference_components[index][name]
                candidate_component = candidate_components[index][name]
                reference_totals[name][0] += reference_component["numerator"]
                reference_totals[name][1] += reference_component["denominator"]
                candidate_totals[name][0] += candidate_component["numerator"]
                candidate_totals[name][1] += candidate_component["denominator"]
        for name in METRIC_NAMES:
            reference_numerator, reference_denominator = reference_totals[name]
            candidate_numerator, candidate_denominator = candidate_totals[name]
            reference_value = (
                reference_numerator / reference_denominator if reference_denominator else 0.0
            )
            candidate_value = (
                candidate_numerator / candidate_denominator if candidate_denominator else 0.0
            )
            samples[name].append(candidate_value - reference_value)

    alpha = (1.0 - confidence_level) / 2.0
    intervals = {
        name: ConfidenceInterval(
            lower=_percentile(values, alpha),
            upper=_percentile(values, 1.0 - alpha),
        )
        for name, values in samples.items()
    }
    return PairedBootstrapResult(
        method=PAIRED_BOOTSTRAP_VERSION,
        seed=seed,
        confidence_level=confidence_level,
        resamples=resamples,
        intervals=intervals,
    )
