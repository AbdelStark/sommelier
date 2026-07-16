# Metrics

Sommelier scores every evaluation with the same five metrics, computed in [`sommelier/evaluation/metrics.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/evaluation/metrics.py). All five are deliberately conservative: parse failures count against every metric, argument comparisons are exact canonical-JSON matches, and there is no partial credit anywhere except argument F1, whose partial credit is itself exact at the level of individual argument values. This page gives the precise definition of each metric with a worked example.

## How scoring works

Evaluation produces exactly one generation per formatted test prompt. Scoring joins each generation to its example by `example_id`, verifies that the generation's `prompt_sha256` equals the digest the formatter recorded (a mismatch is an invariant violation, not a skipped row), and takes the gold call from the example's target. Data preparation guarantees exactly one gold call per example; the [drop summary](../concepts/data.md) records the multi-call rows that were filtered out.

Two rules hold across all metrics:

- **The denominator is every evaluated example.** The four rate metrics divide by the full test-split size. An example whose output failed to parse scores zero on all of them; it is never dropped from the denominator (INV-DATA-005).
- **Every metric is stored as a `MetricValue`**: `{value, numerator, denominator}`, where `value = numerator / denominator` (0.0 when the denominator is 0). Reports keep the raw counts so any rate can be recomputed from the [evaluation report](artifacts.md) without trusting the float.

## Parse statuses

Scoring starts from the parse status the [conservative parser](../concepts/evaluation.md) (`sommelier.parser.v1`) assigned to each raw generation. The parser extracts the first balanced `{}` or `[]` span (string- and escape-aware, so brackets inside quoted values do not confuse it), ignores any surrounding prose, and repairs nothing inside the span.

| Status | When | Effect on metrics |
|--------|------|-------------------|
| `ok` | the span parses as JSON and is exactly `{"name": <non-empty str>, "arguments": <object>}`, or a one-element array wrapping exactly that | eligible for name and argument credit |
| `no_json` | the output contains no `{` or `[` at all | zero on every metric |
| `invalid_json` | an opening bracket exists but no balanced span can be extracted, or the span is not valid JSON | zero on every metric |
| `invalid_shape` | the JSON parses but is not a single tool call: extra or missing keys, an empty or non-string name, non-object arguments, an empty array, or a multi-call array | zero on every metric |

## Argument flattening

`argument_exact_match` compares whole argument objects; `argument_f1` needs finer grain, so it flattens both argument objects into path/value pairs:

- Object keys extend the path with `.key`; list elements are compared by index with `[i]`.
- Every leaf value is serialized as canonical scalar JSON (sorted keys, compact separators, ASCII), so `"Paris"`, `120`, `true`, and `null` each have exactly one spelling.
- Empty objects and empty arrays are leaves, not nothing: `{"options": {}}` produces the pair `options: {}`. An entirely empty arguments object produces the single pair `<root>: {}`, so a gold call with no arguments still contributes to the F1 denominator.

For example:

```json
{"city": "Paris", "filters": {"max_price": 120, "tags": ["romantic", "quiet"]}, "options": {}}
```

flattens to five pairs:

| Path | Canonical value |
|------|-----------------|
| `city` | `"Paris"` |
| `filters.max_price` | `120` |
| `filters.tags[0]` | `"romantic"` |
| `filters.tags[1]` | `"quiet"` |
| `options` | `{}` |

A pair matches only when the path and the value both match. Order inside objects never matters (paths are position-free); order inside lists always matters (indices are part of the path).

## Worked example

Four test examples, each with the two-key gold arguments shown, scored against four model outputs:

| # | Gold call | Model output (essentials) | Parse status |
|---|-----------|---------------------------|--------------|
| 1 | `get_weather` `{"city": "Paris", "units": "celsius"}` | identical call | `ok` |
| 2 | `get_weather` `{"city": "Paris", "units": "celsius"}` | `get_weather` `{"city": "Paris", "units": "kelvin"}` | `ok` |
| 3 | `get_weather` `{"city": "Oslo", "units": "celsius"}` | `fetch_weather` `{"city": "Oslo", "units": "celsius"}` | `ok` |
| 4 | `get_weather` `{"city": "Rome", "units": "celsius"}` | `The weather in Rome is sunny.` | `no_json` |

### valid_json_rate

The share of examples whose output parsed into a schema-valid tool call: numerator is the count of `ok` records, denominator is all examples. Examples 1 to 3 parse; example 4 does not. Result: `{value: 0.75, numerator: 3, denominator: 4}`.

### function_name_accuracy

The share of examples whose parsed call names the gold function exactly (string equality, after an `ok` parse). Examples 1 and 2 name `get_weather`; example 3 named the wrong function; example 4 never parsed. Result: `{value: 0.5, numerator: 2, denominator: 4}`.

### argument_exact_match

The share of examples whose whole arguments object equals the gold arguments object under canonical JSON serialization. The function name is not considered, so example 3 counts even though its name is wrong: examples 1 and 3 match, example 2 differs on `units`, example 4 never parsed. Result: `{value: 0.5, numerator: 2, denominator: 4}`.

### argument_f1

Micro-averaged F1 over flattened argument pairs, pooled across all examples: with `matched` the number of predicted pairs whose path and value both appear in the gold pairs,

```text
F1 = 2 * matched / (predicted_pairs + gold_pairs)
```

The stored numerator is `2 * matched` and the denominator is the pooled pair count, so the `MetricValue` is exact. A record without an `ok` parse contributes zero predicted pairs but its gold pairs still enter the denominator, which is how parse failures count against F1.

In the worked example every gold call flattens to 2 pairs, so `gold_pairs = 8`. Examples 1 to 3 each predict 2 pairs (`predicted_pairs = 6`), and match 2, 1, and 2 pairs respectively (`matched = 5`). Result: `{value: 0.7143, numerator: 10, denominator: 14}`.

### full_call_exact_match

The share of examples that match the gold call completely: an `ok` parse, the exact function name, and canonically equal arguments. This is the metric to read as "the model produced the right call". Only example 1 qualifies. Result: `{value: 0.25, numerator: 1, denominator: 4}`.

## Reading the reference run

The same arithmetic applied to the [reference run](../results/reference-run.md) on 1,000 held-out prompts: the base model's `valid_json_rate` of 0.916 means 84 outputs failed to parse and scored zero on all five metrics, which is a large part of why its `full_call_exact_match` sits at 0.705. The adapter parses all 1,000 (1.000), so its remaining losses are genuine name and argument errors rather than formatting failures.

The metric names and `MetricValue` shape are part of the artifact contract: they appear under `metrics` in every [evaluation report](artifacts.md), once per evaluated language slice and once overall, and the [comparison gate](../concepts/determinism.md) refuses to compare reports whose metric names or slice sets differ. Deltas in the comparison report are plain value differences, adapter minus base, computed per slice and overall. Each delta has a deterministic 95% paired-bootstrap interval produced by resampling identical example identities.

Language gaps use target minus reference. The primary estimate joins every translated row to its exact root and resamples those pairs; its coverage and pair-set digest are part of the artifact. A second block labeled `marginal_full_slices` compares the complete surviving slices without uncertainty and is descriptive because their cohorts can differ.
