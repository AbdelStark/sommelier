# 07 Testing Strategy

- Status: Draft
- Target milestone: v1.0
- Primary RFCs: [RFC-0002](../rfcs/RFC-0002-dataset-preparation-and-split-discipline.md), [RFC-0005](../rfcs/RFC-0005-evaluation-parser-and-metrics.md), [RFC-0011](../rfcs/RFC-0011-security-licensing-and-release-gates.md)

## Test Pyramid

| Layer | Required coverage |
|-------|-------------------|
| Unit | Config validation, schema models, parser, metrics, artifact checksums, redaction. |
| Property | Split disjointness, metric bounds, parser failure classification, manifest round trip. |
| Fixture integration | Data preparation on tiny rows, formatting golden files, report comparison. |
| Remote smoke | Small remote run through data, formatting, one baseline batch, and one training step. |
| Full reference | Manual or scheduled v1.0 run producing release artifacts. |

## Local Test Commands

```text
uv run pytest
uv run ruff check .
uv run mypy sommelier tests
sommelier data validate-fixtures
sommelier config validate --config examples/config.smoke.yaml
```

The non-GPU suite must run on a clean development machine without downloading model weights.

## Required Unit Tests

- Config rejects unknown fields and invalid split sizes.
- Manifest writer refuses absolute artifact paths.
- Data validator classifies malformed `tools` and `answers`.
- Splitter never places the same normalized query in more than one split.
- Formatter golden examples preserve system, user, and assistant roles.
- Parser returns `no_json`, `invalid_json`, `invalid_shape`, or `ok` deterministically.
- Argument F1 handles missing, extra, nested, and scalar arguments.
- Comparison report rejects mismatched test split digests.
- Redaction scanner catches token-like environment values.

## ML-Specific Tests

- The training collator masks prompt tokens and leaves assistant target tokens unmasked.
- The base and adapter evaluation paths use the same prompt digest for each example.
- Deterministic decoding config sets temperature to `0.0` and disables sampling.
- A smoke training step writes an adapter directory and training manifest.

## Release Gates

No v1.0 release is valid unless:

1. Local tests pass.
2. Formatting and parser golden fixtures pass.
3. Remote smoke run passes.
4. Full reference run writes all required artifacts.
5. The comparison report includes limitations and reproduction commands.
6. License preflight passes.

## Test Data

Fixture rows live under `tests/fixtures/` and must be synthetic. Do not commit private tool schemas or credentials.
