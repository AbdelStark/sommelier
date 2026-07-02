# Changelog

All notable, user-visible changes to Sommelier are recorded here.

## Changelog policy

Per [docs/spec/09-release-and-versioning.md](docs/spec/09-release-and-versioning.md#changelog-policy),
every user-visible change records:

- a category: **Added**, **Changed**, **Fixed**, **Deprecated**,
  **Removed**, or **Security**;
- the affected command, module, or artifact schema;
- a migration note whenever behavior changes.

Entries land in the Unreleased section of the pull request that makes the
change; releases move them under a version heading with a date.

## Unreleased

### Added

- `sommelier` CLI stage surface: `config validate`, `data prepare`,
  `data validate-fixtures`, `format build`, `eval run`, `train run`,
  `report compare`, `pipeline run --mode smoke|full`, `serve adapter`,
  and `release preflight`.
- Tool-call formatting: canonical chat messages, tokenizer template
  rendering, prompt digests, and golden prompt fixtures
  (`sommelier.formatted_example.v1`).
- Evaluation: conservative JSON parser (`sommelier.parser.v1`), five
  tool-call metrics with numerators and denominators, deterministic
  generation records (`sommelier.generation.v1`), evaluation reports with
  comparability digests (`sommelier.evaluation_report.v1`), and the
  comparison gate with JSON and Markdown reports
  (`sommelier.comparison_report.v1`).
- Training: completion-only label masking with provable prompt
  boundaries, the QLoRA adapter training stage, training metrics
  (`sommelier.training_metric.v1`), and actionable OOM/timeout resource
  errors.
- Observability: structured JSONL stage logs (`sommelier.log_event.v1`)
  with write-time redaction, the artifact redaction scanner, per-stage
  runtime/hardware/cost metadata (`sommelier.runtime_metadata.v1`), and
  optional wandb experiment tracking via the `tracking` config section.
- Remote: Modal smoke app under `sommelier.remote.app` and separated
  data/train/eval/serving dependency images with GPU and timeout hooks.
- Serving: optional, illustrative single-adapter endpoint with RFC-0010
  request/response schemas and parser-status responses.
- Release gates: MIT `LICENSE`, `licenses/THIRD_PARTY.md`, and
  `sommelier release preflight` writing `release_preflight.json`
  (`sommelier.release_preflight.v1`).
- Docs: install quickstart, reproduction guide, serving limits, and this
  changelog with the v1.0 release checklist.

### Changed

- `format build` defaults to tokenizer chat-template rendering; the
  no-tokenizer path moved behind `--fixture`. Migration: append
  `--fixture` to keep the previous fixture behavior.
- `LogEvent` and `EvaluationReport` schemas gained fields
  (`schema_version` on log events; `created_at`, `test_split_sha256`,
  `prompt_set_sha256`, `decoding` on evaluation reports). Migration:
  regenerate artifacts with the current pipeline; readers fail closed on
  older shapes.
- The `tracking` config section is new and optional; existing configs
  remain valid.
