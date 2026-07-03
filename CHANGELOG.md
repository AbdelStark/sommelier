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
- Data preparation drops rows whose answers contain more than one tool
  call, with the declared `multi_call_answer` drop reason (v1 trains and
  scores exactly one call; the previous behavior scored faithful
  multi-call outputs as failures). Migration: regenerate prepared splits;
  multi-call rows now appear in the drop summary instead of the splits.
- `examples/config.smoke.yaml` raises `train.max_sequence_length` to
  2048; real xlam prompts exceed the previous 1024-token budget.

### Added (remote serving)

- `remote_serving.py`: Modal entrypoint serving the trained adapter with
  vLLM's OpenAI-compatible server (`--enable-lora`), registering both the
  base model and the `sommelier-tool-caller` LoRA on one endpoint, with
  adapter sourcing from the published Hugging Face repo or the artifacts
  volume, optional Bearer-token protection, scale-to-zero, a readiness-
  polling smoke entrypoint that validates completions through the
  sommelier parser, and a `diagnose` entrypoint that runs the engine in
  the foreground with full logs.
- `sommelier.remote.images.vllm_serving_image`: built from the CUDA devel
  base image because vLLM's startup warm-up JIT-compiles kernels with
  nvcc, which slim images lack.

### Added (remote execution)

- `remote_pipeline.py`: Modal entrypoint running the full pipeline on a
  GPU — exports the configured Hugging Face dataset to raw rows, chains
  the shared stages with per-stage GPU cleanup and volume commits, audits
  rendered sequence lengths against the training budget before any model
  loads, and persists artifacts to the `sommelier-artifacts` volume.

### Fixed

- Remote images no longer run pip installs after mounting the package
  source, which Modal rejects at build time (`sommelier.remote.images`).
- Evaluation no longer re-adds special tokens to rendered prompts, which
  doubled the BOS token on Llama-family tokenizers
  (`sommelier.evaluation.generate`).
- Training no longer lets the Trainer strip `prompt_text`/`full_text`
  from batches before the completion-only collator runs
  (`remove_unused_columns=False` in `sommelier.training.qlora`).
- Drop-reason counters are derived from the `DropReason` literal, fixing
  a KeyError when a new reason was added (`sommelier.data.split`).
- Model loading works on Apple Silicon and CPU hosts: `device_map="auto"`
  is now used only when CUDA is available, otherwise the model loads
  normally and moves to MPS/CPU, fixing adapter dispatch failures in
  local serving (`sommelier.evaluation.generate`).
- The serving completions endpoint accepts its JSON body again: a
  postponed-annotation resolution issue had demoted the request body to
  a required query parameter (`sommelier.serving.openai_compat`); an
  HTTP-level end-to-end test now guards the route where fastapi is
  installed.
