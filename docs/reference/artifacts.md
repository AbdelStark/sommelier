# Artifacts and schemas

Every stage communicates through schema-versioned files under one run directory, and this page is the file-level contract: where each file lives, which schema its records carry, and the exact fields of every schema. The reasoning behind the design is in [Artifacts as the interface](../concepts/artifacts.md); this page is for looking things up while staring at a run directory.

## The run directory

Everything a run produces lives under `<artifact_root>/runs/<run_id>/`:

```text
<artifact_root>/runs/<run_id>/
├── config.resolved.yaml         the frozen config (sommelier.config.v2 as YAML);
│                                its SHA-256 is the config_sha256 every manifest carries
├── manifest.json                run manifest: stage → stage-manifest path, run status
├── data_manifest.json           stage manifests, one per stage, at the run root
├── format_manifest.json
├── train_manifest.json
├── eval_manifest.json
├── report_manifest.json
├── data/
│   ├── train.jsonl              sommelier.prepared_example.v1
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── drop_summary.json        sommelier.drop_summary.v1: what was filtered, and why
├── formatted/
│   ├── train.jsonl              sommelier.formatted_example.v1
│   ├── validation.jsonl
│   └── test.jsonl
├── train/
│   ├── adapter/                 LoRA adapter weights as saved by peft
│   └── training_metrics.jsonl   sommelier.training_metric.v1, one record per logged step
├── eval/
│   ├── base/
│   │   ├── generations.jsonl    sommelier.generation.v1, one record per test prompt
│   │   └── evaluation_report.json   sommelier.evaluation_report.v1
│   └── adapter/                 the same two files, model_kind "adapter"
├── report/
│   ├── comparison_report.json   sommelier.comparison_report.v1 (authoritative)
│   └── comparison_report.md     human rendering of the JSON report
└── runtime_metadata.json        sommelier.runtime_metadata.v1: wall clock, GPU, cost evidence
```

Details worth knowing:

- Stage manifests sit at the run root, named `<stage>_manifest.json` for the stage names `data`, `format`, `train`, `eval`, `report`, and `serve`. Base and adapter evaluation are the same `eval` stage run twice, so `eval_manifest.json` reflects the most recent eval invocation; the per-model evidence lives in `eval/base/` and `eval/adapter/`.
- The manifest schema reserves a failed shape: `status: "failed"` plus `error_code` and `error_message`, redacted at build time (any secret-looking fragment replaces the whole message with `stage failed; details redacted`). Current stages fail by raising instead of writing one, so a missing stage manifest means the stage did not succeed.
- `sommelier release preflight` writes `release_preflight.json` at the artifact root itself, not inside a run, because its secret scan covers every run under that root.
- Structured JSONL logging (`sommelier.log_event.v1`) is designed as the source of truth for events, with console output as a rendering. In the current pipeline only the serving endpoint writes such a log (`train/logs/serve.jsonl`, next to the adapter it serves); the pipeline stages themselves do not write a run-root `logs/` directory.

## Schema catalog

`SUPPORTED_SCHEMAS` in [`sommelier/artifacts.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/artifacts.py) is a closed set of thirteen ids. Two more version strings live outside it, listed at the bottom of the table.

| Schema id | One record is | Written by |
|-----------|---------------|------------|
| `sommelier.config.v2` | the resolved run configuration (`config.resolved.yaml`) | run-directory creation, before any stage runs |
| `sommelier.config.v1` | the previous config schema; still recognized so artifacts from earlier runs keep validating | no current writer |
| `sommelier.manifest.v1` | a stage manifest or the run manifest | every stage |
| `sommelier.raw_tool_call_row.v1` | one untrusted source row: query plus raw `tools`/`answers` JSON strings | produced upstream (dataset export, fixtures); consumed by `data prepare` |
| `sommelier.prepared_example.v1` | one validated example with parsed tools, gold call, and split assignment | `data prepare` |
| `sommelier.drop_summary.v1` | the count of dropped rows per drop reason, plus split accounting | `data prepare` |
| `sommelier.formatted_example.v1` | one rendered prompt/target pair with its prompt digest | `format build` |
| `sommelier.generation.v1` | one raw model output with parse status and decoding config | `eval run` |
| `sommelier.evaluation_report.v1` | the five metrics plus identity digests for one model | `eval run` |
| `sommelier.comparison_report.v1` | the gated base-versus-adapter comparison | `report compare` |
| `sommelier.training_metric.v1` | one training log step (loss, learning rate, tokens) | `train run` |
| `sommelier.log_event.v1` | one structured log line | stage loggers (the serving endpoint logs with it) |
| `sommelier.release_preflight.v1` | the release gate results | `release preflight` |
| `sommelier.parser.v1` | not a file schema: the version string of the tool-call parser, recorded in the eval config and every evaluation report | `eval run` |
| `sommelier.runtime_metadata.v1` | per-stage wall clock, hardware, and cost evidence | `pipeline run` |

## Readers fail closed

`read_json_with_schema` and `read_jsonl_with_schema` require a `schema_version` on every JSON object and every JSONL line, and that version must be in `SUPPORTED_SCHEMAS`. A missing, unknown, or unexpected version raises `SchemaValidationError` (SOM202, exit 2) naming the file and line; see the [error reference](errors.md). There is no best-effort mode: an artifact from a different pipeline version is rejected, not reinterpreted.

The two ids outside the closed set behave differently by design. `sommelier.parser.v1` is a version string, not a file format. `sommelier.runtime_metadata.v1` has its own reader that treats an unknown version as absent evidence: the comparison report then renders its runtime section as `{"available": false}` instead of failing, because missing cost data should never block a metrics comparison.

## Record schemas

Field lists below are verified against the code; the config schema has its own [page](configuration.md).

### `sommelier.raw_tool_call_row.v1`

| Field | Type | Notes |
|-------|------|-------|
| `source_id` | str | identifier of the row in the source dataset |
| `query` | str | the natural-language request |
| `tools` | str | raw JSON string, untrusted until validation |
| `answers` | str | raw JSON string, untrusted until validation |
| `source_revision` | str | the pinned dataset revision the row came from |

### `sommelier.prepared_example.v1`

| Field | Type | Notes |
|-------|------|-------|
| `example_id` | str | stable per-example identifier |
| `source_id` | str | carried through from the raw row |
| `query` | str | the request text |
| `tools` | list[ToolSchema] | parsed `{name, description, parameters}` objects |
| `gold_calls` | list[ToolCall] | parsed `{name, arguments}`; exactly one call in v1 |
| `split` | `train` \| `validation` \| `test` | a `query_sha256` appears in exactly one split |
| `query_sha256` | str | digest of the normalized query, the dedupe key |
| `source_revision` | str | pinned dataset revision |

### `sommelier.drop_summary.v1`

| Field | Type | Notes |
|-------|------|-------|
| `counts` | dict[str, int] | dropped rows per [drop reason](../concepts/data.md) |
| `valid_rows` | int | rows that survived validation |
| `deduplicated_rows` | int | rows remaining after deduplication |
| `requested` | dict | the configured `train`/`validation`/`test` sizes |

### `sommelier.formatted_example.v1`

| Field | Type | Notes |
|-------|------|-------|
| `example_id`, `split` | str | join keys back to the prepared example |
| `messages` | list | `{role, content}` for system, user, assistant |
| `prompt_text` | str | system and user messages rendered with the generation prompt |
| `target_text` | str | the canonical JSON of the gold calls, nothing else |
| `full_text` | str | all three messages rendered; must start with `prompt_text` |
| `prompt_sha256` | str | SHA-256 of `prompt_text`; proves prompt identity later |
| `tokenizer_id`, `tokenizer_revision` | str | which tokenizer rendered the template |
| `template_policy` | str | `tokenizer_chat_template` |

### `sommelier.generation.v1`

| Field | Type | Notes |
|-------|------|-------|
| `example_id` | str | must reference a formatted test example |
| `model_kind` | `base` \| `adapter` | which model produced the output |
| `prompt_sha256` | str | must equal the formatted example's digest (INV-ARCH-004) |
| `raw_text` | str | the full generation, retained even on parse failure |
| `parsed_call` | ToolCall \| null | null whenever `parse_status` is not `ok` |
| `parse_status` | `ok` \| `no_json` \| `invalid_json` \| `invalid_shape` | see the [metric reference](metrics.md) |
| `decoding` | dict | `{temperature, do_sample, max_new_tokens}`; must be uniform across the file |

### `sommelier.evaluation_report.v1`

| Field | Type | Notes |
|-------|------|-------|
| `created_at`, `run_id`, `model_kind` | str | report identity |
| `config_sha256` | str | digest of `config.resolved.yaml` |
| `split` | `test` | the only split evaluation reads |
| `metrics` | dict | the five [metric names](metrics.md) mapped to `{value, numerator, denominator}` |
| `generation_artifact` | str | relative path to the scored `generations.jsonl` |
| `parser_version` | `sommelier.parser.v1` | checked by the comparison gate |
| `test_split_sha256` | str | digest of the formatted test split file |
| `prompt_set_sha256` | str | digest over the ordered per-example prompt digests |
| `decoding` | dict | the uniform decoding config of the generations |

Six of these fields form the identity the [comparison gate](../concepts/determinism.md) checks: `config_sha256`, `split`, `test_split_sha256`, `prompt_set_sha256`, `parser_version`, and `decoding`. `generation_artifact` is evidence, not identity.

### `sommelier.comparison_report.v1`

| Field | Type | Notes |
|-------|------|-------|
| `created_at`, `run_id` | str | report identity |
| `shared` | dict | the identity fields both evaluation reports agreed on |
| `base`, `adapter` | dict | each holds `run_id`, `metrics`, `generation_artifact` |
| `deltas` | dict[str, float] | adapter value minus base value, per metric |
| `runtime` | dict | from `runtime_metadata.json`; `{"available": false}` when missing |

### `sommelier.manifest.v1`

Stage manifest fields:

| Field | Type | Notes |
|-------|------|-------|
| `stage` | `data` \| `format` \| `train` \| `eval` \| `report` \| `serve` | |
| `run_id`, `created_at`, `git_commit` | str | `git_commit` is `unknown` outside a repo |
| `config_sha256` | str | ties the stage to the resolved config |
| `dependency_lock_sha256` | str \| null | SHA-256 of `uv.lock`, null when absent |
| `command` | list[str] | the CLI invocation that produced the outputs |
| `seed` | int | `project.seed` |
| `inputs`, `outputs` | list[ArtifactRef] | every file read and written, with checksums |
| `status` | `succeeded` \| `failed` | failed manifests add `error_code` and a redacted `error_message` |

The run manifest (`manifest.json`) shares the schema id and carries `run_id`, `stages` (stage name to stage-manifest path), `config` (an ArtifactRef to the resolved config), `status` (`running`, `succeeded`, or `failed`), and an optional `tracking` object with the external tracker's provider, project, and run URL.

### `sommelier.log_event.v1`

| Field | Type | Notes |
|-------|------|-------|
| `timestamp` | str | UTC ISO 8601 |
| `level` | `debug` \| `info` \| `warning` \| `error` | |
| `run_id`, `stage`, `event` | str | correlation keys for offline joins |
| `message` | str | redacted before writing |
| `fields` | dict | JSON-native scalars only; non-finite floats are rejected at write time |

### `sommelier.training_metric.v1`

| Field | Type | Notes |
|-------|------|-------|
| `step` | int | trainer global step |
| `epoch` | float | position within training |
| `train_loss`, `eval_loss` | float \| null | non-finite values fail closed instead of being persisted |
| `learning_rate` | float | |
| `tokens_seen` | int | cumulative input tokens |
| `peak_gpu_memory_mb` | int \| null | run-level measurement, recorded on the final record only |

### `sommelier.release_preflight.v1`

| Field | Type | Notes |
|-------|------|-------|
| `created_at` | str | |
| `status` | `pass` \| `fail` | the report is written even when gates fail, so evidence survives |
| `gates` | list | each gate is `{name, status, evidence}` with status `pass`, `fail`, or `skip` |

## ArtifactRef and the relative-path rule

Every input and output in a manifest is an `ArtifactRef`:

| Field | Type | Notes |
|-------|------|-------|
| `path` | str | POSIX path relative to the artifact root |
| `kind` | str | e.g. `dataset_split`, `formatted_split`, `generations`, `adapter_weights` |
| `schema_version` | str | empty for schema-less files such as adapter weights |
| `sha256` | str | checksum of the file bytes |
| `bytes` | int | file size |

Paths are always relative to the artifact root, so a run directory can be moved, archived, or published without breaking its own manifests. Building a ref for a path outside the artifact root raises `SchemaValidationError`; absolute paths never appear in input or output references (the `command` field records the argv verbatim, including any absolute paths you passed). Writes are atomic: content goes to `<name>.tmp.<pid>` first and is moved into place only after the writer finishes, so a crash never leaves a half-written artifact under a final name.

## Run IDs

`create_run_id` produces `<UTC timestamp>-<8 hex chars>`, for example `20260702T091500Z-1a2b3c4d` (format `%Y%m%dT%H%M%SZ` plus the first 8 characters of a UUID4). Smoke pipeline runs get a `smoke-` prefix so a later full run can never overwrite smoke artifacts. When a [CLI command](cli.md) is pointed at a path inside a run and `--run-id` is omitted, the run ID is inferred from the path with the pattern `/runs/([^/]+)/`.
