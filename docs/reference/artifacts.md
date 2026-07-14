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
├── tokenization_manifest.json
├── train_manifest.json
├── eval-base_manifest.json
├── eval-adapter_manifest.json
├── report_manifest.json
├── data/
│   ├── train.jsonl              sommelier.prepared_example.v2, all languages in one file
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── drop_summary.json        sommelier.drop_summary.v2: what was filtered per language, and why
│   └── source_inputs/           checksummed raw root/paired rows plus translation
│                                summary and publication manifest; full Hebrew
│                                also has review template + finalized review
├── formatted/
│   ├── train.jsonl              sommelier.formatted_example.v2
│   ├── validation.jsonl
│   └── test.jsonl
├── analysis/
│   └── tokenization/
│       ├── tokenizer_tax_records.jsonl sommelier.tokenizer_tax_record.v1
│       └── tokenizer_tax_report.json   sommelier.tokenizer_tax_report.v1
├── train/
│   ├── adapter/                 LoRA adapter weights as saved by peft
│   └── training_metrics.jsonl   sommelier.training_metric.v1, one record per logged step
├── eval/
│   ├── base/
│   │   ├── generations.<slice>.jsonl   sommelier.generation.v2, one record per test prompt, one file per language slice
│   │   ├── inference_telemetry.json sommelier.inference_telemetry.v2
│   │   └── evaluation_report.json   sommelier.evaluation_report.v3
│   └── adapter/                 the same generations, telemetry, and report files, model_kind "adapter"
├── report/
│   ├── comparison_report.json   sommelier.comparison_report.v3 (authoritative)
│   └── comparison_report.md     human rendering of the JSON report
└── runtime_metadata.json        sommelier.runtime_metadata.v1: wall clock, GPU, cost evidence
```

Details worth knowing:

- Stage manifests sit at the run root, named `<stage>_manifest.json`. Base and adapter evaluation use distinct `eval-base` and `eval-adapter` stages, so neither arm overwrites the other's manifest and both remain joined from the run manifest to their generation, telemetry, and report artifacts.
- The manifest schema reserves a failed shape: `status: "failed"` plus `error_code` and `error_message`, redacted at build time (any secret-looking fragment replaces the whole message with `stage failed; details redacted`). Current stages fail by raising instead of writing one, so a missing stage manifest means the stage did not succeed.
- `sommelier release preflight` writes `release_preflight.json` at the selected artifact root itself, not inside a run. The v2 identity certifies every regular file under that root except the report itself, so adapter publication runs it against the final curated bundle rather than a broader run tree.
- Structured JSONL logging (`sommelier.log_event.v1`) is designed as the source of truth for events, with console output as a rendering. In the current pipeline only the serving endpoint writes such a log (`train/logs/serve.jsonl`, next to the adapter it serves); the pipeline stages themselves do not write a run-root `logs/` directory.

## Schema catalog

`SUPPORTED_SCHEMAS` in [`sommelier/artifacts.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/artifacts.py) is the closed set used by the generic JSON/JSONL readers; the code is authoritative for its current size. Parser/runtime identities and the provider, diagnostic, and publication-transaction contracts use dedicated validators or writers and live outside that set, as marked below. Superseded versions, including v1/v2 evaluation and comparison reports and the v1 translation summary, stay in the set with no current writer so artifacts from earlier runs keep validating.

| Schema id | One record is | Written by |
|-----------|---------------|------------|
| `sommelier.config.v2` | the resolved run configuration (`config.resolved.yaml`) | run-directory creation, before any stage runs |
| `sommelier.manifest.v1` | a stage manifest or the run manifest | every stage |
| `sommelier.raw_tool_call_row.v1` | one untrusted source row: query plus raw `tools`/`answers` JSON strings, plus `source_example_id` on paired-source rows | produced upstream (dataset export, fixtures); consumed by `data prepare` |
| `sommelier.prepared_example.v2` | one validated example with parsed tools, gold call, language, and split assignment | `data prepare` |
| `sommelier.drop_summary.v2` | per-language counts of dropped rows per drop reason, plus split accounting | `data prepare` |
| `sommelier.formatted_example.v2` | one rendered prompt/target pair with its language and prompt digest | `format build` |
| `sommelier.generation.v2` | one raw model output with its language, parse status, and decoding config | `eval run` |
| `sommelier.inference_telemetry.v1` | superseded inference telemetry boundary retained for older artifacts | no current writer |
| `sommelier.inference_telemetry.v2` | sequential end-to-end generator-call elapsed time, one untimed warmup, configured GPU label, decoding, and artifact binding for one evaluation arm | `eval run` |
| `sommelier.evaluation_report.v3` | the five metrics per slice and overall, model identity, and exact paired-language cohorts with confidence intervals | `eval run` |
| `sommelier.comparison_report.v3` | the gated base-versus-adapter comparison with adapter-gain intervals, primary paired gaps, and labeled marginal gaps | `report compare` |
| `sommelier.experiment_report.v1` | gated base/v1/v3 evidence, independently recomputed metrics, paired intervals, bounded TCO evidence, and machine-readable claim decisions | `report experiment` |
| `sommelier.training_metric.v1` | one training log step (loss, learning rate, tokens) | `train run` |
| `sommelier.translation_summary.v2` | a paired-dataset build's target policy, input/output digests, translator request identity, decoding, and drop counts | `data translate` |
| `sommelier.openai_responses_provider_journal.v2` | one raw provider response, error, or replay event with source-id/attempt attribution, returned identity, completion status, usage, and output bytes where applicable | `remote_translate.py` OpenAI backend; dedicated validator |
| `sommelier.openai_responses_provider_journal_summary.v2` | a content-free validated aggregate of one raw provider journal | provider-evidence builder; dedicated validator |
| `sommelier.openai_provider_evidence.v2` | journal digest, exact requested/returned model and tier, clean counts, complete usage, and calculated public-list-price boundary | nested in `translation_summary.json`; dedicated validator |
| `sommelier.translation_semantic_review_template.v1` | machine-locked 200-row Hebrew sample and immutable back-translations before reviewer edits | `semantic-review-create` |
| `sommelier.translation_semantic_review.v1` | locked 200-row Hebrew back-translation sample, immutable model/input identities, review decisions, and whole-publication gate | semantic-review release gate |
| `sommelier.translation_publication_manifest.v1` | canonical paired-row identity and digests binding the summary and finalized semantic review at publication | translation publication |
| `sommelier.tokenizer_tax_record.v1` | exact per-example query/prompt/target/full token counts and ratios to its root | `analyze tokenization` |
| `sommelier.tokenizer_tax_report.v1` | per-language distributions, matched-pair coverage and ratios, budget counts, and projected non-padding workload | `analyze tokenization` |
| `sommelier.sovereign_tco_evidence.v1` | evidence-joined tokenizer, QLoRA training/storage, and three-arm inference-efficiency measurements with unavailable fields kept explicit | nested in `report experiment` |
| `sommelier.log_event.v1` | one structured log line | stage loggers (the serving endpoint logs with it) |
| `sommelier.release_preflight.v1` | historical unbound release-gate results | no current writer; readable for inspection only |
| `sommelier.release_preflight.v2` | release gates plus exact config, revision, source, lock, and artifact-tree identity | `release preflight`; strict adapter-publication validator |
| `sommelier.qlora_shape_preflight.v1` | diagnostic-only full-shape QLoRA execution, hardware, memory, and failure/success evidence | `remote_qlora_preflight.py`; dedicated validator/writer |
| `sommelier.qlora_shape_preflight_artifact_manifest.v1` | stable digests of the non-recursive diagnostic artifact set | QLoRA shape preflight; dedicated validator/writer |
| `sommelier.huggingface_publication_receipt.v1` | validation plan or durable `pending` / `commit_submitting` / `commit_returned_unverified` / `verified` Hub transaction journal with exact file hashes and parent/created commits | `release publish-dataset` / `publish-adapter`; preallocated dedicated writer |
| `sommelier.parser.v1` | not a file schema: the version string of the tool-call parser, recorded in the eval config and every evaluation report | `eval run` |
| `sommelier.runtime_metadata.v1` | per-stage wall clock, hardware, and cost evidence | `pipeline run` |

## Readers fail closed

`read_json_with_schema` and `read_jsonl_with_schema` require a `schema_version` on every JSON object and every JSONL line, and that version must be in `SUPPORTED_SCHEMAS`. A missing, unknown, or unexpected version raises `SchemaValidationError` (SOM202, exit 2) naming the file and line; see the [error reference](errors.md). There is no best-effort mode: an artifact from a different pipeline version is rejected, not reinterpreted.

The ids outside the closed set behave differently by design.
`sommelier.parser.v1` is a version string, not a file format.
`sommelier.runtime_metadata.v1` has its own reader that treats an unknown
version as absent evidence: the comparison report then renders its runtime
section as `{"available": false}` instead of failing, because missing cost data
should never block a metrics comparison. The OpenAI journal, journal summary,
and provider-evidence v2 contracts have strict dedicated validators; an unknown
schema, malformed attribution, mixed model/tier, incomplete usage, or dirty
full-run evidence fails the translation publication gate.
The QLoRA shape-preflight report/manifest and Hugging Face publication receipt
also stay outside the generic stage-artifact registry. Their producers validate
their complete closed contracts directly. The QLoRA digest manifest covers
every regular run-directory file present at finalization except the report and
manifest themselves, which cannot recursively hash their own final bytes;
the validator rehashes the on-disk set and rejects additions, omissions,
reordering, malformed paths, or digest drift. Publication receipts are reserved
with exclusive creation before any Hub access, then durably advance through
`pending`, `commit_submitting`, `commit_returned_unverified`, and finally
`verified` only after immutable-revision enumeration and per-file round-trip
SHA-256 succeeds.

For Modal pipeline runs, `remote_execution.function_timeout_seconds` is the
provider-enforced outer deadline. The adjacent
`configured_stage_planning_estimate_seconds` and
`outer_timeout_planning_headroom_seconds` values are arithmetic planning
evidence, not stage deadlines; `per_stage_watchdogs_enforced` is recorded as
`false` explicitly.

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
| `source_example_id` | str, optional | paired-source rows only: the root example this row translates |

### `sommelier.prepared_example.v2`

| Field | Type | Notes |
|-------|------|-------|
| `example_id` | str | stable per-example identifier, unique across languages |
| `source_id` | str | carried through from the raw row |
| `language` | str | the dataset source this example came from |
| `source_example_id` | str \| null | on paired rows, the root example this row translates; null on root rows |
| `query` | str | the request text |
| `tools` | list[ToolSchema] | parsed `{name, description, parameters}` objects |
| `gold_calls` | list[ToolCall] | parsed `{name, arguments}`; exactly one call |
| `split` | `train` \| `validation` \| `test` | a `query_sha256` appears in exactly one split; paired rows share their root's split |
| `query_sha256` | str | digest of the normalized query, the per-language dedupe key |
| `source_revision` | str | pinned dataset revision |

### `sommelier.drop_summary.v2`

| Field | Type | Notes |
|-------|------|-------|
| `languages` | dict | per language: `counts` per [drop reason](../concepts/data.md), `valid_rows`, `deduplicated_rows`, and final `split_sizes` |
| `requested` | dict | the configured `train`/`validation`/`test` sizes (they describe the root source; paired sources may run short) |

### `sommelier.formatted_example.v2`

| Field | Type | Notes |
|-------|------|-------|
| `example_id`, `split` | str | join keys back to the prepared example |
| `language` | str | carried from the prepared example; evaluation slices filter on it |
| `source_example_id` | str \| null | carried from the prepared example |
| `messages` | list | `{role, content}` for system, user, assistant |
| `prompt_text` | str | system and user messages rendered with the generation prompt |
| `target_text` | str | the canonical JSON of the gold calls, nothing else |
| `full_text` | str | all three messages rendered; must start with `prompt_text` |
| `prompt_sha256` | str | SHA-256 of `prompt_text`; proves prompt identity later |
| `tokenizer_id`, `tokenizer_revision` | str | which tokenizer rendered the template |
| `template_policy` | str | `tokenizer_chat_template` |

### `sommelier.generation.v2`

| Field | Type | Notes |
|-------|------|-------|
| `example_id` | str | must reference a formatted test example |
| `model_kind` | `base` \| `adapter` | which model produced the output |
| `language` | str | the evaluation slice this generation belongs to |
| `prompt_sha256` | str | must equal the formatted example's digest (INV-ARCH-004) |
| `raw_text` | str | the full generation, retained even on parse failure |
| `parsed_call` | ToolCall \| null | null whenever `parse_status` is not `ok` |
| `parse_status` | `ok` \| `no_json` \| `invalid_json` \| `invalid_shape` | see the [metric reference](metrics.md) |
| `decoding` | dict | `{temperature, do_sample, max_new_tokens}`; must be uniform across the file |

### `sommelier.evaluation_report.v3`

| Field | Type | Notes |
|-------|------|-------|
| `created_at`, `run_id`, `model_kind` | str | report identity |
| `model_identity` | dict | pinned base model and tokenizer ids/revisions; checked by the comparison gate |
| `config_sha256` | str | digest of `config.resolved.yaml` |
| `split` | `test` | the only split evaluation reads |
| `slices` | dict | per language: `metrics`, `examples`, `prompt_set_sha256`, and the `generation_artifact` path |
| `paired_slices` | dict | per target language: exact pair count/coverage/digest, root and target metrics, target-minus-root gaps, and deterministic paired-bootstrap intervals |
| `metrics` | dict | the five [metric names](metrics.md) across all slices, mapped to `{value, numerator, denominator}` |
| `adapter_source` | dict \| null | on adapter reports, where the weights came from, plus a local tree digest or immutable-revision signal when available |
| `parser_version` | `sommelier.parser.v1` | checked by the comparison gate |
| `test_split_sha256` | str | digest of the formatted test split file |
| `decoding` | dict | the uniform decoding config of the generations |

The identity the [comparison gate](../concepts/determinism.md) checks is `model_identity`, `config_sha256`, `split`, `test_split_sha256`, `parser_version`, `decoding`, the slice set, every slice's `prompt_set_sha256`, and every translated slice's `pair_set_sha256`. Generation artifact paths are evidence, not identity.

### `sommelier.inference_telemetry.v2`

Each evaluation arm writes one `inference_telemetry.json` next to its
generations. It binds the two generation artifacts, deterministic decoding,
configured GPU label/count, and per-slice and total elapsed time. The clock
scope is the sum of sequential per-example, end-to-end
`TextGenerator.generate` calls after model load. The default Transformers
implementation includes prompt tokenization, input device transfer,
`model.generate`, and generated-token decoding, and does not insert an explicit
device synchronization. Exactly one call using the first example in the first
configured slice warms the same path before measurement; its output is
discarded and the clock is not read around it. Model loading, parsing, and
artifact I/O are excluded, concurrency is one, and English then Hebrew slice
order is recorded. Derived GPU-seconds per exact successful call therefore
describes this bounded configured-allocation path, not kernel-only model time or
serving throughput under batching or concurrency.

### `sommelier.comparison_report.v3`

| Field | Type | Notes |
|-------|------|-------|
| `created_at`, `run_id` | str | report identity |
| `shared` | dict | the identity fields both evaluation reports agreed on |
| `base`, `adapter` | dict | each holds `run_id`, overall `metrics`, and adapter provenance where applicable |
| `deltas` | dict[str, float] | adapter value minus base value, per metric |
| `adapter_gain_ci95` | dict | paired-bootstrap method/seed/resample contract and overall per-metric intervals |
| `slices` | dict | per-language base/adapter metrics, deltas, adapter-gain intervals, prompt digest, and generation paths |
| `paired_language_gaps` | dict | primary target-minus-exact-root estimates, pair coverage/digest, and base/adapter intervals |
| `language_gaps` | dict | descriptive complete-slice gaps labeled `marginal_full_slices`; cohorts can differ |
| `runtime` | dict | from `runtime_metadata.json`; `{"available": false}` when missing |

### Tokenizer-tax schemas

Every `sommelier.tokenizer_tax_record.v1` line carries the example/root identity, language, split, counts for query characters/UTF-8 bytes/whitespace words and query/prompt/target/full tokens, the sequence-budget result, and ratios to the exact root for translated rows.

`sommelier.tokenizer_tax_report.v1` binds those records to the config digest, tokenizer id/revision, formatted-split checksums, and maximum sequence length. Its `languages` section reports mean, p50, p95, p99, max, totals, token rates, and over-budget counts. `pairing` reports exact root-matched coverage and aggregate/per-pair ratios by language and split. `training_workload` projects non-padding full tokens across configured epochs and explicitly excludes dynamic padding; it is a deterministic lower bound, not a billed-token estimate.

### Translation publication schemas

`sommelier.translation_summary.v2` records the accepted-row digest and
canonical publication identity, target-script/bidi policy, retry/drop counts,
source selection, clean implementation revision, translator model revision,
request digest, explicit interface, `trust_remote_code`, output decoder, and
postprocessing/audit schemas. Provider-backed summaries additionally record the
transport independently of the prompt family, the exact dated API snapshot,
provider SDK/request identity, explicit requested and returned service tier,
900-second request timeout, zero SDK retries, worker/chunk contract, CPU runtime, and nested
`sommelier.openai_provider_evidence.v2`. Its v2 accounting publishes both the
maximum canonical request-body UTF-8 byte proxy and the independently observed
maximum response `usage.input_tokens`; both must remain within the pinned
base-rate boundary. The dated snapshot is not a public
weight digest or a guarantee of byte-identical provider regeneration. The
request identity also pins the protected-
placeholder schema used by chat translators. For the generic instruction-chat
interface it additionally pins the v1 selected-tool semantic-context builder:
a unique case-sensitive exact gold-call-name match, name/description plus
sorted parameter name/type/description projection, canonical ASCII JSON with
HTML delimiters escaped, the 8,192-character fail-closed bound, and the inert
non-output/non-executable envelope. Missing or duplicate exact tool names are
checkpointed as `invalid_row`. TranslateGemma and MADLAD identities explicitly
pin semantic context to `none`; their structured and raw-source formats are
unchanged. The complete source-row digest binds the actual `tools` and
`answers` bytes, while the request digest binds the builder policy, so neither
a schema edit nor an old instruction-chat checkpoint can be silently reused.
The audit schema pins
boundary-matched extraction of direct values and comma-delimited components,
explicit code-like function names, source list/dict literals proven equivalent
to gold structures, the Hebrew/Latin-only unprotected-script policy, and
rejection of the Unicode replacement character as corrupt output.

`sommelier.openai_responses_provider_journal.v2` is the private durable replay
surface for the selected Hebrew provider. Every response, request-error, and
replay event has top-level `attribution` with a non-empty `source_id` and
positive `attempt`. Those producer-local fields are excluded from the provider
body and request hash. Identical bodies may coalesce, but each consumer receives
its own attributed replay record. Terminal row progress records
`accepted_attempt` for accepted output or `final_attempt` after an exhausted
drop. Responses are fsynced before the row pipeline receives them; the Modal
volume is committed at the 32-row chunk boundary. This is not exactly once: a
hard kill can lose the current uncommitted chunk, and a death after provider
acceptance but before receipt/fsync can repeat a billed request.

The raw journal contains decoded output and provider ids. It remains in the
durable producer artifacts and is not part of the public paired dataset. Its
content-free `sommelier.openai_responses_provider_journal_summary.v2` feeds the
nested provider evidence, which publishes the journal SHA-256, identity,
counts, usage, both input-bound maxima, and a cost calculated from the pinned
public price table. Full-publication validation binds the runtime ceiling to
that calculated value. The USD value is a local estimate, not an invoice,
observed billing, or provider account/project cap. `store=false` is recorded but is
not evidence of Zero Data Retention, and strict structured output is not a
semantic-accuracy guarantee.

For Hebrew full publication,
`sommelier.translation_semantic_review_template.v1` binds the full paired-row
corpus and summary to a deterministic 200-row sample selected before judgments.
It records the pinned independent Helsinki-NLP OPUS-MT Marian back-translator
and every locked machine input. A reviewer edits a copy's rubric fields; finalization verifies
all locked bytes against the untouched template and writes
`sommelier.translation_semantic_review.v1` with the non-native reviewer
boundary, decision digest, and zero-critical-error whole-publication gate. The
semantic sample cannot be used to remove a failed row and resample.

`sommelier.translation_publication_manifest.v1` is the bridge from local build
to immutable Hub commit. It binds the canonical paired rows and SHA-256 digests
of the translation summary, untouched machine template, and finalized semantic
review. A full remote run downloads all four provenance files from the paired dataset revision and
fails before preparation if any identity or gate differs. Direct
`--translation-run-id` staging is smoke-only. For provider translation, the
published summary contains the content-free provider evidence and raw-journal
digest; the raw journal itself stays out of the Hub commit.

### `sommelier.experiment_report.v1`

The Hebrew three-arm report binds base, the exact preregistered v1 English
adapter, and the canonical train-manifest-bound local v3 English+Hebrew adapter
to one independently verified evaluation identity. `data_provenance` traverses
the exact root rows, paired rows, translation summary, publication manifest,
untouched semantic template, finalized review, succeeded data manifest,
prepared splits, format manifest, formatted splits, tokenizer-tax manifest,
and adapter evaluation input. Every edge is an observed path, byte count, and
SHA-256 reference. It also records the requested and observed full cohort
counts, immutable dataset revisions, zero-critical semantic gate, and clean
source revision.

`arms` includes metrics and SHA-256 evidence for the formatted split,
generations, and evaluation report. `comparisons.v3_vs_v1` contains English and
Hebrew deltas plus paired-bootstrap contracts. `claims` stores the estimate,
interval, criterion, pass decision, and predeclared English margin; a
human-readable statement exists only for a passed gate. `approved_claims` and
`all_claims_passed` therefore cannot imply success when either bound fails.
The report finalizer itself must run from that same clean immutable source
revision; `preregistration.finalizer_source_code` records the check before any
outcome artifact is loaded.

Its `sovereign_tco_evidence` field has schema
`sommelier.sovereign_tco_evidence.v1`. The builder revalidates the resolved
config, tokenization/train/evaluation manifests, exact artifact paths and
hashes, adapter tree, runtime metadata, deterministic decoding, hardware
labels, and inference telemetry before joining them. It distinguishes observed
training/storage/timing values from the deterministic non-padding-token
projection. Cross-arm inference comparability additionally requires identical
observed runtime package versions. Currency cost is forbidden without observed billing evidence, and
full-fine-tuning savings are forbidden without a matched full-parameter arm;
both stay machine-readably unavailable otherwise.

### `sommelier.manifest.v1`

Stage manifest fields:

| Field | Type | Notes |
|-------|------|-------|
| `stage` | `data` \| `format` \| `tokenization` \| `train` \| `eval` \| `eval-base` \| `eval-adapter` \| `report` \| `serve` | pipeline evaluations use the two arm-specific names |
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

### `sommelier.release_preflight.v2`

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | str | exactly `sommelier.release_preflight.v2` |
| `created_at` | timezone-aware str | report creation time |
| `status` | `pass` \| `fail` | the report is written even when gates fail, provided the artifact root itself is a safe writable directory |
| `identity.config` | object | canonical normalized-config algorithm, SHA-256, and byte count |
| `identity.model` | object | exact base-model and tokenizer revisions plus immutability decisions |
| `identity.datasets` | list | ordered root/paired dataset ids, languages, revisions, and immutability decisions |
| `identity.source_code` | object | Git discovery mode, commit or explicit `unknown`, clean-state decision, and status digest; hidden index flags such as assume-unchanged or skip-worktree prevent a clean certification, and strict adapter publication requires an immutable commit and `working_tree_clean: true` |
| `identity.dependency_lock` | object | SHA-256 and byte count for working-tree `uv.lock` bytes that Git's configured clean filters map to the blob at the certified commit; unavailable outside that exact Git repository or on mismatch |
| `identity.artifact_tree` | object | canonical tree SHA-256, file/byte counts, certification status, a second content read that detects same-size mutation independently of timestamps, and the sole `release_preflight.json` exclusion |
| `gates` | list | each gate is `{name, status, evidence}` with status `pass`, `fail`, or `skip` |

The generic reader retains v1 for historical inspection. The adapter publisher
accepts only the closed v2 contract, requires all eight gates to pass, and
recomputes the config and exact on-disk bundle identities while also matching
the clean training source revision and training lock digest.

## ArtifactRef and the relative-path rule

Every input and output in a manifest is an `ArtifactRef`:

| Field | Type | Notes |
|-------|------|-------|
| `path` | str | POSIX path relative to the artifact root |
| `kind` | str | e.g. `dataset_split`, `formatted_split`, `generations`, `adapter_weights` |
| `schema_version` | str | empty for schema-less files such as adapter weights |
| `sha256` | str | checksum of the file bytes |
| `bytes` | int | file size |

Paths are always relative to the artifact root, so a run directory can be moved, archived, or published without breaking its own manifests. Building a ref for a path outside the artifact root raises `SchemaValidationError`; absolute paths never appear in input or output references (the `command` field records the argv verbatim, including any absolute paths you passed).

Writes validate a supplied artifact-root boundary before any filesystem mutation or writer call. The writer receives a path inside a cryptographically random private staging directory; its regular-file output is opened without following symlinks, copied to an exclusive mode-`0600` staging file, and read again to detect content changes without relying on timestamps. The staged file is flushed, atomically replaced into the final name, and the parent directory is flushed where supported. Ordinary failures clean up; a hard process stop can leave a private staging directory, but not a half-written final artifact.

## Run IDs

`create_run_id` produces `<UTC timestamp>-<8 hex chars>`, for example `20260702T091500Z-1a2b3c4d` (format `%Y%m%dT%H%M%SZ` plus the first 8 characters of a UUID4). Smoke pipeline runs get a `smoke-` prefix so a later full run can never overwrite smoke artifacts. When a [CLI command](cli.md) is pointed at a path inside a run and `--run-id` is omitted, the run ID is inferred from the path with the pattern `/runs/([^/]+)/`.
