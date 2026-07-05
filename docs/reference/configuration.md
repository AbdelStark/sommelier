# Configuration reference

One YAML file drives every stage of the pipeline. The schema lives in [`sommelier/config.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/config.py) as pydantic models with `extra="forbid"` on every section, so an unknown or misspelled key anywhere in the file is a `ConfigError` (SOM201, exit 2), never a silently ignored setting. This page lists every field, its type, its default in code, and what it controls.

Three rules apply before any field is read:

- **Secrets are rejected.** `load_config` scans both the raw YAML and the validated dump for secret-shaped values and sensitive key names. A hit raises `SecurityPolicyError` (SOM006, exit 5), because config files end up inside published artifacts. Patterns and allowlists are in [Security](../project/security.md).
- **`artifact_root` must be relative.** An absolute path fails validation. The root is resolved against the directory containing the config file, so a checkout can move without editing the config, and artifacts cannot be aimed outside the project by accident.
- **The resolved config is an artifact.** Every run writes the validated config back out as `config.resolved.yaml` at the run root and records its SHA-256 digest as `config_sha256` in every stage manifest and evaluation report. The [comparison gate](../concepts/determinism.md) refuses to compare two evaluation reports unless they carry the same digest. Editing one field between the base eval and the adapter eval does not skew the numbers; it prevents the report from existing.

To validate a file without running anything:

```bash
uv run sommelier config validate --config examples/config.smoke.yaml
```

Add `--write-resolved <dir>` to also write `config.resolved.yaml` and inspect exactly what a run would record. See the [CLI reference](cli.md) for the full command surface.

## Why so many fields are pinned

Several fields are typed as a `Literal` with exactly one allowed value: `data.dedupe_key`, `formatting.template_policy`, `formatting.target_format`, `train.scheduler`, `train.quantization`, `train.compute_dtype`, `eval.split`, and `eval.parser_version`. These are the parts of the reference contract that the published evidence depends on. Pinning them in the type system means loosening one is a code change reviewed in the open, not a config edit that quietly changes what the numbers mean while the file still looks like a Sommelier config. The reasoning per field is in [Design decisions](../concepts/design-decisions.md).

## Top level

The file must start with the schema declaration:

```yaml
schema_version: sommelier.config.v2
```

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `schema_version` | `"sommelier.config.v2"` | required | Identifies the config schema. Any other value fails validation, with one exception below. |

Every section below is required except `tracking`, which defaults to disabled when omitted.

A file declaring `sommelier.config.v1` still loads. The loader upgrades it in memory and emits a `DeprecationWarning`: the single `dataset` section becomes the one `en` entry under `datasets`, and `train.languages` and `eval.slices` resolve to English only. The resolved config written into the run directory is always the v2 form, so two runs of the same settings produce the same `config_sha256` whether the file on disk was v1 or v2.

## `project`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `name` | `str` | required | A label for the configuration. It travels in `config.resolved.yaml`; no stage branches on it. |
| `artifact_root` | `Path` | required | Root directory for all run artifacts, relative to the config file's directory. Absolute paths are rejected. |
| `seed` | `int` | `42` | The single seed for the whole run: split shuffling in [data prepare](../concepts/data.md), the trainer seed in [training](../concepts/training.md), and the `seed` field recorded in manifests and reports. |

## `model`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `base_model_id` | `str` | required | Hugging Face model id of the base model, e.g. `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. |
| `base_model_revision` | `str` | required | Revision to load model weights from. Required so the run records what it actually used. |
| `tokenizer_revision` | `str` | required | Revision to load the tokenizer (and its chat template) from. Pinned separately because the prompt rendering depends on it. |
| `allow_remote_code` | `bool` | `false` | Passed as `trust_remote_code` everywhere the tokenizer or model is loaded (formatting, evaluation, training). |
| `remote_code_reason` | `str \| null` | `null` | Required whenever `allow_remote_code` is `true`; validation fails without it. |

The `allow_remote_code` and `remote_code_reason` coupling exists so that executing repository code from the Hub is never a one-character change. Turning it on forces you to write down why, and the reason ships inside the resolved config with everything else.

## `datasets`

A non-empty list of per-language dataset sources. Each entry:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `language` | `str` | required | Two-letter lowercase ISO 639-1 code (`en`, `fr`). At most one source per language. |
| `dataset_id` | `str` | required | Source dataset id, e.g. `Salesforce/xlam-function-calling-60k`. |
| `dataset_revision` | `str` | required | Dataset revision; stamped into every prepared example as `source_revision`. |
| `query_column` | `str` | `"query"` | Column holding the user query, read when raw rows are exported from the source dataset. |
| `tools_column` | `str` | `"tools"` | Column holding the tool schemas as a JSON string. |
| `answers_column` | `str` | `"answers"` | Column holding the gold tool calls as a JSON string. |
| `source_id_column` | `str \| null` | `null` | When set, names the source-dataset column holding the root row reference; like the other column fields it is read when raw rows are exported, mapping into the canonical `source_example_id` raw-row field. When null, this is the root source. |

Exactly one source must omit `source_id_column`: the root source, which gets independent split assignment during data prepare. Every other source is paired, row by row, to root rows through the column this field names. The pairing exists so that a translated variant of a query can never land in a different split than its original, which would leak test content into training. How prepare enforces this is described in [Data policy](../concepts/data.md).

## `data`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `n_train` | `int` | `15000` | Training split size. |
| `n_validation` | `int` | `1000` | Validation split size. |
| `n_test` | `int` | `1000` | Held-out test split size; the only split [evaluation](../concepts/evaluation.md) reads. |
| `min_query_chars` | `int` | `10` | Rows with shorter queries are dropped (reason `query_too_short`). |
| `max_query_chars` | `int` | `2000` | Rows with longer queries are dropped (reason `query_too_long`). |
| `dedupe_key` | `"normalized_query"` | pinned | Deduplication key: the SHA-256 of the query after casefolding, stripping, and whitespace collapsing. Duplicates drop with reason `duplicate_query`. |

One model-level validator guards this section: the three split counts must be positive, `min_query_chars` must be positive, and `max_query_chars` must be strictly greater than `min_query_chars`. If the surviving rows cannot fill `n_train + n_validation + n_test`, data prepare fails rather than shrinking a split. The full drop-reason taxonomy is in [Data policy](../concepts/data.md).

## `formatting`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `system_prompt` | `str` | required | Prepended to the system message, followed by the canonical JSON of the available tools. |
| `template_policy` | `"tokenizer_chat_template"` | pinned | Prompts are rendered by the tokenizer's own chat template, never by a hand-rolled format string. |
| `target_format` | `"json_tool_call"` | pinned | The training target is the canonical JSON of the gold call and nothing else. |

## `train`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `epochs` | `int` | `2` | Training epochs over the train split. |
| `per_device_batch_size` | `int` | `8` | Per-device micro-batch size. |
| `gradient_accumulation_steps` | `int` | `2` | Accumulation steps; effective batch is the product of the two. |
| `learning_rate` | `float` | `2e-4` | Peak learning rate. |
| `scheduler` | `"cosine"` | pinned | Learning rate schedule. |
| `warmup_ratio` | `float` | `0.03` | Fraction of steps spent warming up. |
| `max_sequence_length` | `int` | `2048` | Token budget per rendered example. Truncation that would remove every target token is an error, never a silent trim. |
| `quantization` | `"nf4-4bit"` | pinned | Base model quantization for QLoRA. |
| `compute_dtype` | `"bfloat16"` | pinned | Compute dtype during training. |
| `lora_rank` | `int` | `16` | LoRA rank. |
| `lora_alpha` | `int` | `32` | LoRA alpha. |
| `lora_dropout` | `float` | `0.05` | LoRA dropout. |
| `target_modules` | `list[str]` | required | Projection modules the adapter attaches to. No default: the choice is model-specific and belongs in the record. |
| `languages` | `list[str]` | all configured | Languages whose examples feed training. Empty or omitted resolves to every language under `datasets`, in configuration order; the resolved config always records the explicit list. Every entry must name a configured source. |

None of these are ever adjusted at runtime. If training runs out of GPU memory, the failure is a `ResourceError` whose hint quotes your current `per_device_batch_size`, `gradient_accumulation_steps`, `max_sequence_length`, and `remote.gpu` and tells you to change one of them yourself. See [Training](../concepts/training.md) and [Errors](errors.md).

## `eval`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `split` | `"test"` | pinned | Evaluation only ever reads the held-out test split. |
| `slices` | `list[str]` | `["en"]` | Language slices to evaluate. Every entry must name a configured source under `datasets`; duplicates are rejected. |
| `temperature` | `float` | `0.0` | Must be exactly `0.0` at run time; anything else raises `EvaluationError`. |
| `do_sample` | `bool` | `false` | Must be `false` at run time; `true` raises `EvaluationError`. |
| `max_new_tokens` | `int` | `512` | Generation budget per prompt. Must be positive. |
| `parser_version` | `"sommelier.parser.v1"` | pinned | The parser identity recorded in every report; one of the fields the comparison gate matches on. |

`temperature` and `do_sample` are plain types rather than Literals because their values are recorded in the `decoding` block of every evaluation report, which the [comparison gate](../concepts/determinism.md) checks field by field. The eval stage still refuses to run with non-deterministic settings; it errors instead of coercing, so the config you wrote is always the config that ran.

## `remote`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `enabled` | `bool` | `true` | Declares that this config targets remote execution. Current stage code reads the GPU and timeouts from this section but does not branch on this flag. |
| `gpu` | `str` | required | Modal GPU type, e.g. `A10G` or `L40S`. Recorded in runtime metadata and quoted in out-of-memory hints. |
| `data_timeout_seconds` | `int` | `1800` | Time budget for the remote data stage. |
| `train_timeout_seconds` | `int` | `14400` | Time budget for training. Exceeding it is a `ResourceError` that names this field. |
| `eval_timeout_seconds` | `int` | `7200` | Time budget for each evaluation stage. |

Each remote stage gets exactly its own budget; an unknown stage name fails instead of inheriting another stage's timeout. How these values reach Modal is covered in [Remote execution](../guides/remote-execution.md).

## `report`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `retain_raw_generations` | `bool` | `true` | Declared in the schema; the v1 pipeline always writes raw generations regardless, because the evidence requires them. |
| `redact_fields` | `list[str]` | `[]` | Key names to replace with `[redacted]` anywhere in the evaluation and comparison reports before they are written. |

## `tracking`

The only optional section. Omitting it is equivalent to writing `enabled: false`.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `enabled` | `bool` | `false` | Opt-in experiment tracking. Disabled is a strict no-op; nothing is imported. |
| `provider` | `"wandb"` | pinned | The only supported provider. Enabling it without `wandb` installed is an `ExternalDependencyError`. |
| `project` | `str` | `"sommelier"` | Tracking project name. The run URL, when one exists, is recorded in the run manifest. |

## The example configs

Two working configs ship in [`examples/`](https://github.com/AbdelStark/sommelier/blob/main/examples), plus [`config.invalid.yaml`](https://github.com/AbdelStark/sommelier/blob/main/examples/config.invalid.yaml), which exists to demonstrate a validation failure (`n_train: -100`). Both working configs share the same model (`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`), dataset (`Salesforce/xlam-function-calling-60k`), seed, system prompt, learning rate, quantization, and target modules. They differ only in scale:

| Setting | [`config.smoke.yaml`](https://github.com/AbdelStark/sommelier/blob/main/examples/config.smoke.yaml) | [`config.full.yaml`](https://github.com/AbdelStark/sommelier/blob/main/examples/config.full.yaml) |
|---------|-----------------|----------------|
| Purpose | Prove the chain end to end, cheaply | The reference run configuration |
| Splits (train/val/test) | 100 / 20 / 20 | 15,000 / 1,000 / 1,000 |
| Epochs | 1 | 2 |
| Batch · accumulation | 2 · 1 | 4 · 4 (effective batch 16) |
| `max_sequence_length` | 2048 | 4096 |
| LoRA rank · alpha | 8 · 16 | 16 · 32 |
| `eval.max_new_tokens` | 256 | 512 |
| GPU | A10G | L40S |
| Timeouts (data/train/eval) | 900 / 3600 / 1800 s | 1800 / 28800 / 28800 s |
| `tracking` section | omitted (disabled by default) | explicit, disabled |

`config.full.yaml` is the configuration behind the published [reference run](../results/reference-run.md); the [reproduction guide](../guides/reproduction.md) walks through running both. Start every change from one of these files: the validation error hints point back to `examples/config.smoke.yaml` for a reason.
