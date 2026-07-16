# Security

Sommelier is a pipeline you run from your own machine and your own Modal
account, not a hosted multi-tenant Sommelier service. Hebrew v3 dataset creation
does cross one additional service boundary: its one-time teacher sends bounded
translation requests from Modal to OpenAI. The realistic local threat is
quieter: a Hugging Face or OpenAI token, provider journal, or home directory
path ends up inside an artifact that is committed, shared, or published next to
an adapter. Most of the security machinery exists to make credential leaks fail
loudly and to keep raw provider records outside the public dataset. The
enforcement lives in [`sommelier/security.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/security.py) and [`sommelier/redaction.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/redaction.py).

## Secrets live in the environment, never in config

Config files are portable artifacts: they get committed, copied into every run directory as `config.resolved.yaml`, and shipped alongside results. So the config loader refuses to load a config that looks like it contains a secret. `validate_no_secrets` runs on the raw YAML mapping when a config is loaded, again on the validated config dump, and once more on the resolved config before `config.resolved.yaml` is written. A violation raises `SecurityPolicyError` (code `SOM006`, exit 5, see the [error reference](../reference/errors.md)).

The check has two parts:

- **Sensitive key names.** Any key matching `token`, `secret`, `password`, `api_key`, or `apikey` as a delimited word is rejected. A small allowlist (`dedupe_key`, `query_column`, `tools_column`, `answers_column`, `redact_fields`, `target_modules`) keeps legitimate field names from tripping it.
- **Token-shaped values.** Strings matching known credential formats are rejected: `hf_`, `sk-`, `ghp_`, and `xox[baprs]-` prefixes with minimum lengths.

The English reference workflow needs `HF_TOKEN` for Hugging Face downloads.
Locally it lives in `.env` (git-ignored; `.env.example` is committed); pipeline
and semantic-review entrypoints forward it as a dotenv-backed Modal secret.
Hebrew v3 translation additionally requires two exact named Modal secrets:
`openai-api-key` containing `OPENAI_API_KEY` and `huggingface-read-token`
containing `HF_TOKEN`. The OpenAI producer does not silently fall back to the
dotenv secret. The [remote guide](../guides/remote-execution.md) gives the
provisioning commands. None of these values is written to artifacts. Other
dotenv values, such as a wandb key for opt-in tracking, travel only through
entrypoints that explicitly mount the dotenv-backed secret.

## Redaction at write time

Every log message and string log field passes through `redact_text` before it reaches disk, as do the evidence strings in the release preflight report:

- Token-shaped substrings (`hf_...`, `sk-...`, `ghp_...`, `xox...`) become `[redacted]`.
- The current value of any environment variable whose name contains `TOKEN`, `KEY`, `SECRET`, or `PASSWORD` (and is at least 8 characters long) becomes `[redacted]`.
- The home directory path is shortened to `~`.

Failed stage manifests are stricter. If the error message contains `hf_`, `sk-`, `ghp_`, `token`, `secret`, or `password` (case-insensitive), the entire message collapses to `stage failed; details redacted`; otherwise it is truncated to 500 characters. The blunt rule is deliberate: a library that embeds an auth header in its exception text cannot smuggle it into a manifest, at the cost of occasionally over-redacting a harmless message.

## The artifact secret scanner

Write-time redaction covers what Sommelier writes. The scanner covers text artifacts under the artifact root, regardless of who wrote them. It inspects every `.json`, `.jsonl`, `.md`, `.txt`, `.yaml`, and `.yml` file. JSON is parsed structurally so findings carry a field path; JSONL is scanned line by line with line numbers; everything else is scanned as text. Token-shaped JSON object keys are scanned as well as values. Adapter publication additionally parses and scans the `__metadata__` object inside `adapter_model.safetensors` instead of treating the binary container as opaque.

| Finding kind | Trigger |
|--------------|---------|
| `sensitive_key` | A JSON key matching the sensitive-name pattern |
| `secret_value` | A token-shaped string (`hf_` / `sk-` / `ghp_` / `xox` prefixes) |
| `sensitive_env_value` | The current value of a `TOKEN`/`KEY`/`SECRET`/`PASSWORD` environment variable appearing in content |
| `home_path` | The current home directory path appearing in content |

The scanner is a release preflight gate (`artifact_secret_scan`): any finding fails closed with `SecurityPolicyError` and exit 5, and publishing stops. Preflight v2 scans and hashes each file through one coherent inspection pass, so a report cannot claim a clean scan while certifying different bytes. The full gate list is on the [licensing page](licensing.md). Note that the env-value and home-path checks compare against the *current* environment, so the scanner catches your secrets and your paths; it is a hygiene gate for artifacts you produced, not a general-purpose secret detector.

Publication adds a second mutation boundary. Each public command first copies
the caller's inputs into a private mode-0700 snapshot and validates that copy;
it then materializes a read-only upload snapshot whose inode, size, and SHA-256
are checked before and after the commit call. The receipt must live outside the
source bundle and is reserved before any Hub access. These constraints keep a
concurrent edit to the original bundle from changing the bytes that were
validated and submitted. The returned immutable revision is still downloaded
and hash-checked before the receipt can become `verified`.

## Trust boundaries

The pipeline treats external content as data, never as instructions or code:

| Input | Trust | Handling |
|-------|-------|----------|
| Raw dataset rows | Untrusted JSON strings | Validated field by field; failures are dropped with a recorded reason, never repaired ([data policy](../concepts/data.md)) |
| Queries and tool schemas | Untrusted content | Instruction-chat translation places only the exact selected tool's name/description and parameter name/type/description projection in escaped canonical JSON, labels it inert/non-output/non-executable, and executes nothing; other translator interfaces receive no schema context |
| Pipeline model repository code | Untrusted by default | `trust_remote_code` is false in formatting, training, and evaluation unless the config records an explicit reason |
| Hebrew v3 teacher request | Untrusted public source text sent to an external API | Exact dated Responses snapshot and Flex tier; protected literals are placeholder-masked; only the selected tool's bounded projection is included; tools/answers are never translated; returned model/tier, usage, and audits are recorded |
| Model output | Untrusted text | The parser extracts the first balanced JSON span with no repair and executes nothing ([evaluation method](../concepts/evaluation.md)) |

Trusting remote model code is an explicit, recorded decision. For pipeline
model and tokenizer loads, `model.allow_remote_code` defaults to false; setting
it to true fails config validation unless `model.remote_code_reason` states
why. Both fields appear in `config.resolved.yaml`, and every pipeline load
passes the configured value through as `trust_remote_code`.

Translation is a separate tool and provider boundary. The Hebrew v3 command
pins the dated API identity `gpt-5.5-2026-04-23`, explicit Flex service,
instruction-chat request, OpenAI SDK 2.45.0, and the clean implementation
revision. It runs in a CPU-only Modal image and rejects a returned model or tier
that differs from the request. A dated API snapshot is not a public weight
digest, a sandbox, or a guarantee of byte-identical regeneration.

Protected literals are deterministic placeholders before provider access and
are restored and audited after the response. This prevents those exact values
from entering the provider request, but the source query and the selected
tool's name/description and parameter name/type/description projection still
leave Modal and are processed by OpenAI. Requests use strict JSON Schema,
`store=false`, background mode disabled, truncation disabled, reasoning effort
`none`, a 900-second timeout, zero SDK retries, and a stable non-PII safety
identifier. SDK retries remain zero so every repeated request is a visible
source-id/attempt event rather than a hidden transport retry. `store=false`
prevents this workflow from retrieving a stored response; it is not a claim of
Zero Data Retention. Strict structure and script/literal audits do not prove
semantic accuracy or resistance to every prompt injection.

The raw provider journal contains decoded output, response/request ids, usage,
and source-id/attempt attribution. It is fsynced for replay and kept in durable
producer artifacts, not published in the paired dataset. The translation
summary exposes a content-free v2 aggregate and the journal SHA-256 instead.
This is not an exactly-once protocol: a process death after provider acceptance
but before response receipt/fsync can repeat a billed request, and a hard kill
can lose the current Modal chunk before its volume commit.

Tool descriptions remain prompt-injection input even after projection. The
instruction-chat envelope therefore treats their canonical JSON as untrusted
semantic hints only: literal `<`, `>`, and `&` characters are Unicode-escaped
so a description cannot close the context delimiter, control characters stay
JSON-escaped, and the model is told not to follow, execute, translate, or emit
the metadata. Selection uses one unique case-sensitive exact gold call name;
missing or duplicate matches fail closed as an `invalid_row`. The builder
never reads gold arguments and never includes schema defaults, examples, or
enums. This is prompt-boundary hardening, not a sandbox or a guarantee that a
model will resist every semantic injection; normal output audits still apply.

## Domain-specific redaction

If your data itself is sensitive (tool schemas naming internal endpoints, for example), `report.redact_fields` takes a list of field names and replaces their values with `[redacted]` anywhere in the JSON tree of the evaluation and comparison reports. It applies to the reports only: raw generations in `generations.jsonl` and decoded translations in the raw provider journal are stored verbatim, so treat the artifact root accordingly. Do not use the hosted teacher for data you are not authorized to send to that provider. See the [configuration reference](../reference/configuration.md).

## What Sommelier does not provide

Stating the gaps is part of the posture:

- **No multi-tenant isolation or access control.** Protecting the artifact root is filesystem permissions, nothing more.
- **No authentication on the local server.** `sommelier serve adapter` binds `127.0.0.1` by default and has no auth; it exists to inspect the adapter, not to serve traffic.
- **The remote vLLM endpoint is open unless you close it.** Setting `SOMMELIER_SERVE_API_KEY` in `.env` enables Bearer auth; without it the deployment logs a warning and accepts unauthenticated requests. Details in the [serving guide](../guides/serving.md).
- **No sandboxing of remote model code.** `allow_remote_code: true` means "I trust this model repository"; nothing contains it after that.
- **No Zero Data Retention guarantee for the Hebrew teacher.** `store=false` is
  recorded, but provider data controls and account eligibility are separate.
- **Pattern-based detection only.** Redaction and scanning target known token shapes and your current environment values. A secret pasted in an unrecognized format will not be caught.
