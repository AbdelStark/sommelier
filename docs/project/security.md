# Security

Sommelier is a pipeline you run on your own machine and your own Modal account, not a service. There is no server to harden and no tenant to isolate. The realistic threat is quieter: a Hugging Face token or a home directory path ends up inside an artifact, and that artifact gets committed, shared, or published next to an adapter. Most of the security machinery exists to make that class of leak fail loudly before anything leaves your disk. The enforcement lives in [`sommelier/security.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/security.py) and [`sommelier/redaction.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/redaction.py).

## Secrets live in the environment, never in config

Config files are portable artifacts: they get committed, copied into every run directory as `config.resolved.yaml`, and shipped alongside results. So the config loader refuses to load a config that looks like it contains a secret. `validate_no_secrets` runs on the raw YAML mapping when a config is loaded, again on the validated config dump, and once more on the resolved config before `config.resolved.yaml` is written. A violation raises `SecurityPolicyError` (code `SOM006`, exit 5, see the [error reference](../reference/errors.md)).

The check has two parts:

- **Sensitive key names.** Any key matching `token`, `secret`, `password`, `api_key`, or `apikey` as a delimited word is rejected. A small allowlist (`dedupe_key`, `query_column`, `tools_column`, `answers_column`, `redact_fields`, `target_modules`) keeps legitimate field names from tripping it.
- **Token-shaped values.** Strings matching known credential formats are rejected: `hf_`, `sk-`, `ghp_`, and `xox[baprs]-` prefixes with minimum lengths.

The one secret the reference workflow needs is `HF_TOKEN`, for Hugging Face downloads. Locally it lives in `.env` (git-ignored; `.env.example` is committed); [remote execution](../guides/remote-execution.md) forwards it as a dotenv-backed Modal secret, and it is never written to artifacts. Anything else you put in `.env`, such as a wandb key for opt-in tracking, travels the same way.

## Redaction at write time

Every log message and string log field passes through `redact_text` before it reaches disk, as do the evidence strings in the release preflight report:

- Token-shaped substrings (`hf_...`, `sk-...`, `ghp_...`, `xox...`) become `[redacted]`.
- The current value of any environment variable whose name contains `TOKEN`, `KEY`, `SECRET`, or `PASSWORD` (and is at least 8 characters long) becomes `[redacted]`.
- The home directory path is shortened to `~`.

Failed stage manifests are stricter. If the error message contains `hf_`, `sk-`, `ghp_`, `token`, `secret`, or `password` (case-insensitive), the entire message collapses to `stage failed; details redacted`; otherwise it is truncated to 500 characters. The blunt rule is deliberate: a library that embeds an auth header in its exception text cannot smuggle it into a manifest, at the cost of occasionally over-redacting a harmless message.

## The artifact secret scanner

Write-time redaction covers what Sommelier writes. The scanner covers everything under the artifact root, regardless of who wrote it. `scan_artifact_tree` walks the tree and inspects every `.json`, `.jsonl`, `.md`, `.txt`, `.yaml`, and `.yml` file. JSON is parsed structurally so findings carry a field path; JSONL is scanned line by line with line numbers; everything else is scanned as text.

| Finding kind | Trigger |
|--------------|---------|
| `sensitive_key` | A JSON key matching the sensitive-name pattern |
| `secret_value` | A token-shaped string (`hf_` / `sk-` / `ghp_` / `xox` prefixes) |
| `sensitive_env_value` | The current value of a `TOKEN`/`KEY`/`SECRET`/`PASSWORD` environment variable appearing in content |
| `home_path` | The current home directory path appearing in content |

The scanner is a release preflight gate (`artifact_secret_scan`): any finding fails closed with `SecurityPolicyError` and exit 5, and publishing stops. The full gate list is on the [licensing page](licensing.md). Note that the env-value and home-path checks compare against the *current* environment, so the scanner catches your secrets and your paths; it is a hygiene gate for artifacts you produced, not a general-purpose secret detector.

## Trust boundaries

The pipeline treats external content as data, never as instructions or code:

| Input | Trust | Handling |
|-------|-------|----------|
| Raw dataset rows | Untrusted JSON strings | Validated field by field; failures are dropped with a recorded reason, never repaired ([data policy](../concepts/data.md)) |
| Queries and tool schemas | Untrusted content | Rendered into prompts as data; nothing in them is executed |
| Model repository code | Untrusted by default | `trust_remote_code` is false in every tokenizer and model load unless configured otherwise |
| Model output | Untrusted text | The parser extracts the first balanced JSON span with no repair and executes nothing ([evaluation method](../concepts/evaluation.md)) |

Trusting remote model code is an explicit, recorded decision. `model.allow_remote_code` defaults to false; setting it to true fails config validation unless `model.remote_code_reason` states why. Both fields appear in `config.resolved.yaml`, so every run records whether that trust was granted and on what grounds. Every tokenizer and model load passes the configured value straight through as `trust_remote_code`.

## Domain-specific redaction

If your data itself is sensitive (tool schemas naming internal endpoints, for example), `report.redact_fields` takes a list of field names and replaces their values with `[redacted]` anywhere in the JSON tree of the evaluation and comparison reports. It applies to the reports only: raw generations in `generations.jsonl` are stored verbatim, so treat the artifact root accordingly. See the [configuration reference](../reference/configuration.md).

## What Sommelier does not provide

Stating the gaps is part of the posture:

- **No multi-tenant isolation or access control.** Protecting the artifact root is filesystem permissions, nothing more.
- **No authentication on the local server.** `sommelier serve adapter` binds `127.0.0.1` by default and has no auth; it exists to inspect the adapter, not to serve traffic.
- **The remote vLLM endpoint is open unless you close it.** Setting `SOMMELIER_SERVE_API_KEY` in `.env` enables Bearer auth; without it the deployment logs a warning and accepts unauthenticated requests. Details in the [serving guide](../guides/serving.md).
- **No sandboxing of remote model code.** `allow_remote_code: true` means "I trust this model repository"; nothing contains it after that.
- **Pattern-based detection only.** Redaction and scanning target known token shapes and your current environment values. A secret pasted in an unrecognized format will not be caught.
