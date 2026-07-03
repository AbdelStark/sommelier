# Errors and exit codes

Failure behavior is part of Sommelier's contract. Every failure surfaces as a subclass of `SommelierError` with a stable error code and a stable exit code, defined in one place: [`sommelier/errors.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/errors.py). The error code names the kind of failure; the exit code says whose fault it is, which is the thing a script or a CI job actually needs to branch on.

## Message format

Errors print to stderr as one line, sometimes followed by a hint:

```text
sommelier: SOM201: config file not found: examples/missing.yaml
hint: Pass --config with the path to a Sommelier YAML config.
```

Any exception that is not a `SommelierError` is caught at the CLI boundary and reported as `sommelier: SOM000: unexpected error: <message>` with exit 5. Passing the global `--debug` flag (before the subcommand) additionally prints the full Python traceback for either case. See the [CLI reference](cli.md).

## Exit codes

| Exit | Whose fault | Your move |
|------|-------------|-----------|
| 0 | Nobody's | Nothing, it worked |
| 2 | Your input: flags, config, missing or malformed artifacts | Fix the input and rerun |
| 3 | Your environment: an optional dependency or external requirement is missing | Install the named extra or satisfy the requirement |
| 4 | Resources or remote execution: out of memory, timeouts, evaluation aborts | Change the config fields named in the hint, or rent a larger GPU |
| 5 | Sommelier: a bug or a broken invariant | [File an issue](https://github.com/AbdelStark/sommelier/issues) |

The split between 2 and 3 is deliberate: a wrong config field and a missing CUDA stack are different problems with different fixes, and a retry loop should treat them differently. Exit 5 means the pipeline detected a state that its own design says must be impossible; retrying will not help, and the maintainers want to know.

## The hierarchy

```text
SommelierError                     SOM000   exit 5
├── UserInputError                 SOM002   exit 2
│   ├── ConfigError                SOM201   exit 2
│   ├── SchemaValidationError      SOM202   exit 2
│   └── ArtifactNotFoundError      SOM203   exit 2
├── ExternalDependencyError        SOM003   exit 3
├── RemoteExecutionError           SOM004   exit 4
│   └── ResourceError              SOM401   exit 4
├── EvaluationError                SOM005   exit 4
├── InvariantViolation             SOM005   exit 5
└── SecurityPolicyError            SOM006   exit 5
```

| Class | Code | Exit | Raised when |
|-------|------|------|-------------|
| `SommelierError` | SOM000 | 5 | Base class; also the label the CLI gives unexpected exceptions |
| `UserInputError` | SOM002 | 2 | Bad flag combinations (`--adapter` with `--model base`), a missing raw input file, or `report compare` pointed at a directory without a resolved config |
| `ConfigError` | SOM201 | 2 | Config file missing, YAML that does not parse, or fields that fail schema validation |
| `SchemaValidationError` | SOM202 | 2 | A record with a missing or unsupported `schema_version`, or an artifact path that escapes the artifact root |
| `ArtifactNotFoundError` | SOM203 | 2 | A stage input that does not exist, typically a split the preceding stage never produced |
| `ExternalDependencyError` | SOM003 | 3 | An optional extra is not installed (torch and friends for training, cudf for `--gpu`, uvicorn for serving, wandb for tracking), or a release gate other than the secret scan fails |
| `RemoteExecutionError` | SOM004 | 4 | The classification for remote execution failures; in the current codebase it surfaces only through its subclass `ResourceError` |
| `ResourceError` | SOM401 | 4 | Training out of GPU memory, or training past its time budget |
| `EvaluationError` | SOM005 | 4 | A non-deterministic decoding config, a generation count that does not match the test split, or a comparison gate rejection |
| `InvariantViolation` | SOM005 | 5 | A prompt digest mismatch between formatting and evaluation, a broken prompt-target boundary, mixed decoding configs in one generations file, or a non-finite float in logs or metrics |
| `SecurityPolicyError` | SOM006 | 5 | Secret material detected in a config, manifest, report, or the artifact tree |

`ResourceError` deserves a note because it embodies a policy: Sommelier never retries with silently altered settings. An out-of-memory failure comes back with a hint naming `train.per_device_batch_size`, `train.gradient_accumulation_steps`, `train.max_sequence_length`, and `remote.gpu` with their current values; a timeout names `remote.train_timeout_seconds`. You change the config, the change lands in the resolved config digest, and the record stays honest.

## One code, two exit codes

`EvaluationError` and `InvariantViolation` both carry SOM005 but exit with 4 and 5 respectively. This is a known wart in the v1 contract: the code alone cannot tell you whether evaluation hit a fixable condition or the pipeline broke one of its own guarantees. Branch on the exit code, not the error code, when scripting against Sommelier.

## Failure evidence in manifests

The stage-manifest schema reserves a failed shape: `status: "failed"` plus `error_code` and `error_message` fields, with the message redacted at build time. If it contains any of `hf_`, `sk-`, `ghp_`, `token`, `secret`, or `password` (case-insensitive), the entire message is replaced with `stage failed; details redacted`; otherwise it is truncated to 500 characters, and the manifest passes the same no-secrets validator as every other artifact. In the current pipeline, a failing stage propagates its error to the CLI, which prints it and exits with the codes above, rather than writing a failed manifest; the absence of a stage's `succeeded` manifest is itself the failure evidence. Manifest structure is documented in [Artifacts and schemas](artifacts.md), and the wider redaction policy in [Security](../project/security.md).

For the full picture of why stages fail loudly instead of degrading, see [The pipeline](../concepts/pipeline.md) and [Design decisions](../concepts/design-decisions.md).
