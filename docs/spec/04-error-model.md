# 04 Error Model

- Status: Draft
- Target milestone: v1.0
- Primary RFCs: [RFC-0008](../rfcs/RFC-0008-cli-and-python-public-api.md), [RFC-0011](../rfcs/RFC-0011-security-licensing-and-release-gates.md)

## Error Hierarchy

```python
class SommelierError(Exception):
    code: str
    exit_code: int
    hint: str | None

class UserInputError(SommelierError): ...
class ConfigError(UserInputError): ...
class SchemaValidationError(UserInputError): ...
class ArtifactNotFoundError(UserInputError): ...
class ExternalDependencyError(SommelierError): ...
class RemoteExecutionError(SommelierError): ...
class ResourceError(RemoteExecutionError): ...
class EvaluationError(SommelierError): ...
class InvariantViolation(SommelierError): ...
class SecurityPolicyError(SommelierError): ...
```

## Exit Codes

| Exit code | Error class | Meaning |
|-----------|-------------|---------|
| 0 | none | Success. |
| 2 | `UserInputError` | The user can fix config, paths, schema, or command flags. |
| 3 | `ExternalDependencyError` | Required external service, credential, package, or license acceptance is missing. |
| 4 | `RemoteExecutionError` | Remote resource, timeout, or out-of-memory failure. |
| 5 | `InvariantViolation` | The code observed a state the spec forbids. |

## Failure Modes

| Failure mode | Detection | Response |
|--------------|-----------|----------|
| Missing config file | CLI path check | Exit 2 with expected path. |
| Unknown config key | Strict config parser | Exit 2 with offending key path. |
| Invalid raw row JSON | Data validation | Drop row, count reason, write validation summary. |
| Empty split after filtering | Split validation | Exit 2 before training. |
| Duplicate query across splits | Split manifest check | Exit 5 and refuse artifacts. |
| Tokenizer template mismatch | Prompt fixture digest check | Exit 2 before training or evaluation. |
| GPU out of memory | Training exception mapping | Exit 4 with batch and sequence-length hint. |
| Generated text has no JSON | Parser status | Count failure; do not abort evaluation. |
| Generation artifact missing | Artifact registry check | Exit 2. |
| Secret appears in artifact | Redaction scanner | Exit 5 and mark run failed. |
| License acknowledgement missing | Preflight check | Exit 3 before dataset or model access. |

## Error Message Contract

Every CLI error prints:

```text
sommelier: <CODE>: <short message>
hint: <actionable next step>
```

Stack traces are hidden by default for expected `SommelierError` subclasses and shown only with `--debug`.

## Manifest Behavior on Failure

If a stage reaches Sommelier error handling after creating the run directory, it writes:

```python
class FailedStageManifest(StageManifest):
    status: Literal["failed"]
    error_code: str
    error_message: str
```

The failed manifest must not contain secrets, tokens, or raw exception messages from authentication libraries.

## Retry Policy

Sommelier does not silently retry model training or evaluation because retries can change costs and observability. The CLI may retry idempotent remote upload/download operations at most three times with exponential backoff and must log each retry.
