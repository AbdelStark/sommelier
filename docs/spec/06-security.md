# 06 Security

- Status: Draft
- Target milestone: v1.0
- Primary RFC: [RFC-0011](../rfcs/RFC-0011-security-licensing-and-release-gates.md)

## Threat Model

Sommelier processes untrusted dataset rows and user-provided configuration. It also uses secrets to access gated model and tracking services. The main threats are:

- Malformed JSON causing parser or prompt injection failures.
- Tool schemas or queries leaking into logs in unexpected places.
- Secrets written to artifacts, reports, or failed manifests.
- Accidental publication of model artifacts without complying with base model and dataset licenses.
- Execution of arbitrary code from remote model repositories without an explicit trust decision.

## Trust Boundaries

| Boundary | Trusted input | Untrusted input |
|----------|---------------|-----------------|
| Config loader | Repository config schema | User-edited YAML values |
| Data preparation | Dataset revision identifier | Raw rows and JSON strings |
| Formatting | Validated `PreparedExample` | Tool descriptions and query text |
| Training | Local code and pinned dependencies | Model repository code unless explicitly allowed |
| Evaluation | Parser and metric code | Generated model output |
| Reporting | Manifests and metrics | Raw examples and generations |

## Secret Handling

Secrets are read only from the execution environment or remote secret store. They are never accepted in config files. The supported secret names are:

```text
HF_TOKEN
WANDB_API_KEY
```

The exact remote secret mapping is configured in the remote entrypoint, not in portable artifacts.

## License Gate

Before downloading gated model or dataset artifacts, Sommelier runs a preflight that verifies:

- The user has acknowledged the base model license.
- The dataset license is recorded in `licenses/THIRD_PARTY.md`.
- Derived model artifacts include required notices, including any "Built with Llama" notice required by the base model terms.
- The repository has a project code license.

The preflight fails with exit code 3 when a required acknowledgement cannot be verified.

## Dependency Policy

The base package must not import GPU or remote execution dependencies at module import time. Optional dependency groups isolate large or platform-specific packages.

`trust_remote_code=True` is forbidden by default. If a selected model requires it, the config must set:

```yaml
model:
  allow_remote_code: true
  remote_code_reason: "..."
```

The resolved config and report must include that decision.

## Data Privacy

The reference dataset is public, but Sommelier is intended to be adapted to private schemas. The code must support:

- Disabling raw generation retention.
- Redacting configured fields from reports.
- Keeping artifact roots outside the repository by default.

v1.0 does not implement multi-tenant access control.
