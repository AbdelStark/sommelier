# RFC-0011: Security, Licensing, and Release Gates

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v1.0

## Summary

Sommelier gates releases on secret hygiene, license acknowledgement, dependency discipline, local tests, remote smoke validation, and reference-run artifacts. The project code license is MIT unless maintainers replace it before v1.0.

## Motivation

The PRD calls out model and dataset license obligations and asks for reproducibility. The security and release specs require trust boundaries, secret handling, and documented claims. A release gate prevents publishing an incomplete reference result or noncompliant derived artifact.

## Goals

- Keep secrets out of config, logs, and artifacts.
- Record third-party model, dataset, and package obligations.
- Add a project `LICENSE` and third-party attribution file.
- Require local tests, remote smoke, and full reference artifacts before v1.0.
- Ensure public claims are backed by commands and artifacts.

## Non-Goals

- Legal advice.
- Automated interpretation of external licenses.
- Production security certification.
- Signed model artifact distribution in v1.0.

## Proposed Design

### License Files

```text
LICENSE
licenses/THIRD_PARTY.md
```

`THIRD_PARTY.md` records base model, dataset, and major runtime package obligations. The release report links to it.

### Preflight

```python
def run_release_preflight(config: SommelierConfig, artifact_root: Path) -> PreflightReport: ...
```

Checks:

- Project license file exists.
- Third-party attribution file exists.
- Base model license acknowledgement is configured or verified.
- Dataset license entry is present.
- Required notice text for derived artifacts is present.
- No configured artifact contains token-like secrets.
- Dependency lock exists for the release environment.

### Release Checklist

```python
class ReleaseGate(TypedDict):
    name: str
    status: Literal["pass", "fail", "skip"]
    evidence: str
```

The checklist is written to `release_preflight.json`.

### Claim Review

The release report may state only:

- The configured adapter improved or did not improve specified metrics on the configured held-out split.
- The run used the recorded hardware, dependencies, and dataset revision.
- The observed cost and runtime for that run.

It may not state broad reliability, generalization, or production readiness.

## Alternatives Considered

- Leave licensing to README prose. Rejected because release gates need machine-checkable evidence.
- Publish adapters by default. Rejected because derived artifact obligations must be confirmed first.
- Skip remote smoke until full run. Rejected because smoke failures are cheaper to diagnose.

## Drawbacks

- License preflight cannot replace maintainer review.
- Release gates add work before the first public result.
- MIT may not be the final license if maintainers choose differently.

## Migration / Rollout

1. Add `LICENSE` and `licenses/THIRD_PARTY.md`.
2. Add secret scanner and preflight command.
3. Add CI for local tests, lint, type check, and docs link checks.
4. Add remote smoke instructions.
5. Require full reference report before v1.0 tagging.

## Testing Strategy

- Unit-test secret scanner on synthetic values.
- Unit-test missing license and attribution failures.
- Unit-test claim-review fixtures for disallowed language.
- CI-test local gates.
- Manually verify remote smoke and full reference run evidence before v1.0.

## Open Questions

None for v1.0.

## References

- [06-security](../spec/06-security.md)
- [09-release-and-versioning](../spec/09-release-and-versioning.md)
