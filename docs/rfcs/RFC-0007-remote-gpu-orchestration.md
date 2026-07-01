# RFC-0007: Remote GPU Orchestration

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v0.3

## Summary

Sommelier keeps stage logic in importable Python modules and uses remote GPU functions only as wrappers around those modules. The remote layer provides images, volumes, secrets, GPU selection, and timeouts, but it must not define separate business logic.

## Motivation

The PRD requires a single-GPU remote execution path and notes that data preparation, training, evaluation, and optional serving use different dependency stacks. The architecture spec requires local fixture tests without GPU imports. Wrapping common stage functions preserves testability and avoids remote-only behavior.

## Goals

- Isolate data, training, evaluation, and serving dependency images.
- Keep stage contracts identical locally and remotely.
- Use remote volumes for artifacts.
- Configure secrets through the remote secret store.
- Support smoke and full runs.

## Non-Goals

- Multi-cloud scheduling.
- Multi-node training.
- Long-running production serving.
- Provider-neutral orchestration abstraction in v1.0.

## Proposed Design

### Remote Entrypoints

```python
def remote_prepare(config_path: str, run_id: str) -> str: ...
def remote_format(config_path: str, run_id: str) -> str: ...
def remote_eval_base(config_path: str, run_id: str) -> str: ...
def remote_train(config_path: str, run_id: str) -> str: ...
def remote_eval_adapter(config_path: str, run_id: str) -> str: ...
def remote_compare(config_path: str, run_id: str) -> str: ...
```

Each function returns the stage manifest path.

### Image Separation

- Data image: GPU dataframe stack plus dataset loader.
- Training image: model loading, quantization, adapter training, and tracking dependencies.
- Evaluation image: inference runtime, parser, metrics, and report dependencies.
- Serving image: optional OpenAI-compatible endpoint dependencies.

### Modes

```text
sommelier pipeline run --mode smoke
sommelier pipeline run --mode full
```

Smoke mode overrides split sizes through a generated resolved config and marks the run as smoke in the manifest. It must not overwrite full-run artifacts.

### Secrets

Remote secrets map to environment variables consumed by the same preflight checks as local commands. Missing secrets fail before remote GPU allocation when possible.

## Alternatives Considered

- Put all dependencies in one image. Rejected because RAPIDS, training, inference, and serving stacks can conflict and slow iteration.
- Write remote-only stage code. Rejected because local tests would not cover production behavior.
- Make remote execution optional for full training. Rejected for v1.0 because the PRD targets remote single-GPU execution.

## Drawbacks

- Multiple images increase setup complexity.
- Remote smoke tests can still cost money.
- Provider APIs may change outside the package's control.

## Migration / Rollout

1. Refactor the existing smoke entrypoint into `sommelier.remote.app`.
2. Add remote preflight for secrets and artifact volume.
3. Implement smoke mode.
4. Implement full stage chaining.
5. Record remote runtime and cost metadata in manifests when available.

## Testing Strategy

- Unit-test that remote functions call shared stage functions.
- Unit-test missing-secret preflight.
- Local fixture-test generated resolved configs for smoke mode.
- Run one remote smoke command before v1.0.
- Verify remote manifests match local schema.

## Open Questions

None for v1.0.

## References

- [01-architecture](../spec/01-architecture.md)
- [08-performance-budget](../spec/08-performance-budget.md)
