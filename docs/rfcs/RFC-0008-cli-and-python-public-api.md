# RFC-0008: CLI and Python Public API

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v0.1

## Summary

Sommelier exposes one CLI and a small typed Python API. The CLI is the primary user workflow; the Python API is the shared implementation surface for tests, remote wrappers, and advanced users.

## Motivation

The PRD describes incremental phases with visible results. The public API spec requires commands for config validation, data preparation, formatting, evaluation, training, reporting, pipeline execution, and optional serving. A clear CLI/API boundary prevents remote wrappers and examples from inventing private entrypoints.

## Goals

- Provide stable command names for the v1.0 workflow.
- Return structured manifests from Python functions.
- Map expected errors to documented exit codes.
- Keep GPU dependencies behind command execution, not package import.

## Non-Goals

- Rich terminal UI.
- Notebook-first workflow.
- Backward compatibility before documented APIs exist.

## Proposed Design

### CLI

The CLI uses subcommands:

```text
sommelier config validate
sommelier data prepare
sommelier data validate-fixtures
sommelier format build
sommelier eval run
sommelier train run
sommelier report compare
sommelier pipeline run
sommelier serve adapter
```

Each command supports `--config`, `--artifact-root`, `--run-id`, and `--debug` when applicable.

### Python Functions

The Python API is the set of functions listed in [02-public-api](../spec/02-public-api.md). All return `StageManifest` or `RunManifest`. Functions raise `SommelierError` subclasses for expected failures.

### Import Discipline

`import sommelier` must not import GPU, remote, or tracking dependencies. Commands import heavy modules only inside stage execution.

### Error Mapping

```python
def main(argv: list[str] | None = None) -> int:
    try:
        return dispatch(argv)
    except SommelierError as exc:
        render_error(exc)
        return exc.exit_code
```

Unexpected exceptions return exit code 5 unless `--debug` is enabled, in which case a stack trace is printed.

## Alternatives Considered

- One `modal_app.py` as the public interface. Rejected because it couples users to remote execution and hides local validation.
- Only Python scripts. Rejected because the PRD requires an incremental workflow contributors can run consistently.
- Expose all internal modules as public. Rejected because it would freeze unstable details.

## Drawbacks

- Maintaining both CLI and Python API requires discipline.
- Command names may need migration if the project later expands beyond one pipeline.
- Strict import discipline requires tests.

## Migration / Rollout

1. Add package layout and CLI entrypoint in `pyproject.toml`.
2. Move the smoke function into package modules.
3. Implement `config validate` and fixture commands.
4. Add each stage command as the implementation lands.
5. Document commands in README.

## Testing Strategy

- Unit-test CLI argument parsing.
- Unit-test exit-code mapping.
- Test `python -c "import sommelier"` without optional GPU packages installed.
- Integration-test fixture commands through subprocess.
- Snapshot-test help output for public commands.

## Open Questions

None for v1.0.

## References

- [02-public-api](../spec/02-public-api.md)
- [04-error-model](../spec/04-error-model.md)
