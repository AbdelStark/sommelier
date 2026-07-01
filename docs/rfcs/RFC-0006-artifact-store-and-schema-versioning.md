# RFC-0006: Artifact Store and Schema Versioning

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v0.1

## Summary

Sommelier stores every pipeline output under a run-scoped artifact root and treats schema-versioned manifests as the authoritative registry. Readers fail closed on unknown schema versions and reject missing checksums.

## Motivation

The PRD requires reproducibility and visible artifacts at each phase. The architecture spec requires independent stages. A structured artifact store lets contributors rerun, inspect, and compare stages without hidden remote state.

## Goals

- Define one run directory layout.
- Require checksums for all persisted artifacts.
- Version every JSON and JSONL schema.
- Make partial writes safe.
- Keep artifacts outside the repository by default.

## Non-Goals

- Implement a remote object store abstraction in v1.0.
- Provide artifact garbage collection.
- Support multiple schema versions before `1.0.0`.

## Proposed Design

### Artifact Root

The resolved config sets:

```yaml
project:
  artifact_root: artifacts
```

The CLI creates `artifacts/runs/<run_id>/`. A `latest` symlink is optional and must not be used by manifests.

### Atomic Writes

```python
def write_artifact_atomic(path: Path, writer: Callable[[Path], None]) -> ArtifactRef: ...
```

Writers write to `path.tmp.<pid>`, validate the content, compute SHA-256, then move into place.

### Schema Registry

```python
SUPPORTED_SCHEMAS = {
    "sommelier.config.v1",
    "sommelier.manifest.v1",
    "sommelier.prepared_example.v1",
    "sommelier.formatted_example.v1",
    "sommelier.generation.v1",
    "sommelier.evaluation_report.v1",
}
```

Readers reject records whose `schema_version` is absent or unsupported.

### Artifact Manifest

Each stage manifest lists declared inputs and outputs. A root `manifest.json` indexes stage manifests:

```python
class RunManifest(TypedDict):
    schema_version: Literal["sommelier.manifest.v1"]
    run_id: str
    stages: dict[str, str]
    config: ArtifactRef
    status: Literal["running", "succeeded", "failed"]
```

## Alternatives Considered

- Write outputs directly into top-level `results/`. Rejected because it makes repeated runs overwrite each other.
- Store only final reports. Rejected because stage artifacts are needed for debugging and reproduction.
- Use a database. Rejected because files are enough for a reference implementation.

## Drawbacks

- Manifests add implementation overhead.
- File artifacts can become large when raw generations are retained.
- Schema migration discipline is required.

## Migration / Rollout

1. Add artifact helpers and checksum utilities.
2. Move stage outputs into run-scoped directories.
3. Add schema-version checks to readers.
4. Add root manifest updates after each stage.

## Testing Strategy

- Unit-test checksum generation.
- Unit-test unknown schema rejection.
- Unit-test atomic write cleanup on failure.
- Unit-test root manifest stage indexing.
- Integration-test a fixture pipeline that writes all expected files.

## Open Questions

None for v1.0.

## References

- [01-architecture](../spec/01-architecture.md)
- [03-data-model](../spec/03-data-model.md)
