# 09 Release and Versioning

- Status: Draft
- Target milestone: v1.0
- Primary RFC: [RFC-0011](../rfcs/RFC-0011-security-licensing-and-release-gates.md)

## Version Scheme

Sommelier uses semantic versioning for the Python package:

- `0.x`: public contracts are stabilizing.
- `1.0.0`: documented CLI, Python API, artifact schemas, and metric names are stable.
- Patch releases fix bugs without changing defaults or schemas.
- Minor releases may add optional commands, metrics, or artifact fields.
- Major releases may change public contracts with migration notes.

## Release Artifacts

A v1.0 release includes:

- Source distribution and wheel.
- `SPEC.md` and docs corpus.
- Changelog.
- License file.
- Third-party license and attribution file.
- Reference run report.
- Machine-readable report JSON.

The release does not publish model weights or adapters unless license obligations are verified and documented.

## Changelog Policy

Every user-visible change records:

- Added, changed, fixed, deprecated, removed, or security category.
- Affected command, module, or artifact schema.
- Migration note when behavior changes.

## Documentation Gate

Before release:

- README install and quickstart commands must work or state prerequisites.
- The docs must state GPU, account, license, and cost assumptions.
- The report must distinguish fixture, smoke, and full reference evidence.
- Public claims must link to a reproduction command or artifact.

## Artifact Compatibility

Artifact readers support only declared schema versions. v1.0 readers may reject pre-v1 artifacts unless a migration command exists.

## Project License Decision

The project code license is MIT unless maintainers choose another license before v1.0. A missing `LICENSE` file blocks v1.0.

## Release Blockers

- Missing license acknowledgement for base model or dataset.
- Unpinned reference-run dependencies.
- Missing comparison report.
- Mismatched base and adapter evaluation split digests.
- Secrets in artifacts.
- Test suite failure.
- Undocumented change to a public schema or metric.
