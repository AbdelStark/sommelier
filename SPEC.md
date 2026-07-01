# Sommelier Specification

- Status: Draft baseline for implementation
- Created: 2026-07-01
- Target: v1.0 reference implementation
- Source of intent: [prd.md](prd.md) and [docs/blogposts/sommelier_experiment_braindump.md](docs/blogposts/sommelier_experiment_braindump.md)

Sommelier is an open, reproducible reference implementation for fine-tuning a small open language model to emit schema-valid tool calls. The v1.0 scope is intentionally narrow: one base model, one tool-calling dataset, one single-GPU training path, one deterministic evaluation harness, and one report that compares the base model against the fine-tuned adapter on a held-out split.

The project is not a claim of general tool-use capability, production serving readiness, or superiority over larger hosted models. It is a buildable, inspectable pipeline for practitioners who want to understand and adapt the data preparation, formatting, adapter training, evaluation, and artifact-reporting path.

## Corpus

- [00-overview](docs/spec/00-overview.md): thesis, goals, non-goals, milestones, and risks.
- [01-architecture](docs/spec/01-architecture.md): system architecture, module boundaries, and data flow.
- [02-public-api](docs/spec/02-public-api.md): package, CLI, Python API, and versioning contracts.
- [03-data-model](docs/spec/03-data-model.md): schemas, invariants, artifact layout, and schema versions.
- [04-error-model](docs/spec/04-error-model.md): error taxonomy, failure responses, and recovery behavior.
- [05-observability](docs/spec/05-observability.md): logs, metrics, reports, traces, and redaction.
- [06-security](docs/spec/06-security.md): threat model, trust boundaries, licenses, and secret handling.
- [07-testing-strategy](docs/spec/07-testing-strategy.md): unit, property, integration, ML, and release-gate tests.
- [08-performance-budget](docs/spec/08-performance-budget.md): runtime, memory, cost, and profiling budgets.
- [09-release-and-versioning](docs/spec/09-release-and-versioning.md): semantic versioning, deprecation, changelog, and artifact policy.
- [10-glossary](docs/spec/10-glossary.md): canonical terms.

## RFC Index

- [RFC-0001: Project Configuration and Run Manifest](docs/rfcs/RFC-0001-project-configuration-and-run-manifest.md)
- [RFC-0002: Dataset Preparation and Split Discipline](docs/rfcs/RFC-0002-dataset-preparation-and-split-discipline.md)
- [RFC-0003: Tool-Call Chat Formatting](docs/rfcs/RFC-0003-tool-call-chat-formatting.md)
- [RFC-0004: Adapter Training Contract](docs/rfcs/RFC-0004-adapter-training-contract.md)
- [RFC-0005: Evaluation Parser and Metrics](docs/rfcs/RFC-0005-evaluation-parser-and-metrics.md)
- [RFC-0006: Artifact Store and Schema Versioning](docs/rfcs/RFC-0006-artifact-store-and-schema-versioning.md)
- [RFC-0007: Remote GPU Orchestration](docs/rfcs/RFC-0007-remote-gpu-orchestration.md)
- [RFC-0008: CLI and Python Public API](docs/rfcs/RFC-0008-cli-and-python-public-api.md)
- [RFC-0009: Observability and Run Reports](docs/rfcs/RFC-0009-observability-and-run-reports.md)
- [RFC-0010: Optional Inference Service](docs/rfcs/RFC-0010-optional-inference-service.md)
- [RFC-0011: Security, Licensing, and Release Gates](docs/rfcs/RFC-0011-security-licensing-and-release-gates.md)

## v1.0 Success Criteria

Sommelier v1.0 is complete when:

1. A clean checkout can install the package, validate configuration, and run local unit tests without GPU access.
2. The remote pipeline can prepare data, format examples, evaluate the base model, train an adapter, evaluate the adapter, and write a comparison report using one GPU.
3. Every persisted artifact has a schema version, manifest entry, checksum, and producing command.
4. Evaluation uses the same prompt construction and parser for the base and fine-tuned models.
5. The release documentation states exact commands, environment prerequisites, license obligations, costs observed during the reference run, and limitations.

## Future Work Outside v1.0

These items remain outside the v1.0 implementation issue set unless a later RFC accepts them:

- Multi-turn tool use.
- Multi-call tool plans.
- Additional datasets.
- Preference optimization.
- Multi-GPU or multi-node training.
- Claims against public leaderboards beyond a clearly labeled optional benchmark run.
