# 00 Overview

- Status: Draft
- Target milestone: v1.0
- Source: [prd.md](../../prd.md)

## Thesis

Sommelier demonstrates a reproducible, single-GPU workflow for adapting a small open language model to emit JSON tool calls from natural-language requests and tool schemas. The value is the complete reference path: deterministic data preparation, audited formatting, parameter-efficient training, identical base and adapter evaluation, and a report that states improvement and limitations without overclaiming.

## Design Reflection

The project solves a narrow problem that a generic fine-tuning example does not solve: it shows the exact contracts required to convert raw tool-calling rows into schema-valid training targets and to evaluate the resulting adapter with the same parser and prompts used for the base model.

The three hardest technical problems are:

1. Preventing data leakage while preserving enough examples for a meaningful held-out evaluation.
2. Training only on assistant tool-call tokens so the adapter learns structured outputs rather than prompt text.
3. Scoring generated tool calls in a deterministic, inspectable way that handles parse failures and partial argument matches.

The load-bearing abstractions are `SommelierConfig`, `RunManifest`, `ToolCallExample`, `FormattedExample`, `AdapterArtifact`, and `EvaluationReport`. They enforce schema versions, deterministic splits, prompt identity, artifact checksums, and metric provenance.

Failure modes that must be designed against include dataset schema drift, malformed tool JSON, train-test leakage, tokenizer template drift, out-of-memory training, invalid generated JSON, adapter/base prompt mismatch, secret leakage, and unsupported hardware.

The v1.0 scope excludes multi-turn agents, multi-call planning, production serving, preference optimization, and claims about broad model superiority. Those exclusions keep the project buildable and measurable.

A contributor six months from now should be able to add a new dataset adapter or metric by following the data model, public API, and RFC contracts without asking for unstated intent.

## Goals

- `G1`: Produce a fine-tuned adapter that improves held-out tool-call metrics over the configured base model.
- `G2`: Prepare data with GPU dataframe operations for filtering, deduplication, deterministic shuffling, and splitting.
- `G3`: Keep the default run within one GPU and publish the observed runtime, memory, and cost envelope.
- `G4`: Evaluate the base model and adapter with identical prompts, deterministic decoding, and a shared parser.
- `G5`: Make every artifact reproducible from a versioned config, manifest, command, checksum, and seed.
- `G6`: Provide a readable open-source codebase with typed APIs, tests, docs, and release gates.

## Non-Goals

- Pretraining or full-parameter fine-tuning.
- Multi-GPU or multi-node orchestration.
- Production-grade serving, autoscaling, or multi-tenant operations.
- Agent orchestration beyond a single tool call response.
- Claims of absolute superiority over frontier hosted models.
- A public benchmark claim unless the exact benchmark command, input revision, and caveats are committed.

## Users

- Developers adapting a small model to a known set of tool schemas.
- Practitioners learning the end-to-end post-training workflow.
- Maintainers who need a documented reference pipeline with clear extension points.

## Milestones

### v0.1: Local Package and Contracts

- Package layout, config schema, manifest schema, CLI skeleton, and local tests.
- Data model, error model, and artifact schema fixtures.
- Documentation corpus and issue tracker.

### v0.2: Data and Formatting

- Dataset download, cleaning, deduplication, validation, deterministic split generation, and formatted training examples.
- Golden fixtures for prompt rendering and split manifests.

### v0.3: Baseline Evaluation

- Deterministic base-model generation path.
- Parser, metrics, report schema, and baseline metrics artifact.

### v0.4: Adapter Training

- QLoRA training command, assistant-token-only loss, checkpointing, adapter artifact manifest, and training telemetry.

### v1.0: Reference Result

- Base versus adapter evaluation table.
- Reproduction instructions and limitations.
- Optional single-adapter inference service marked as illustrative, not production.

## Success Criteria

- A clean local environment can run `sommelier config validate`, `sommelier data validate-fixtures`, and the non-GPU test suite.
- The reference remote run produces `data_manifest.json`, `format_manifest.json`, `train_manifest.json`, `evaluation_base.json`, `evaluation_adapter.json`, and `comparison_report.md`.
- The report includes valid-JSON rate, function-name accuracy, argument exact match, argument F1, and full-call exact match for both base and adapter.
- The report includes the exact config digest, git commit, dependency lock digest, dataset revision, and artifact checksums.
- Documentation states that results are task-specific and do not imply production readiness or general tool-use capability.

## Risks

- `RISK: Dataset schema drift`. Resolution: pin dataset revision in config and validate raw rows against `RawToolCallRow` before cleaning.
- `RISK: Leakage through duplicate or near-duplicate requests`. Resolution: deduplicate exact normalized requests in v1.0 and file future work for semantic deduplication only if exact deduplication is insufficient.
- `RISK: Prompt template drift across tokenizer versions`. Resolution: persist tokenizer revision, rendered prompt fixtures, and prompt digest in every formatted artifact.
- `RISK: Evaluation parser hides invalid generations`. Resolution: persist raw generations and parse status for every example; invalid parse counts as metric failure.
- `RISK: Cost target may vary by GPU market and queue time`. Resolution: report observed runtime, GPU type, and billed cost separately from acceptance gates.

## Claim Boundaries

Sommelier may claim a measured improvement only for the configured base model, dataset split, prompt format, parser, and decoding settings. It must not claim broad agent reliability, multilingual performance, public benchmark ranking, or production cost savings without separate evidence.
