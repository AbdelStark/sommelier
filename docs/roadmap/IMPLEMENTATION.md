# Implementation Tracker - 2026-07-01

Generated from the specification corpus committed in [PR #1](https://github.com/AbdelStark/sommelier/pull/1). Every implementable unit of work in the v1.0 spec corpus is filed below. Each issue is intended to be independently shippable; dependency edges are noted after the milestone tables.

## Milestone: v0.1

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #2 | config: implement strict SommelierConfig loader | config | p0 | m | RFC-0001 | open |
| #3 | artifacts: implement checksums and atomic writes | artifacts | p0 | m | RFC-0006 | open |
| #4 | artifacts: enforce schema-versioned readers | artifacts | p0 | m | RFC-0006 | open |
| #5 | manifests: write run and stage manifests | artifacts | p0 | m | RFC-0001, RFC-0006 | open |
| #6 | cli: add package layout and console entrypoint | cli | p0 | m | RFC-0008 | open |
| #7 | cli: map SommelierError classes to exit codes | cli | p0 | m | RFC-0008 | open |
| #8 | cli: enforce import discipline without GPU deps | cli | p0 | s | RFC-0008 | open |
| #10 | security: prevent secrets in config and manifests | security | p0 | m | RFC-0001, RFC-0011 | open |
| #11 | security: add artifact redaction scanner | security | p0 | m | RFC-0009, RFC-0011 | open |
| #12 | release: add project license and third-party notices | release | p0 | s | RFC-0011 | open |
| #13 | ci: add local lint type and test gates | release | p0 | m | RFC-0008, RFC-0011 | open |
| #14 | config: add smoke and full config examples | config | p0 | s | RFC-0001 | open |

## Milestone: v0.2

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #15 | data: validate raw rows and classify drop reasons | data | p0 | m | RFC-0002 | open |
| #16 | data: implement deterministic dedupe and split writer | data | p0 | m | RFC-0002 | open |
| #17 | data: add preparation fixtures and split invariants | data | p0 | s | RFC-0002 | open |
| #18 | data: add GPU dataframe preparation path | data | p1 | m | RFC-0002 | open |
| #19 | formatting: implement canonical tool-call messages | formatting | p0 | m | RFC-0003 | open |
| #20 | formatting: render tokenizer templates and prompt digests | formatting | p0 | m | RFC-0003 | open |
| #21 | formatting: add golden prompt fixtures | formatting | p0 | s | RFC-0003 | open |
| #30 | cli: implement stage subcommands | cli | p0 | l | RFC-0008 | open |

## Milestone: v0.3

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #9 | observability: implement structured JSONL logger | observability | p1 | m | RFC-0009 | open |
| #23 | evaluation: implement conservative JSON parser | evaluation | p0 | m | RFC-0005 | open |
| #24 | evaluation: implement tool-call metrics | evaluation | p0 | m | RFC-0005 | open |
| #25 | evaluation: add parser and metric fixture tests | evaluation | p0 | s | RFC-0005 | open |
| #26 | evaluation: implement deterministic generation runner | evaluation | p0 | l | RFC-0005 | open |
| #28 | remote: move smoke app into package wrappers | remote | p0 | m | RFC-0007 | open |
| #29 | remote: define separated data train and eval images | remote | p0 | m | RFC-0007 | open |
| #35 | remote: implement smoke and full pipeline commands | remote | p0 | l | RFC-0007, RFC-0008 | open |

## Milestone: v0.4

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #22 | training: implement assistant-token label masking | training | p0 | m | RFC-0003, RFC-0004 | open |
| #31 | training: implement QLoRA adapter training stage | training | p0 | l | RFC-0004 | open |
| #32 | training: record metrics and adapter manifests | training | p0 | m | RFC-0004, RFC-0006 | open |
| #33 | training: map resource failures to actionable errors | training | p1 | s | RFC-0004 | open |
| #34 | training: add one-step smoke training coverage | training | p0 | m | RFC-0004 | open |

## Milestone: v1.0

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #27 | evaluation: write reports and comparison gate | evaluation | p0 | m | RFC-0005 | open |
| #36 | remote: record runtime memory and cost metadata | remote | p1 | m | RFC-0007, RFC-0009 | open |
| #37 | observability: write local comparison reports | observability | p0 | m | RFC-0009 | open |
| #38 | observability: integrate optional experiment tracking | observability | p2 | m | RFC-0009 | open |
| #39 | serving: implement adapter endpoint schemas | serving | p2 | m | RFC-0010 | open |
| #40 | serving: start optional adapter service | serving | p2 | m | RFC-0010 | open |
| #41 | serving: document illustrative serving limits | serving | p2 | s | RFC-0010 | open |
| #42 | release: implement license and artifact preflight | release | p0 | m | RFC-0011 | open |
| #43 | docs: write install quickstart and reproduction guide | docs | p1 | m | RFC-0008, RFC-0011 | open |
| #44 | release: publish v1.0 checklist and changelog discipline | release | p1 | s | RFC-0011 | open |

## Tracking Issues

- #45 [Tracking] Project configuration and run manifest
- #46 [Tracking] Dataset preparation and split discipline
- #47 [Tracking] Tool-call chat formatting
- #48 [Tracking] Adapter training contract
- #49 [Tracking] Evaluation parser and metrics
- #50 [Tracking] Artifact store and schema versioning
- #51 [Tracking] Remote GPU orchestration
- #52 [Tracking] CLI and Python public API
- #53 [Tracking] Observability and run reports
- #54 [Tracking] Optional inference service
- #55 [Tracking] Security licensing and release gates

## Cross-Cutting Dependencies

- #2 blocks #5, #9, #10, #14, #15, and #30 because strict config loading and config digests are shared across manifests, observability, security, examples, data, and CLI commands.
- #3 and #4 block #5 because manifests require checksummed artifact references and schema-versioned readers.
- #5 blocks #10, #16, #20, #27, and #32 because those stages must record or validate stage manifests.
- #6 blocks #7, #8, #13, #28, and #30 because error mapping, import checks, CI, remote wrappers, and stage commands require the package entrypoint.
- #7 blocks #30 and #33 because stage commands and resource-error handling need the documented exit-code model.
- #8 blocks #13 and #29 because CI and remote image definitions must preserve lightweight base imports.
- #9 blocks #11, #32, #36, #37, and #38 because redaction, training metrics, runtime metadata, reports, and tracking use structured events.
- #10 blocks #11 because artifact redaction builds on the secret-hygiene policy.
- #11 blocks #37 and #42 because reports and release preflight must scan artifacts before publication.
- #12 blocks #42 and #43 because release preflight and reproduction docs need license evidence.
- #15 blocks #16 and #19 because split generation and formatting require validated prepared examples.
- #16 blocks #17, #18, and #35 because fixtures, GPU preparation, and pipeline execution depend on deterministic splits.
- #19 blocks #20 and #39 because tokenizer rendering and serving schemas reuse canonical message construction.
- #20 blocks #21, #22, #26, and #35 because prompt fixtures, label masking, generation, and pipeline execution require prompt digests.
- #22 blocks #31 because adapter training requires assistant-token-only labels.
- #23 blocks #24, #25, #26, and #39 because metrics, fixtures, generation, and serving responses all reuse the parser contract.
- #24 blocks #25 and #27 because parser/metric tests and comparison reports need metric implementations.
- #26 blocks #27 and #35 because comparison and pipeline execution require deterministic generation.
- #27 blocks #35, #37, and #38 because pipeline completion, local reports, and tracking require evaluation comparison output.
- #28 blocks #29 because remote images should wrap shared stage functions rather than duplicate smoke code.
- #29 blocks #31 and #40 because training and serving need separated remote dependency images.
- #30 blocks #31, #35, #40, and #43 because training, pipeline, serving, and docs rely on public stage commands.
- #31 blocks #32, #33, and #34 because training metrics, resource mapping, and smoke training require the adapter training stage.
- #32 blocks #34 and #38 because smoke training and optional tracking need training metrics.
- #33 blocks #34 because smoke training must exercise actionable resource failures.
- #35 blocks #36 because runtime and cost metadata are collected from pipeline execution.
- #36 blocks #37 because local reports include runtime, hardware, and cost metadata.
- #37 blocks #42 and #43 because release preflight and reproduction docs need final report artifacts.
- #39 blocks #40 because the optional service requires request and response schemas.
- #40 blocks #41 because serving documentation should describe the implemented illustrative endpoint.
- #42, #13, and #43 block #44 because the release checklist depends on preflight, CI, and reproduction documentation.
