# Changelog

All notable, user-visible changes to Sommelier are recorded here.

## Changelog policy

Every user-visible change records:

- a category: **Added**, **Changed**, **Fixed**, **Deprecated**,
  **Removed**, or **Security**;
- the affected command, module, or artifact schema;
- a migration note whenever behavior changes.

Entries land in the Unreleased section of the pull request that makes the
change; releases move them under a version heading with a date.

## Unreleased

### Added

- `sommelier` CLI stage surface: `config validate`, `data prepare`,
  `data validate-fixtures`, `format build`, `eval run`, `train run`,
  `report compare`, `pipeline run --mode smoke|full`, `serve adapter`,
  and `release preflight`.
- Tool-call formatting: canonical chat messages, tokenizer template
  rendering, prompt digests, and golden prompt fixtures
  (`sommelier.formatted_example.v1`).
- Evaluation: conservative JSON parser (`sommelier.parser.v1`), five
  tool-call metrics with numerators and denominators, deterministic
  generation records (`sommelier.generation.v1`), evaluation reports with
  comparability digests (`sommelier.evaluation_report.v1`), and the
  comparison gate with JSON and Markdown reports
  (`sommelier.comparison_report.v1`).
- Training: completion-only label masking with provable prompt
  boundaries, the QLoRA adapter training stage, training metrics
  (`sommelier.training_metric.v1`), and actionable OOM/timeout resource
  errors.
- Observability: structured JSONL stage logs (`sommelier.log_event.v1`)
  with write-time redaction, the artifact redaction scanner, per-stage
  runtime/hardware/cost metadata (`sommelier.runtime_metadata.v1`), and
  optional wandb experiment tracking via the `tracking` config section.
- Remote: Modal smoke app under `sommelier.remote.app` and separated
  data/train/eval/serving dependency images with GPU and timeout hooks.
- Serving: optional, illustrative single-adapter endpoint with strict
  request/response schemas and parser-status responses.
- Initial release gates: MIT `LICENSE`, `licenses/THIRD_PARTY.md`, and
  `sommelier release preflight`. The original
  `sommelier.release_preflight.v1` report is now historical and has no current
  writer; the identity-bound v2 contract below supersedes it.
- Docs: install quickstart, reproduction guide, serving limits, and this
  changelog with the v1.0 release checklist.

### Added (Hebrew v3 experiment)

- Hebrew paired-data production and publication contracts: constrained
  query-only translation; exact root pairing; provider-independent structured
  instruction-chat prompts; placeholder, script, bidi, and protected-span
  audits; durable resume state; `sommelier.translation_summary.v2`; and the
  digest-bound `sommelier.translation_publication_manifest.v1`.
- An explicit OpenAI Responses producer for the dated GPT-5.5 snapshot in a
  CPU-only Modal image. The v2 raw journal attributes every response, replay,
  provider error, and availability retry to a source row and audited attempt;
  public evidence is a content-free aggregate with exact returned model/tier,
  usage, request/runtime identity, and a pinned-public-list-price calculation.
  SDK retries remain disabled. Exact Flex availability failures use same-row,
  journaled provider-call attempts at fixed 1/2/4/8/16-second delays with no
  tier switch; they are distinct from the three semantic/audit row attempts.
  A required operator-supplied USD ceiling (`1.00` smoke, `50.00` full) adds
  conservative pre-batch admission and post-response stop guards without
  claiming to be an invoice or provider-side account/project spend cap.
- A preregistered 200-row semantic-review gate with a pinned independent
  Hebrew-to-English back-translator, immutable machine template, and one named
  human reviewer's stable id plus canonical Ed25519 public key/fingerprint
  committed before Phase-A translation. The canonical decision attestation is
  signed under the dedicated OpenSSH namespace; finalization verifies that
  signature, recomputes the locked back-translations, enforces the
  zero-critical-error release criterion, and binds exact
  template/review/publication digests
  (`sommelier.translation_semantic_review_template.v1`,
  `sommelier.translation_semantic_review_attestation.v1`, and
  `sommelier.translation_semantic_review.v1`).
  Reviewer identity cannot be supplied through the semantic-review CLI, and
  the private key never enters the repository, Modal, Sommelier, or Codex.
- `sommelier analyze tokenization` and the tokenizer-tax record/report schemas,
  measuring exact query/prompt/target/full tokens, matched English-Hebrew
  ratios, sequence-budget failures, and projected non-padding training tokens.
  Hebrew en+he evidence now separates an English-only arithmetic counterfactual,
  the additive Hebrew examples/token tax, and the actual combined workload,
  with raw-record-derived Hebrew ratios and combined-vs-English multipliers.
- Matched-pair evaluation and deterministic paired-bootstrap intervals;
  per-call inference telemetry; separate base/adapter evaluation manifests;
  `sommelier.evaluation_report.v3`; `sommelier.comparison_report.v3`; and the
  three-arm `sommelier report experiment` claim gate. The embedded
  `sommelier.sovereign_tco_evidence.v1` keeps measured QLoRA runtime, memory,
  storage, and inference hardware-time distinct from projections and from
  unavailable currency billing.
- `report experiment` now emits `sommelier.experiment_report.v2` plus a fresh,
  deterministic
  `evaluation_evidence/` subbundle with privacy-minimized per-row metric
  components plus allowlisted eval manifests and telemetry. Adapter
  publication requires the exact tree and recomputes every reported metric,
  delta, seeded paired-bootstrap interval, and McNemar result from those rows;
  regenerate the experiment output and copy the report and evidence directory
  together.
- Hebrew v3 remote execution gates: immutable published paired-source loading,
  exact data-provenance traversal, clean source-revision binding, pinned v1
  baseline identity, outer-timeout admission evidence, and a dedicated
  diagnostic L40S QLoRA shape preflight using synthetic near-4096-token
  English/Hebrew rows for one real optimizer step plus one evaluation forward.
- `sommelier release publish-dataset` and `publish-adapter`. Both validate
  exact allowlisted bundles by default; mutation requires `--execute`, exact
  repository confirmation, and a fresh receipt. Optional repository creation
  is public and `exist_ok=false`; execution durably reserves a pending receipt,
  binds the commit to the observed parent revision, journals a returned commit
  before verification, and records success only after immutable-revision
  enumeration and SHA-256 round-trip verification. Adapter releases require
  byte-exact reviewed NVIDIA/Llama agreements and `NOTICE` alongside evidence.
- `sommelier release preflight --artifact-root <dir>` and the closed
  `sommelier.release_preflight.v2` contract. It binds the normalized config;
  model, tokenizer, and ordered dataset revisions plus their immutability
  decisions; producer commit and cleanliness decision; dependency-lock digest;
  and one coherent streamed scan/tree identity, excluding only the report
  itself. Adapter publication revalidates those bindings against the unchanged
  final curated bundle and requires immutable revisions plus a clean producer.
  Without the flag, artifact-root resolution remains relative to the config
  file for backward compatibility.

Migration: regenerate multilingual data, formatted rows, generations,
telemetry, evaluation reports, comparisons, and manifests with the current
schemas. Historical v1/v2 reports and French v1 translation summaries remain
readable evidence but cannot satisfy the Hebrew v3 full-publication or
three-arm claim gates. Paid OpenAI launches must now pass an explicit positive
`--openai-list-price-limit-usd`; install the `publish` extra only on a host that
will execute a Hub publication.

### Added (Hebrew local-MT slice)

- `local_hy_translate.py`: a local Hebrew translation driver that reproduces the
  full-run root-cohort selection and translates each query with
  `tencent/Hy-MT2-1.8B` (Q8_0 GGUF) served by ollama, masking protected spans
  with a short-sentinel retry ladder and reusing the shared protected-span and
  translation-audit primitives. It emits `sommelier.raw_tool_call_row.v1` rows
  and a `sommelier.local_hy_translation_summary.v1` accounting file.
- `remote_hy_pipeline.py`: a Modal pipeline for the local-MT Hebrew slice that
  reuses the standard data/format/tokenization/eval/train/compare stages without
  the preregistered-evidence admission gate, scoped through a distinct project
  name and Hebrew dataset repository so `is_hebrew_v3_config` stays false.
- `examples/config.he-hymt-full.yaml` and `examples/config.he-hymt-smoke.yaml`:
  configs for the local-MT slice pinned to the published Hebrew dataset revision.
- Docs: a Hebrew local-MT results page and honest dataset and adapter cards that
  state the machine-translation method and the absence of human semantic review.
  This slice is deliberately separate from the preregistered Hebrew v3 evidence
  and makes no claim to be it.
- Publication scanning now recognizes standard tokenizer/PEFT metadata keys and
  tokenizer vocabulary maps as public model data while continuing to scan every
  vocabulary string for credential-shaped values. Raw evaluation rows remain
  fail-closed and are omitted from the public Hy-MT2 bundle when they contain
  credential-shaped synthetic benchmark content.
- The exact Hy-MT2 training dataset snapshot is privately quarantined after the
  release scan found 15 upstream APIGen strings shaped like GitHub personal
  access tokens. The public dataset keeps all 16,272 rows, replaces only those
  15 `tools` substrings with `[redacted]`, and publishes a closed input/output
  sanitization manifest; it is explicitly not byte-identical to training data.
- Published the curated adapter/evidence bundle as immutable Hub tag
  `he-hymt-full-002` (`f09e5d31ab29dca49e7c2df7113e810cf3dfb43a`) and the
  release-sanitized dataset as `he-hymt-sanitized-v1`
  (`f7159c08823e2f986375927f998263f969738f43`).

### Security

- Hebrew v3 adapter publication now closed-validates the final experiment
  report, re-derives both claim gates from the fixed paired-bootstrap bounds,
  and requires the model card's claim section to match the deterministic
  renderer exactly. Minimal or edited reports, inconsistent approved-claim
  lists, missing TCO/provenance sections, and hand-written or unapproved card
  claims fail before Hub access. Nested TCO validation also binds paired-scope
  counts to the observed cohorts, rejects negative or inconsistent training and
  inference measurements, and verifies artifact source paths, hashes, schemas,
  and storage totals. Migration: regenerate the report with the current
  `sommelier report experiment` command and replace the adapter-card claim
  placeholder with `render_hebrew_v3_claim_section(...)` output.
- Atomic artifact writes now use a private random staging directory, exclusive
  no-follow regular files, descriptor-bound copying, mutation checks, fsync,
  and atomic replacement. Release preflight v2 likewise scans and hashes one
  coherent descriptor-bound artifact snapshot and rejects symlinks and special
  files instead of certifying bytes through path-only reads.
- Dataset and adapter publication now validates private source snapshots and
  submits a second read-only upload snapshot whose identity and digest are
  checked before and after the Hub commit. Receipts must be outside every
  source or snapshot path, including filesystem aliases; ambiguous parentless
  repositories fail closed; JSON object keys and safetensors metadata receive
  the same secret-shape checks as public text artifacts.
- Executed publication keeps the originally reserved receipt handle open across
  the complete Hub transaction, verifies exact prior-content hashes and sizes
  before every state transition, reads back each durable update, and closes the
  handle on every success or error path. Filesystem inode reuse can no longer
  make an unlinked/recreated receipt appear to be the original journal.

### Changed

- Canonical full-sized Hebrew v3 training now requires
  `pipeline run --mode full`. The complete paired-input/publication/semantic-review validator issues
  a process-local, config-bound, single-use capability to the train stage only
  after it passes; standalone `train run` and direct `train_adapter` calls fail
  before model or adapter work. Migration: launch Hebrew v3 full training
  through the full pipeline. Ordinary, v1, v2, and bounded diagnostic training
  remain unchanged.
- Full `pipeline run` attempts now accept only safe single-component run IDs,
  require the same grammar for paired-smoke `--translation-run-id`, and
  reject aliased translation directories before export. Full runs atomically
  reserve a fresh directory; completed or failed full IDs are non-resumable,
  while smoke IDs remain reusable. Migration: choose a new `--run-id` for every
  full attempt and replace path-like legacy IDs.
- The full Hebrew `remote_translate.py` producer now records
  `sommelier.translation_run_identity.v1` through an exclusive reservation
  before dataset or provider access, permits progress-only resume only when the
  exact config, reviewer, selection, translator, provider, source, and resource
  identity still match, and permanently rejects a run ID after terminal output
  appears. Dataset publication now requires that identity and the exact
  Phase-A config bytes as `translation_run_identity.json` and
  `translation_config.yaml`. Migration: preserve old identity-less or
  finalized directories and continue with a new safe run ID.
- The remote Hebrew semantic-review producer now reserves its final template
  path exclusively before input or model work. Caught failures release only
  the same still-empty reservation; a replaced, nonempty, or hard-crash marker
  remains fail-closed. Migration: retry directly only after caught cleanup; for
  a hard crash, verify no producer is active and remove only a confirmed
  zero-byte reservation using the documented recovery command.

- `format build` defaults to tokenizer chat-template rendering; the
  no-tokenizer path moved behind `--fixture`. Migration: append
  `--fixture` to keep the previous fixture behavior.
- `LogEvent` and `EvaluationReport` schemas gained fields
  (`schema_version` on log events; `created_at`, `test_split_sha256`,
  `prompt_set_sha256`, `decoding` on evaluation reports). Migration:
  regenerate artifacts with the current pipeline; readers fail closed on
  older shapes.
- The `tracking` config section is new and optional; existing configs
  remain valid.
- Data preparation drops rows whose answers contain more than one tool
  call, with the declared `multi_call_answer` drop reason (v1 trains and
  scores exactly one call; the previous behavior scored faithful
  multi-call outputs as failures). Migration: regenerate prepared splits;
  multi-call rows now appear in the drop summary instead of the splits.
- `examples/config.smoke.yaml` raises `train.max_sequence_length` to
  2048; real xlam prompts exceed the previous 1024-token budget.
- `examples/config.full.yaml` now records the settings that produced the
  published reference run (`nemotron-8b-full-3`): batch 4 with gradient
  accumulation 4, `max_sequence_length: 4096`, `remote.gpu: L40S`, and
  8-hour train/eval timeouts. The previous values (batch 8, 2048 tokens,
  A10G) failed the sequence-length audit on real xlam rows, which reach
  2,166 tokens. Migration: rerun `sommelier config validate` if you
  derived a config from the old example.

### Added (remote serving)

- `remote_serving.py`: Modal entrypoint serving the trained adapter with
  vLLM's OpenAI-compatible server (`--enable-lora`), registering both the
  base model and the `sommelier-tool-caller` LoRA on one endpoint, with
  adapter sourcing from the published Hugging Face repo or the artifacts
  volume, optional Bearer-token protection, scale-to-zero, a readiness-
  polling smoke entrypoint that validates completions through the
  sommelier parser, and a `diagnose` entrypoint that runs the engine in
  the foreground with full logs.
- `sommelier.remote.images.vllm_serving_image`: built from the CUDA devel
  base image because vLLM's startup warm-up JIT-compiles kernels with
  nvcc, which slim images lack.

### Added (remote execution)

- `remote_pipeline.py`: Modal entrypoint running the full pipeline on a
  GPU. It exports the configured Hugging Face dataset to raw rows, chains
  the shared stages with per-stage GPU cleanup and volume commits, audits
  rendered sequence lengths against the training budget before any model
  loads, and persists artifacts to the `sommelier-artifacts` volume.

### Fixed

- `sommelier.remote.images` now pins the evidence pipeline image and runtime
  gate to the same exact Python 3.13.3 patch, preventing a rebuilt image from
  drifting along the floating Python 3.13 line before paid work begins.

- Protected-span extraction (`sommelier.data.translate`) no longer crashes when a
  query contains a set literal with unhashable list elements (for example
  `{[1, 2]}`): `_parse_source_structure` now also catches the `TypeError` that
  `ast.literal_eval` raises for such a candidate and treats it as unparseable.

- Public v1/v2 prose, paper, and video sources now distinguish published
  aggregate reports from non-published raw generations, maintainer-observed
  billing from checksummed run evidence, and unequal marginal language slices
  from an exact paired language effect. The rebuilt paper PDF carries the same
  claim boundary.
- Installation/reproduction docs now agree that the Modal client is a base
  dependency, and package metadata declares the repository's MIT license with
  the PEP 621 license expression.
- Production QLoRA now sets its evaluation batch size explicitly and shares
  its NF4/model/LoRA/checkpointing/seed contract with the Hebrew-v3 shape
  preflight. The diagnostic additionally requires one visible L40S with a
  CUDA-0-only `hf_device_map`, proves exact one-English/one-Hebrew source
  pairs, and validates its terminal report, digest manifest, and on-disk tree
  as one closed contract.
- Source distributions now use an explicit OSS release allowlist and reject
  generated dependencies, caches, local artifacts, checkpoints, site output,
  and rendered video. Building from a working tree that contains those files
  no longer packages them into the sdist; wheel contents are unchanged.
- Remote images no longer run pip installs after mounting the package
  source, which Modal rejects at build time (`sommelier.remote.images`).
- Evaluation no longer re-adds special tokens to rendered prompts, which
  doubled the BOS token on Llama-family tokenizers
  (`sommelier.evaluation.generate`).
- Training no longer lets the Trainer strip `prompt_text`/`full_text`
  from batches before the completion-only collator runs
  (`remove_unused_columns=False` in `sommelier.training.qlora`).
- Drop-reason counters are derived from the `DropReason` literal, fixing
  a KeyError when a new reason was added (`sommelier.data.split`).
- Model loading works on Apple Silicon and CPU hosts: `device_map="auto"`
  is now used only when CUDA is available, otherwise the model loads
  normally and moves to MPS/CPU, fixing adapter dispatch failures in
  local serving (`sommelier.evaluation.generate`).
- The serving completions endpoint accepts its JSON body again: a
  postponed-annotation resolution issue had demoted the request body to
  a required query parameter (`sommelier.serving.openai_compat`); an
  HTTP-level end-to-end test now guards the route where fastapi is
  installed.

### Removed

- The design-phase planning corpus: `prd.md`, `SPEC.md`, `docs/spec/`,
  `docs/rfcs/`, and the stale `docs/roadmap/` table. The durable content
  (architecture, data model, error codes, security posture, testing
  strategy) lives in the project documentation instead. Migration: no
  command, schema, or artifact behavior changes; update any bookmarks
  into `docs/spec/` or `docs/rfcs/` to point at the documentation site.
