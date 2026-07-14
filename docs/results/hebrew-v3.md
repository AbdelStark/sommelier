# Hebrew v3: teacher selected, full results pending

This page defines the Hebrew v3 experiment before its full evidence run. A
bounded teacher-selection probe and a 140-row provider smoke exist, but there is
still no full-corpus Hebrew accuracy, tokenizer-tax, QLoRA, or pipeline-TCO
result. Full claims stay pending until the referenced JSON artifacts exist,
pass their identity gates, and are published with an immutable dataset and
adapter revision, a clean implementation revision, and the exact dated provider
snapshot and request identity. That API snapshot is not a public weight digest
or evidence of byte-identical provider regeneration.

## Question and claim gates

The experiment asks whether an English+Hebrew QLoRA adapter improves Hebrew single-call tool accuracy over the published English-only v1 adapter without materially reducing English accuracy. All three arms use the same pinned Nemotron-Nano-8B base, tokenizer, stored English/Hebrew prompts, gold calls, parser, and deterministic decoding:

| Arm | Weights | Training exposure |
|-----|---------|-------------------|
| Base | Pinned base checkpoint | none |
| v1 | Published English adapter at an immutable revision | English only |
| v3 | New QLoRA adapter from this run | English + Hebrew pairs |

The two machine-readable claim gates use `full_call_exact_match` and deterministic 95% paired-bootstrap intervals:

1. **Hebrew uplift:** the lower bound of v3 minus v1 on Hebrew must be greater than zero.
2. **English non-inferiority:** the lower bound of v3 minus v1 on English must be at least -0.01 (a predeclared absolute margin of one percentage point).

A failed gate withholds its statement; it is not converted into a softer claim. Other metrics remain diagnostics and are reported with their intervals.

## Exact cohorts, not unmatched slices

Every Hebrew test row names its English root through `source_example_id`, retains byte-identical tools and gold answers, and inherits the root split. The primary language gap compares only those exact pairs and records pair count, English coverage, and `pair_set_sha256`. Confidence intervals resample matched identities.

The report also retains complete-slice gaps under `cohort: marginal_full_slices`. Because translation rejection can leave fewer Hebrew rows than English roots, those values are descriptive and never replace the paired estimate. The accepted rows are a machine-translated survivor corpus: machine-translation error and selection through the translation audits remain limitations even for the matched analysis.

## Translation semantic-audit gate

Syntactic checks run on every accepted row. The selected
`gpt-5.5-2026-04-23` Responses teacher uses the provider-independent
instruction-chat contract. It replaces protected values with deterministic
ASCII placeholders before the provider request, restores them afterward, and
then audits their byte-identical preservation. Only the query is translated;
tool schemas are used solely through the bounded selected-tool projection
described below, and tools and gold answers remain byte-identical in the paired
rows. TranslateGemma uses the same placeholder family. The retained local
MADLAD seq2seq interface instead sends an unchanged source query; it is a
diagnostic/compatibility path, not the preregistered v3 teacher.

Instruction-chat completions additionally have a strict assistant envelope:
exactly one JSON object with only `schema_version` set to
`sommelier.instruction_chat_assistant_payload.v1` and a non-empty string
`target_text` containing no Unicode control, format, or surrogate code points.
The provider-independent row boundary parses after completion-token decoding
and before placeholder restoration, so an alternative instruction-chat backend
cannot bypass the envelope by returning plain text. Plain text, fenced JSON,
duplicate, missing, or extra keys, a wrong schema, and a non-stop partial
completion fail closed as `prompt_leakage`; the internal progress journal
retains the decoded malformed or partial completion behind an invalid-payload
marker and records the provider finish reason for diagnosis. A prompt rejected
before generation remains an empty output. Structured `target_text` is not
subjected to legacy quote/fence/label stripping. TranslateGemma and MADLAD
output decoding and plain-text post-processing are unchanged.

The raw OpenAI journal uses
`sommelier.openai_responses_provider_journal.v2`; every response, error, and
replay event carries the source row id and audited attempt number without adding
either field to the provider request body or request hash. Identical request
bodies may still coalesce, but each consumer receives its own attributed replay
event. Accepted progress records `accepted_attempt`; exhausted drops record
`final_attempt`. Responses are fsynced before they return to the row pipeline,
and the Modal volume is committed at the row-chunk boundary. This reduces
duplicate billing on resume but is not exactly once: a hard kill can lose the
current uncommitted chunk, and a process death after provider acceptance but
before response receipt and fsync can cause a repeated request.

The durable raw journal contains decoded outputs and provider response ids and
is not the public evidence surface. The translation summary publishes the
content-free `sommelier.openai_provider_evidence.v2` aggregate: journal digest,
requested/returned model and tier, counts, complete usage, and the calculated
public-list-price estimate. Strict JSON, placeholder preservation, target-script
coverage, and a clean journal still cannot prove that an action was translated
with the right intent.

Before publication, the release freezes a deterministic 200-row sample
balanced across root split, source-query length decile, protected-span count,
and tool/action family, with a fixed quota for ambiguous high-risk action
verbs. The sample IDs, full paired-corpus digest, and locked review-input
digest are selected before judgments.

The independent back-translator is
`Helsinki-NLP/opus-mt-tc-big-he-en@134c5a850dcaa763eec85bd1f4eb25112fecedbb`
(CC-BY-4.0), using greedy Hebrew-to-English Marian decoding under the fixed
`sommelier.marian_backtranslation_request.v1` request contract. It tokenizes
without truncation, rejects a source above 512 tokens, and caps internal
batches at eight. The model card self-reports BLEU 44.1 on FLORES-101 devtest
and 53.8 on its Tatoeba test set; those upstream figures are attribution
context, not validation on Sommelier data. An English-language,
non-native reviewer compares source, Hebrew translation, and back-translation
for action/tool intent, omissions or additions, polarity, quantities, and
entity relations. The release gate is zero critical errors; one failure causes
prompt/model correction and whole-run regeneration, never row removal. The
`sommelier.translation_semantic_review_template.v1` artifact locks the complete
paired corpus, forward translator, back-translator revision and decoding, and
sample before review. `sommelier.translation_semantic_review.v1` must preserve
those bytes while adding the rubric, every decision, and reviewer boundary.
No native-speaker review has been performed yet. Passing supports only
the bounded statement “200-row preregistered non-native back-translation
audit: zero critical errors”; it does not establish native fluency or
full-corpus semantic correctness.

## Tokenizer and training-cost evidence

`analyze tokenization` runs on the exact formatted strings consumed by evaluation and training. It records query characters, UTF-8 bytes, whitespace words, query tokens, prompt tokens, target tokens, and full tokens for every row. English↔Hebrew ratios use exact roots, with coverage and p50/p95/max per-pair ratios. The run also records over-budget rows and projected non-padding full tokens across configured epochs.

The allowed claim is narrow: **observed token inflation on this paired corpus under this pinned tokenizer**. The projected workload excludes dynamic padding and is a deterministic lower bound, not a cloud invoice and not evidence that Hebrew script alone caused the difference.

The three-arm experiment embeds `sommelier.sovereign_tco_evidence.v1`. It can
report observed QLoRA train-stage wall time, configured GPU-hours, peak
allocated GPU memory, trainer-reported input tokens, and both packaged-adapter
and tensor-only bytes. It can also report deterministic projected non-padding
tokens and, for each base/v1/v3 inference arm, sequential end-to-end
generator-call seconds per example and configured-GPU-seconds per exact
successful call. The default path
includes prompt tokenization, input device transfer, `model.generate`, and
generated-token decoding. It excludes model load, one deterministic discarded
warmup call, parsing, and artifact I/O, uses no explicit device synchronization,
and has concurrency one. Translation compute is separate from this pipeline
TCO. The provider-backed translation summary records API usage and a
deterministic public-list-price calculation separately; that value is not an
invoice or observed billing.

Pipeline currency cost remains explicitly unavailable unless an observed
billing artifact is joined. Without a matched full-parameter fine-tuning arm,
v3 will not claim a measured saving versus full fine-tuning. Adapter storage,
peak memory, runtime, and task accuracy are observed QLoRA characteristics, not
substitutes for that missing comparison.

## Teacher selection and bounded smoke

Instruction-chat translator candidates use one bounded semantic aid for
domain-term disambiguation. For each row, the producer resolves exactly one
tool schema by a case-sensitive exact match to the gold call name, then exposes
only tool name/description and sorted parameter name/type/description fields in
escaped canonical JSON. It does not inspect gold arguments or include defaults,
examples, enums, or non-selected tools. A system-role instruction declares the
HTML-safe canonical JSON user payload inert, non-output, and non-executable;
missing/duplicate matches, oversized contexts, and over-budget prompts fail
closed. Source-row and request digests bind the schema bytes, builder policy,
and tokenizer-based prompt budget for resume safety. This aid is specific to the
instruction-chat interface: TranslateGemma and the raw MADLAD seq2seq request
remain context-free, so candidate comparisons report the interface rather than
attributing differences to checkpoint quality alone.

The selection set deliberately concentrated difficult rows. On 21 rows,
`gpt-5.5-2026-04-23` mechanically accepted 20; the model-assisted, non-native
diagnostic assessment labeled 16 clean, four minor, and zero hard semantic
errors. Qwen3-Next-80B accepted 14,
with six clean, four minor, four hard, and seven mechanical rejects. This
bounded comparison selected the external teacher; it did not validate the full
corpus. The exact rows and decision are in
[`hebrew-teacher-selection.json`](evidence/hebrew-teacher-selection.json) and
[`hebrew-teacher-probe-results.jsonl`](evidence/hebrew-teacher-probe-results.jsonl).
The public row file omits correlatable OpenAI request and response identifiers;
its updated digest is recorded in the selection evidence. The provider's raw
journal remains non-public. Two rows preserve the source dataset's literal
`testpassword`/`securepassword` strings as synthetic protected-span test data;
they are not authentication credentials.

The follow-up Flex smoke translated all 140 selected rows. It accepted 140/140
after 143 provider requests, including three additional audited row attempts.
These are distinct from the new same-row Flex availability retry ledger. The
model-assisted, non-native diagnostic inspection—not independent human
review—labeled 127 clean, 13 minor, and zero hard. Usage was 73,359 input
tokens, zero cached input tokens, 11,618 output tokens, zero reasoning tokens,
and 84,977 total tokens. Applying the pinned public prices and the Flex
multiplier gives **$0.357667500**. This is a calculated list-price estimate,
not an invoice or billing-console observation. The smoke used a 256-token
output limit and historical v1 journal/provider-evidence schemas; it selected
the teacher/runtime but did not validate the final 512-token/v2 production
contract. The run came from a dirty worktree and records only its base Git SHA,
not an immutable producer-diff digest. It is diagnostic, not a full-corpus
result, native-speaker review, provider-weight checksum, accuracy result, or
proof of semantic correctness.

## Reproduction commands

The full config is [`examples/config.v3-he-full.yaml`](https://github.com/AbdelStark/sommelier/blob/main/examples/config.v3-he-full.yaml). Its Hebrew dataset revision is currently provisional (`main`); replace it with the published immutable commit before the evidence run.

Provision the two named Modal secrets without putting credentials in the config
or artifacts:

```bash
uv run modal secret create openai-api-key OPENAI_API_KEY="$OPENAI_API_KEY"
uv run modal secret create huggingface-read-token HF_TOKEN="$HF_TOKEN"
```

Then build the audited Hebrew pairs with the exact dated Responses snapshot.
Choosing `--runtime-backend openai_responses` is the explicit paid-inference
authorization; model-name matching alone never selects a provider:

```bash
SOMMELIER_TIMEOUT_SECONDS=28800 \
uv run modal run --detach remote_translate.py \
  --config examples/config.v3-he-full.yaml \
  --run-id he-v3-translate-full --mode full --max-rows 60000 \
  --target-language he \
  --model-id gpt-5.5-2026-04-23 \
  --model-revision gpt-5.5-2026-04-23 \
  --max-new-tokens 512 --translator-interface instruction_chat \
  --max-model-len 0 --output-decoder standard \
  --runtime-backend openai_responses \
  --openai-service-tier flex --openai-max-workers 8 \
  --openai-list-price-limit-usd 50.00
```

The producer runs in a CPU-only Modal image pinned to Python 3.13.3,
`openai==2.45.0`, and `datasets==5.0.0`. It requests strict structured output
with `store=false`, background mode disabled, truncation disabled, reasoning
effort `none`, a 900-second per-request timeout, SDK retries disabled, and a
stable non-PII safety identifier. It requires the returned model and returned
service tier to equal the request. SDK retries stay at zero. The row pipeline
owns exactly three visible semantic/audit attempts. Separately, exact Flex HTTP
429 `resource_unavailable` responses retry the same row attempt as journaled
`provider_call_attempt` values after fixed 1, 2, 4, 8, and 16 second delays.
They never switch tier or consume another row attempt. It sends provider work
with eight workers and commits 32-row chunks. OpenAI's
[Flex processing guide](https://developers.openai.com/api/docs/guides/flex-processing)
recommends a 15-minute timeout and notes that Flex is slower and can occasionally
return an uncharged `429 Resource Unavailable`; callers must tolerate that
availability tradeoff rather than silently switching tiers. `store=false`
disables stored response retrieval for this workflow; it is not a claim of Zero
Data Retention. The source query and bounded selected-tool projection leave the
Modal boundary and are processed by OpenAI.

Before dataset export or provider construction, a full Hebrew launch validates
the complete v3 project, base/tokenizer, English root, split, formatting,
QLoRA, evaluation, remote, reporting, and tracking contract. The Hebrew dataset
revision is the sole pre-publication exception: the committed `main` placeholder
is accepted until the audited rows are published. The run directory also locks
an exact config and run identity. A matching progress-only attempt can resume;
once accepted `rows.he.jsonl`, `translation_summary.json`, or the publication
manifest exists, that run ID cannot be launched again or overwritten. The
identity reservation is not a distributed mutex: launching the same incomplete
run ID concurrently is unsupported. Operators must wait for an invocation to
exit before starting its identity-matched resume.

The 140-row smoke cost scales naively to **$43.4310535714** (about **$43.43**)
for 17,000 rows before retries. Adding 15% gives **$49.9457116**, so the required
`50.00` USD full-run ceiling is a local pre-request
admission and post-response stop estimate. That extrapolation and ceiling are
not an invoice, billing record, or provider account/project cap; check current
pricing and provider-side spend controls before authorizing the full command.
Earlier Dicta, TranslateGemma, MADLAD, and Qwen runs remain candidate
diagnostics. In particular, the isolated Transformers 4.57.6 MADLAD probe
established only that one local checkpoint/runtime combination loaded and
generated text. None of those outputs substitutes for the selected provider
contract or the full semantic gate.

Create the locked back-translation template from that exact full translation
run, then pull a local copy for review:

```bash
SOMMELIER_GPU=A10G SOMMELIER_TIMEOUT_SECONDS=14400 \
uv run modal run --detach remote_semantic_review.py \
  --translation-run-id he-v3-translate-full

mkdir -p artifacts/translation
uv run modal volume get sommelier-artifacts \
  translation/he-v3-translate-full/ artifacts/translation/
cp artifacts/translation/he-v3-translate-full/translation_semantic_review_template.json \
  artifacts/translation/he-v3-translate-full/translation_semantic_review_reviewed.json
```

The remote producer accepts only a 1-128 character safe run-id component, then
exclusively creates and volume-commits an empty, deliberately invalid file at
the final template path before loading the config, rows, or backtranslation
model. Any existing file or symlink is refused. Failure handling is
deliberately conditional: a caught config/data/model exception removes and
volume-commits only the exact reservation inode if it is still empty; the same
completed translation run can then retry the semantic job without repeating
17,000 provider translations. If the path was replaced or gained any bytes, it
is preserved and the producer remains fail-closed; use a new full translation
run ID rather than deleting possible evidence. A hard process/container crash
cannot run cleanup and therefore leaves an inspectable empty reservation. Only
after confirming that no producer is active, download it and verify it is
exactly zero bytes. Explicit recovery may then remove only that file before
retrying the semantic job:

```bash
RECOVERY_COPY="$(mktemp)"
uv run modal volume get --force sommelier-artifacts \
  translation/<full-id>/translation_semantic_review_template.json \
  "$RECOVERY_COPY"
test ! -s "$RECOVERY_COPY"
rm "$RECOVERY_COPY"
uv run modal volume rm sommelier-artifacts \
  translation/<full-id>/translation_semantic_review_template.json
```

Exclusive creation closes the mounted-filesystem
check/write race; it is not a claim of provider-wide locking across separately
launched Modal containers, so do not launch the same ID concurrently. The pure
local template builder may still replace disposable test fixtures, but it is
not the remote release-evidence boundary.

Fill only the `review` fields in the copied file. Keep the machine template
untouched, then finalize all 200 judgments and regenerate the publication
manifest:

```bash
uv run sommelier data semantic-review-finalize \
  --config examples/config.v3-he-full.yaml \
  --root-input artifacts/translation/he-v3-translate-full/rows.en.jsonl \
  --paired-input artifacts/translation/he-v3-translate-full/rows.he.jsonl \
  --translation-summary artifacts/translation/he-v3-translate-full/translation_summary.json \
  --template artifacts/translation/he-v3-translate-full/translation_semantic_review_template.json \
  --reviewed artifacts/translation/he-v3-translate-full/translation_semantic_review_reviewed.json \
  --out artifacts/translation/he-v3-translate-full/translation_semantic_review.json \
  --reviewer-id <stable-non-secret-reviewer-id>
```

Any critical error fails this publication; fix the translation contract and
regenerate the whole full run rather than deleting the row. After the gate
passes, stage the exact allowlisted dataset bundle. The tracked card template
declares CC-BY-4.0, Salesforce attribution, the machine-translation/provider
boundary, and the survivor-corpus limitation. Replace its pending block with a
release-specific evidence statement and remove its marker before validation:

```bash
DATASET_BUNDLE=artifacts/publication/hebrew-dataset
mkdir -p "$DATASET_BUNDLE"
cp docs/release/hebrew-v3-dataset-card.md "$DATASET_BUNDLE/README.md"
# Edit README.md from verified full evidence; remove only the resolved
# REPLACE_FROM_VERIFIED_DATASET_BUNDLE marker.
for name in rows.he.jsonl translation_summary.json \
  translation_semantic_review_template.json \
  translation_semantic_review.json translation_publication.json; do
  cp "artifacts/translation/he-v3-translate-full/$name" "$DATASET_BUNDLE/$name"
done

# No Hub import or mutation: validate the complete local contract and the
# intended first-publication plan. This assumes the destination is absent.
uv run sommelier release publish-dataset \
  --config examples/config.v3-he-full.yaml \
  --bundle "$DATASET_BUNDLE" \
  --root-input artifacts/translation/he-v3-translate-full/rows.en.jsonl \
  --repo-id abdelstark/sommelier-xlam-single-call-splits-he \
  --commit-message "Publish audited Hebrew v3 paired rows" \
  --create-repo
```

Review that JSON plan, install the isolated publication dependency, and make
the first public commit only from the authenticated release host. If the
reserved repository is still absent, the first execution uses `--create-repo`;
the reviewed validation-only plan above must include it too. Omit the flag from
both passes only when the dedicated repository already has an immutable HEAD.
A pre-existing empty repository is not eligible for an unguarded parentless
commit:

```bash
uv sync --extra publish
uv run --extra publish sommelier release publish-dataset \
  --config examples/config.v3-he-full.yaml \
  --bundle "$DATASET_BUNDLE" \
  --root-input artifacts/translation/he-v3-translate-full/rows.en.jsonl \
  --repo-id abdelstark/sommelier-xlam-single-call-splits-he \
  --commit-message "Publish audited Hebrew v3 paired rows" \
  --execute --create-repo \
  --confirm-repo-id abdelstark/sommelier-xlam-single-call-splits-he \
  --receipt artifacts/publication/hebrew-dataset-receipt.json
```

The publisher refuses symlinks, extra files, raw provider journals, secret-like
content, incomplete semantic/provider evidence, unrelated remote files, and an
existing or inside-bundle receipt. It validates, scans, hashes, and uploads one
private byte snapshot, then downloads every file from the returned immutable
revision and verifies its SHA-256 before writing a verified receipt. The
publication manifest must bind the row identity plus the SHA-256 digests of the
summary, untouched template, and semantic review.
The summary embeds the content-free provider-evidence v2 aggregate and the
SHA-256 of `openai_responses_provider.jsonl`. The raw provider journal remains
in the durable producer artifacts for audit and replay; do not publish it in
the paired dataset.

Replace the provisional Hebrew revision in the config with the exact immutable
`repository.commit_sha` from the verified receipt. Full evidence runs consume
and verify the published rows and all four provenance files; diagnostic
`--translation-run-id` staging is smoke-only.

There are two complementary paid pre-full checks, and neither is full evidence.
The current-contract paired smoke in the
[remote execution guide](../guides/remote-execution.md#pipeline-and-translation-commands)
runs the 512-token, provider-evidence-v2 translation path and then the reduced
smoke pipeline, so it checks provider/data/pipeline integration. Its training
and evaluation config deliberately uses smaller limits and is not the full v3
resource shape. The completed historical 140-row Flex smoke used a 256-token,
v1 evidence contract and does not count as this current-contract smoke. The
L40S check below does the opposite: it exercises the exact full QLoRA resource
shape with synthetic rows, but does not call the provider, consume the
published dataset, or run end-to-end accuracy. Do not describe either check as
run until its own artifact exists, and do not combine them into a release
claim.

Before the two full training/evaluation allocations, exercise the exact v3
memory shape once from the clean producer revision:

```bash
uv run modal run --detach remote_qlora_preflight.py \
  --config examples/config.v3-he-full.yaml \
  --run-id he-v3-l40s-shape-001
```

This paid L40S diagnostic uses synthetic paired English/Hebrew examples only.
It verifies four near-4096-token batch-4 microbatches, one real gradient-
accumulated optimizer update, and one batch-4 evaluation forward under the full
NF4/bfloat16 QLoRA module/checkpointing contract. Each source has exactly one
English and one Hebrew row, the runtime exposes exactly one L40S, and every
`hf_device_map` entry must resolve to CUDA device 0 with no CPU or disk offload.
Its report preserves peak
memory and redacted failure evidence, but is explicitly ineligible for dataset,
accuracy, full-training, TCO-saving, or release claims. The full pipeline still
has to measure the actual dataset and complete all claim gates.

For the two pipeline commands below, `SOMMELIER_TIMEOUT_SECONDS` is Modal's
provider-enforced outer deadline. The config's legacy-named data, train, and
evaluation timeout values are planning estimates used to admit the outer
allocation; they do not stop an individual stage. Runtime evidence records
`per_stage_watchdogs_enforced: false`. The external-v1 arm omits training and
therefore records a 37,800-second planning sum with 48,600 seconds of arithmetic
headroom under the 86,400-second outer deadline. The v3 training arm records an
81,000-second sum and 5,400 seconds of headroom.

```bash
# v1 English-only adapter on the English/Hebrew v3 prompts
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=86400 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.v3-he-full.yaml --mode full --max-rows 60000 \
  --adapter-id abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora \
  --adapter-revision 45a6e2fa3e29f8393ddf1e9bda51a9461b41ee0e \
  --run-id he-v3-v1-baseline

# v3 English+Hebrew QLoRA training and evaluation
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=86400 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.v3-he-full.yaml --mode full --max-rows 60000 \
  --run-id he-v3-full
```

Run the current-contract paired smoke before these full allocations. Its output
proves bounded integration wiring only and must not fill the result table; the
full-shape diagnostic independently proves bounded resource fit only.

After both full evaluation arms exist, build the claim-gated experiment artifact:

The command fails unless the checkout is clean and at the exact immutable
source revision recorded by both full runs. The downloaded bundles live under
the ignored `artifacts/` tree, so they do not dirty that check.

```bash
mkdir -p artifacts/runs
uv run modal volume get sommelier-artifacts \
  artifacts/runs/he-v3-v1-baseline/ artifacts/runs/
uv run modal volume get sommelier-artifacts \
  artifacts/runs/he-v3-full/ artifacts/runs/

uv run sommelier report experiment \
  --base artifacts/runs/he-v3-full/eval/base \
  --v1-en artifacts/runs/he-v3-v1-baseline/eval/adapter \
  --v3-en-he artifacts/runs/he-v3-full/eval/adapter \
  --english-non-inferiority-margin 0.01 \
  --seed 42 --resamples 2000 \
  --out artifacts/experiments/he-v3
```

### Adapter publication handoff

Publication order is evidence-bearing. Stay on the exact clean training SHA
recorded by `artifacts/runs/he-v3-full/train_manifest.json` while finalizing the
experiment, assembling and certifying the adapter, and publishing it. Do this
**before** editing tracked result tables, README claims, or paper text. If the
main checkout has already moved or become dirty, use a separate clean worktree
at that SHA; do not rewrite the run manifests to match a newer commit.

Confirm the checkout, then assemble the exact allowlisted bundle under the
ignored `artifacts/` tree. The run manifest and resolved config come from the
run root; PEFT files come from `train/adapter`; the experiment report comes
from the finalizer output above. Training metrics are evidence in the run tree,
but are not an allowed adapter-bundle file.

```bash
V3_RUN=artifacts/runs/he-v3-full
EXPERIMENT=artifacts/experiments/he-v3/experiment_report.json
ADAPTER_BUNDLE=artifacts/publication/hebrew-adapter
TRAINING_SHA="$(jq -er '.git_commit' "$V3_RUN/train_manifest.json")"

test "$(git rev-parse HEAD)" = "$TRAINING_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=normal)"
test ! -e "$ADAPTER_BUNDLE"
mkdir -p "$ADAPTER_BUNDLE/adapter"

cp docs/release/hebrew-v3-adapter-card-template.md "$ADAPTER_BUNDLE/README.md"
cp licenses/THIRD_PARTY.md "$ADAPTER_BUNDLE/THIRD_PARTY.md"
cp licenses/LICENSE-NVIDIA-OPEN-MODEL.txt \
  licenses/LICENSE-LLAMA-3.1.txt licenses/NOTICE "$ADAPTER_BUNDLE/"
cp "$V3_RUN/config.resolved.yaml" \
  "$V3_RUN/manifest.json" \
  "$V3_RUN/train_manifest.json" \
  "$ADAPTER_BUNDLE/"
cp "$EXPERIMENT" "$ADAPTER_BUNDLE/experiment_report.json"
cp "$V3_RUN/train/adapter/README.md" \
  "$V3_RUN/train/adapter/adapter_config.json" \
  "$V3_RUN/train/adapter/adapter_model.safetensors" \
  "$ADAPTER_BUNDLE/adapter/"

for name in added_tokens.json chat_template.jinja special_tokens_map.json \
  tokenizer.json tokenizer.model tokenizer_config.json; do
  if test -f "$V3_RUN/train/adapter/$name"; then
    cp "$V3_RUN/train/adapter/$name" "$ADAPTER_BUNDLE/adapter/$name"
  fi
done
```

Fill the copied model card only from the bundle. These commands derive every
required identity and expose the claim decisions; retain only claims whose
gates passed, then remove every `REPLACE_FROM_VERIFIED_BUNDLE` marker.

```bash
uv run python -c \
  'from pathlib import Path; from sommelier.evaluation.generate import adapter_tree_sha256; print(adapter_tree_sha256(Path("artifacts/publication/hebrew-adapter/adapter")))'
shasum -a 256 "$ADAPTER_BUNDLE/experiment_report.json"
jq -er '.git_commit' "$ADAPTER_BUNDLE/train_manifest.json"
uv run python -c \
  'from pathlib import Path; from sommelier.config import load_config; print(load_config(Path("artifacts/publication/hebrew-adapter/config.resolved.yaml")).dataset_for("he").dataset_revision)'
jq '{all_claims_passed, approved_claims, claims}' \
  "$ADAPTER_BUNDLE/experiment_report.json"
```

Run preflight last: it writes `release_preflight.json` and certifies the final
tree while excluding only that self-referential report. A passing preflight is
the end of bundle mutation.

```bash
SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1" \
uv run sommelier release preflight \
  --config "$ADAPTER_BUNDLE/config.resolved.yaml" \
  --artifact-root "$ADAPTER_BUNDLE"
```

For a destination that does not yet exist, include `--create-repo` in both the
validation-only plan and the executed command so the reviewed plan matches the
mutation. Omit it from both commands only when the dedicated repository already
has an immutable HEAD; a pre-existing empty repository is not an eligible
parentless target.

```bash
ADAPTER_REPO=abdelstark/Llama-3.1-Nemotron-Nano-8B-xlam-tool-calling-he-en-lora

# Validation only: no Hub import or mutation.
uv run sommelier release publish-adapter \
  --bundle "$ADAPTER_BUNDLE" \
  --repo-id "$ADAPTER_REPO" \
  --commit-message "Publish claim-gated Hebrew v3 QLoRA adapter" \
  --create-repo

# Deliberate first publication after reviewing the JSON plan.
uv sync --extra publish
test ! -e artifacts/publication/hebrew-adapter-receipt.json
uv run --extra publish sommelier release publish-adapter \
  --bundle "$ADAPTER_BUNDLE" \
  --repo-id "$ADAPTER_REPO" \
  --commit-message "Publish claim-gated Hebrew v3 QLoRA adapter" \
  --execute --create-repo \
  --confirm-repo-id "$ADAPTER_REPO" \
  --receipt artifacts/publication/hebrew-adapter-receipt.json
```

Require receipt status `verified`, record its immutable
`repository.commit_sha`, and only then edit tracked result tables, README
claims, paper text, or release links. A failed attempt deliberately owns its
fresh receipt path; inspect that journal and the Hub before choosing a new path
or retrying.

## Result placeholders

| Evidence | Required artifact | Status |
|----------|-------------------|--------|
| Teacher selection and 140-row Flex smoke | `evidence/hebrew-teacher-selection.json` plus checksummed diagnostic artifacts | Diagnostic complete; not full evidence |
| Translation yield, protected-span/script/bidi drops, provider identity/usage/list-price calculation | `translation_summary.json` (`sommelier.translation_summary.v2`, nested `sommelier.openai_provider_evidence.v2`) | Pending full run |
| Preregistered sample and locked Helsinki-NLP OPUS-MT back-translations | `translation_semantic_review_template.json` (`sommelier.translation_semantic_review_template.v1`) | Pending full run |
| Preregistered semantic sample and back-translation judgments | `translation_semantic_review.json` (`sommelier.translation_semantic_review.v1`) | Pending full run |
| Published row/summary/template/review binding | `translation_publication.json` (`sommelier.translation_publication_manifest.v1`) | Pending full run |
| English↔Hebrew token ratios and projected workload | `analysis/tokenization/tokenizer_tax_report.json` | Pending full run |
| Base and v3 adapter metrics with paired intervals | `report/comparison_report.json` (`sommelier.comparison_report.v3`) | Pending full run |
| v1 versus v3 Hebrew uplift and English non-inferiority | gated three-arm `experiment_report.json` | Pending full run |
| Bounded QLoRA/TCO evidence | `experiment_report.json.sovereign_tco_evidence` (`sommelier.sovereign_tco_evidence.v1`) | Pending full run |

Do not replace “Pending full run” with hand-copied console values. Fill those
rows from the checksummed full artifacts, link their immutable publication
revisions, and keep any unavailable cost field explicitly unavailable. The
diagnostic row remains labeled diagnostic even after full evidence exists.
