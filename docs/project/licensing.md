# Licensing

Sommelier's output is a derivative of other people's work: an adapter trained on an NVIDIA model that is itself built on Meta's Llama 3.1, using a Salesforce dataset. Multilingual releases also declare their forward-translation provider and back-translation model. Each layer carries obligations. The release preflight checks the project-level subset, while the dataset and adapter publishers enforce their artifact-specific contracts. This page describes that chain and those gates.

!!! warning "Not legal advice"
    The gates are engineering support for compliance: they verify that the recorded obligations are present and acknowledged, not that your use is lawful. Re-verify each source card before publishing any derived artifact.

## The chain

| Artifact | Terms | Key obligations |
|----------|-------|-----------------|
| Sommelier code | MIT | [LICENSE](https://github.com/AbdelStark/sommelier/blob/main/LICENSE) at the repo root |
| Base model [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) | NVIDIA Open Model License, with the Llama 3.1 Community License applying because the model is built on Meta Llama 3.1 | Obligations flow to derived artifacts, below |
| Source dataset [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | CC-BY-4.0 | Attribute Salesforce; indicate filtering/deduplication/splitting and, for multilingual derivatives, query translation |
| Hebrew v3 dataset teacher, OpenAI Responses `gpt-5.5-2026-04-23` | Applicable OpenAI API/account terms; no provider weights are distributed | Verify current account/data-use terms, disclose the external teacher, and record the exact dated snapshot, service tier, request boundary, and usage evidence |
| Semantic-review back-translator [Helsinki-NLP/opus-mt-tc-big-he-en](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-he-en) | CC-BY-4.0 | Attribute Helsinki-NLP and OPUS-MT, record the exact preregistered revision and review-only use, and indicate that the model is not redistributed here |
| Trained adapters | Derivatives of the base model | Inherit both model licenses |

The ledger for all of this is [licenses/THIRD_PARTY.md](https://github.com/AbdelStark/sommelier/blob/main/licenses/THIRD_PARTY.md). It records the source card, governing terms, exact Hebrew provider snapshot/back-translator pin, concrete obligations, and runtime package licenses. The preflight checks the configured base model and root dataset against this file; its v2 identity records every configured dataset. The translation publication artifacts separately bind the provider request identity, independent back-translator revision, and review boundary.

The forward-translation runtime is also pinned evidence. Hebrew v3 uses a
CPU-only Modal image with Python 3.13.3, `openai==2.45.0`, and
`datasets==5.0.0`. The producer records the exact dated model identity, Flex
tier, 900-second request timeout, returned model/tier, request contract, usage,
content-free v2 journal aggregate, and clean implementation revision. This
makes the client-side boundary auditable; it does not expose provider weights,
prove byte-identical regeneration, or establish translation quality on the
full Hebrew corpus.

The semantic-review producer is also a pinned evidence boundary. Its runtime
is exactly Python 3.13.3, torch 2.11.0, transformers 5.13.1, tokenizers 0.22.2,
accelerate 1.14.0, huggingface-hub 1.22.0, sentencepiece 0.2.2, and sacremoses
0.1.1. The producer must run from the same clean immutable Git SHA recorded by
the full translation, and the template records whether the boundary was Modal
GPU or local plus the hardware identity; remote evidence also records the
dispatched timeout. These checks do not replace the CC-BY-4.0 attribution
obligations of Helsinki-NLP's OPUS-MT checkpoint or the licenses of its runtime
packages; they make the machine-produced review evidence attributable and
reproducible. The model card's BLEU 44.1 (FLORES-101 devtest) and 53.8
(Tatoeba test) figures are self-reported upstream benchmarks, not Sommelier
review accuracy and not evidence for this corpus.

## What being a Llama derivative means

An adapter trained by this pipeline is a derivative of a model built on Llama 3.1, so the recorded obligations include displaying "Built with Llama" prominently on distributed artifacts, including a copy of the Llama 3.1 license terms and the prescribed `NOTICE` attribution, and respecting the restriction on using outputs to improve non-Llama models. The published reference adapter carries "Built with Llama" ([abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora)); the new publisher additionally requires every new model repository basename and card title to begin with the literal `Llama`. The [published splits dataset](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits) stays under CC-BY-4.0 with Salesforce attribution.

The repository tracks the reviewed October 24, 2025 NVIDIA terms in
`licenses/LICENSE-NVIDIA-OPEN-MODEL.txt`, Meta's Llama 3.1 agreement in
`licenses/LICENSE-LLAMA-3.1.txt`, and both lineage notices in
`licenses/NOTICE`. Their authoritative sources remain the
[NVIDIA agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
and [Meta license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
Because upstream terms can change, a later release must re-review those pages;
the adapter publisher requires byte-exact copies of the reviewed local files
rather than accepting a link or a license name in prose.

## The preflight gates

`sommelier release preflight --config examples/config.full.yaml` evaluates eight gates from [`sommelier/release.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/release.py) and, when the artifact root is a safe writable directory, writes `release_preflight.json` there *before* raising for failed gates. Its v2 report binds the normalized config; exact model, tokenizer, and ordered dataset revisions plus their immutability decisions; producer Git commit and cleanliness decision; `uv.lock`; and a coherent identity of every regular file in the selected tree. Files are read again before certification, so same-size mutation is detected without trusting timestamp metadata. Only the root-level report itself is excluded from that tree digest. Run the command from the exact producer repository root: the project license, notices, lock, and Git identity are discovered from the current directory. The configured default artifact root must resolve inside the config directory; an explicit `--artifact-root` is an intentional operator-selected target for certifying a downloaded or curated tree elsewhere:

| Gate | Passes when |
|------|-------------|
| `project_license` | `LICENSE` is a regular file whose captured bytes match the certified producer commit through Git's configured clean filters |
| `third_party_notices` | `licenses/THIRD_PARTY.md` is a UTF-8 regular file whose captured bytes match the certified producer commit through Git's configured clean filters |
| `base_model_obligations` | The configured `model.base_model_id` appears in those certified notice bytes |
| `dataset_license` | The configured root dataset id appears in those certified notice bytes |
| `derived_artifact_notice` | The literal `Built with Llama` appears in those certified notice bytes |
| `base_model_license_ack` | `SOMMELIER_ACK_BASE_MODEL_LICENSE` equals the configured `base_model_id` |
| `dependency_lock` | `uv.lock` is a regular file whose captured bytes match the certified producer commit through Git's configured clean filters; no exact Git repository means failure |
| `artifact_secret_scan` | The [secret scanner](security.md) finds nothing under the artifact root (skipped if no artifacts exist) |

Failure behavior follows the [error contract](../reference/errors.md): a failing `artifact_secret_scan` raises `SecurityPolicyError` (exit 5); any other failing gate raises `ExternalDependencyError` (exit 3). Both name the failing gates and point at the written report. There is one deliberate exception: if the artifact root is a symlink, non-directory, uninspectable, or cannot be created safely, preflight refuses to write through it and the error explicitly says that no report was written.

Two of the gates deserve a word of explanation. `dependency_lock` looks like a build concern, but it belongs here: the runtime package licenses in `THIRD_PARTY.md` were read from the locked environment, so the ledger is only meaningful against a pinned dependency set. The source certification also rejects Git index entries marked assume-unchanged or skip-worktree; those flags cannot turn hidden working-tree changes into a clean claim. Git's clean-filter hash is used instead of comparing raw repository blobs, so legitimate CRLF and configured-filter checkouts still certify. Every gate's evidence string passes through the same write-time redaction as logs and manifests, so the preflight report can name filesystem paths without failing its own secret scan.

The acknowledgement gate is deliberately awkward. Accepting a model license is a human decision, so it cannot live in a config file; it is an environment variable, and its value must be the exact base model id. For adapter publication, first assemble the complete allowlisted bundle except for `release_preflight.json`, from the exact clean producer commit. Then run preflight against that final directory using its copied resolved config:

```bash
export SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
uv run sommelier release preflight \
  --config artifacts/publication/hebrew-adapter/config.resolved.yaml \
  --artifact-root artifacts/publication/hebrew-adapter
```

Tying the value to the model id means the acknowledgement names what was acknowledged. Change the base model in your config and the old acknowledgement stops passing. A standalone preflight records revision immutability and source cleanliness; the adapter publisher is the strict consumer that requires both to be true. Do not add, remove, or edit a bundle file after preflight: adapter validation recomputes the v2 identity and rejects config, source, lock, revision, or tree drift.

## Publication is a deliberate act

Nothing in the pipeline publishes weights or data. A full run leaves the
adapter under `runs/<run_id>/train/adapter/` on your disk or Modal volume, and
it stays there. The separate `release publish-dataset` and
`release publish-adapter` commands are validation-only by default. They mutate
the Hugging Face Hub only with `--execute`, an exact repeated repository id, a
new receipt path outside the bundle, and—only for an absent repository—an explicit
`--create-repo`. They reject unexpected bundle/remote files, symlinks, raw
provider journals, incomplete license/provenance evidence, and secret-like
content. The exact validated bytes are copied into a private upload snapshot;
later changes to caller paths cannot change the commit. An executed commit is
not recorded as verified until every allowed file is downloaded from the
returned immutable revision and matches its local SHA-256.

The dataset publisher enforces CC-BY-4.0 plus Salesforce and machine-
translation attribution. The adapter publisher requires the passing preflight,
exact reviewed NVIDIA and Llama agreement copies, `NOTICE`, the prominent
`Built with Llama` text, the exact base identity, an unmerged LoRA-only
safetensors tree, succeeded manifests, and the final claim-gated experiment
report. The [CLI reference](../reference/cli.md#release-publish-dataset)
documents the exact transaction. The [v1.0 checklist](../release/v1.0-checklist.md)
walks the order of operations, and the [reproduction guide](../guides/reproduction.md)
covers the acknowledgement in context.
