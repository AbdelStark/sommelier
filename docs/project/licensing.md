# Licensing

Sommelier's output is a derivative of other people's work: an adapter trained on an NVIDIA model that is itself built on Meta's Llama 3.1, using a Salesforce dataset. Each layer carries obligations, and the release preflight turns every obligation that can be machine-checked into a hard gate. This page describes the chain and the gates.

!!! warning "Not legal advice"
    The gates are engineering support for compliance: they verify that the recorded obligations are present and acknowledged, not that your use is lawful. Re-verify each source card before publishing any derived artifact.

## The chain

| Artifact | Terms | Key obligations |
|----------|-------|-----------------|
| Sommelier code | MIT | [LICENSE](https://github.com/AbdelStark/sommelier/blob/main/LICENSE) at the repo root |
| Base model [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) | NVIDIA Open Model License, with the Llama 3.1 Community License applying because the model is built on Meta Llama 3.1 | Obligations flow to derived artifacts, below |
| Source dataset [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | CC-BY-4.0 | Attribute Salesforce; indicate changes made (Sommelier filters, deduplicates, and splits rows; it does not rewrite row content) |
| Trained adapters | Derivatives of the base model | Inherit both model licenses |

The ledger for all of this is [licenses/THIRD_PARTY.md](https://github.com/AbdelStark/sommelier/blob/main/licenses/THIRD_PARTY.md). It records the source card, the governing terms, and the concrete obligations for each dependency, plus a table of runtime package licenses. The preflight checks against this file by content, so an obligation that is not written down fails the gate.

## What being a Llama derivative means

An adapter trained by this pipeline is a derivative of a model built on Llama 3.1, so the recorded obligations include displaying "Built with Llama" prominently on distributed artifacts, including a copy of the Llama 3.1 license terms, and respecting the restriction on using outputs to improve non-Llama models. The published reference adapter follows this: it is named with the required `llama` prefix ([abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora)) and carries "Built with Llama". The [published splits dataset](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits) stays under CC-BY-4.0 with Salesforce attribution.

## The preflight gates

`sommelier release preflight --config examples/config.full.yaml` evaluates eight gates from [`sommelier/release.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/release.py) and writes `release_preflight.json` at the artifact root *before* raising, so the evidence survives a failure:

| Gate | Passes when |
|------|-------------|
| `project_license` | `LICENSE` exists at the project root |
| `third_party_notices` | `licenses/THIRD_PARTY.md` exists |
| `base_model_obligations` | The configured `model.base_model_id` appears in `THIRD_PARTY.md` |
| `dataset_license` | The configured `dataset.dataset_id` appears in `THIRD_PARTY.md` |
| `derived_artifact_notice` | The literal `Built with Llama` appears in `THIRD_PARTY.md` |
| `base_model_license_ack` | `SOMMELIER_ACK_BASE_MODEL_LICENSE` equals the configured `base_model_id` |
| `dependency_lock` | `uv.lock` exists |
| `artifact_secret_scan` | The [secret scanner](security.md) finds nothing under the artifact root (skipped if no artifacts exist) |

Failure behavior follows the [error contract](../reference/errors.md): a failing `artifact_secret_scan` raises `SecurityPolicyError` (exit 5); any other failing gate raises `ExternalDependencyError` (exit 3). Both name the failing gates and point at the written report.

Two of the gates deserve a word of explanation. `dependency_lock` looks like a build concern, but it belongs here: the runtime package licenses in `THIRD_PARTY.md` were read from the locked environment, so the ledger is only meaningful against a pinned dependency set. And every gate's evidence string passes through the same write-time redaction as logs and manifests, so the preflight report can name filesystem paths without failing its own secret scan.

The acknowledgement gate is deliberately awkward. Accepting a model license is a human decision, so it cannot live in a config file; it is an environment variable, and its value must be the exact base model id:

```bash
export SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
```

Tying the value to the model id means the acknowledgement names what was acknowledged. Change the base model in your config and the old acknowledgement stops passing.

## Publication is a deliberate act

Nothing in the pipeline publishes weights. A full run leaves the adapter under `runs/<run_id>/train/adapter/` on your disk or Modal volume, and it stays there. Publishing is a manual upload, done after preflight passes, with the notices in place. The [v1.0 checklist](../release/v1.0-checklist.md) walks the order of operations, and the [reproduction guide](../guides/reproduction.md) covers the acknowledgement in context.
