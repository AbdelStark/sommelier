# Sommelier: Product Requirements Document

**Project:** Sommelier
**Type:** Open, reproducible reference implementation
**Companion document:** see `SPEC.md` for the technical design.

Sommelier fine-tunes a small open language model to produce reliable, schema-valid tool calls (function calling). It uses `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` as the base model, `Salesforce/xlam-function-calling-60k` as the dataset, RAPIDS for GPU-accelerated data preparation, and Modal for single-GPU training and evaluation.

> A sommelier selects the one right option from a long list and pairs it with the right accompaniments. A function-calling model must do the same: select the right tool and fill the right arguments.

---

## 1. Purpose

Sommelier is a minimal, end-to-end example of post-training a small open model for tool calling. It is designed to be built and understood incrementally, one component at a time, from raw data to a measured result, and to be reproducible by anyone with a single GPU.

## 2. Background and problem statement

Agentic applications depend on a model reliably turning a user request plus a set of available tools into a correct, structured tool call. Two common approaches each have drawbacks:

- Calling a large frontier model on every request is accurate but expensive per call and offers limited control over behaviour and hosting.
- Using a small general model off the shelf is cheap but often not reliable enough on a specific set of tool schemas.

Fine-tuning a small open model on tool-calling data closes much of that gap. The result can be reliable on the target schemas, cheap to run, controllable, and self-hostable. Sommelier demonstrates that workflow end to end on a single GPU.

## 3. Goals

- **G1.** Produce a fine-tuned model that emits valid, correct tool calls from natural-language requests and tool schemas.
- **G2.** Prepare and handle the dataset with RAPIDS (GPU-accelerated).
- **G3.** Keep the whole pipeline runnable on a single GPU, cheaply and quickly.
- **G4.** Measure and report a clear improvement of the fine-tuned model over the base model, with a reproducible evaluation.
- **G5.** Provide a clean, well-documented, incremental codebase that others can follow and reproduce.

## 4. Non-goals

- Pretraining, full-parameter fine-tuning, or multi-GPU and multi-node training.
- Beating frontier models in absolute terms. The metric of interest is the improvement over the base model on the target task.
- Production hardening. Serving is optional and illustrative.

## 5. Users and use cases

- **Developers building agentic products** who want a cheap, controllable, self-hostable tool-calling model fine-tuned on their own schemas. Sommelier is a template they can adapt.
- **Practitioners learning the workflow** who want a small, complete, readable example that spans data preparation, fine-tuning, and evaluation.

Representative user stories:

- As a developer, I can prepare a tool-calling dataset with RAPIDS and produce clean train, validation, and test splits.
- As a developer, I can fine-tune the base model with QLoRA on a single GPU with one command.
- As a developer, I can run one evaluation that reports base versus fine-tuned metrics in a single table.
- As a learner, I can build the project phase by phase, each phase producing a visible, checkable result.

## 6. Functional requirements

- **FR1 Data preparation.** Download the dataset, then clean, deduplicate, filter, validate, and split it using RAPIDS (cuDF). Output train, validation, and test files.
- **FR2 Formatting.** Render each example into the base model's chat template, with tools in the system message, the request in the user message, and the gold call as the target.
- **FR3 Training.** Fine-tune the base model with QLoRA (4-bit) on a single GPU, with configurable hyperparameters.
- **FR4 Evaluation.** Run the base and fine-tuned models on the held-out test set with deterministic decoding, and report the metrics in section 7.
- **FR5 Reproducibility.** Provide a single configuration file, a fixed random seed, and pinned dependencies.
- **FR6 Serving (optional).** Serve the fine-tuned model behind an OpenAI-compatible endpoint.

## 7. Success metrics

Measured on a held-out test set, comparing the fine-tuned model against the base model:

| Metric | Definition | Target |
|--------|------------|--------|
| Valid-JSON rate | Output parses as a JSON tool call | at or near 100 percent (fine-tuned) |
| Function-name accuracy | Predicted tool name matches gold | clear gain over base |
| Argument exact-match | Predicted arguments equal gold | clear gain over base |
| Argument F1 | Per-key precision and recall | reported, for partial credit |
| Full-call exact-match | Name and arguments both correct | reported |

Project-level constraints: the full pipeline runs on a single GPU in a few GPU-hours, at low cost (target under 30 USD including iteration).

## 8. Scope

**In scope (v1):** English tool calling, one dataset, RAPIDS data preparation, QLoRA fine-tune, automatic evaluation, and a base versus fine-tuned report.

**Stretch (v2):** optional serving with vLLM, an optional multilingual evaluation slice (the base model supports several languages, so cross-lingual generalization can be measured), and an optional run on a standard function-calling benchmark.

**Out of scope:** production serving and scaling, agent orchestration beyond single tool calls, and datasets other than the chosen one, though the pipeline is written to be swappable.

## 9. Milestones (incremental build plan)

| Phase | Deliverable | Result to check |
|-------|-------------|-----------------|
| 0 Setup | Repo, config, accounts, Modal auth | `modal run` executes a function on a GPU |
| 1 Data | RAPIDS preparation produces splits | train, val, test files with expected counts |
| 2 Format | Chat-template formatting | a few rendered examples look correct |
| 3 Baseline | Base-model evaluation | a baseline metrics table |
| 4 Train | QLoRA fine-tune | a saved adapter and a training curve |
| 5 Evaluate | Fine-tuned evaluation and comparison | a base versus fine-tuned table |
| 6 Stretch | Serving, multilingual slice, benchmark | an endpoint and extra metrics |

Each phase is self-contained and produces a visible artifact, so the project can be built and understood one step at a time.

## 10. Assumptions and constraints

- Access to a single NVIDIA GPU with compute capability 7.0 or newer and CUDA 12, provided through Modal.
- Accounts for Modal, Hugging Face, and Weights and Biases.
- Acceptance of the base model license and the dataset license.

## 11. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Base model already competent at tool calling, so gains look small | choose specific schemas, report argument exact-match and F1, and include the valid-JSON rate where base models often fail |
| Train and test leakage inflating results | deduplicate on the request field before splitting |
| GPU out-of-memory | 4-bit QLoRA, modest batch size with gradient accumulation, gradient checkpointing, or a larger-memory GPU |
| Environment or CUDA mismatch | pin dependencies and follow the image definitions in `SPEC.md` |
| License or attribution omissions | follow section 13 |

## 12. Future work

- Swap in other base models or datasets through configuration.
- Add multi-call and multi-turn tool use.
- Add preference optimization on top of supervised fine-tuning.
- Package the model for production serving.

## 13. License and attribution

- **Base model:** governed by the NVIDIA Open Model License and the Llama 3.1 Community License. Any derived artifact must include a "Built with Llama" notice. Verify the current terms on the model card.
- **Dataset:** verify and cite the dataset license as stated on its dataset card.
- **This repository:** choose and state a license for the project code.