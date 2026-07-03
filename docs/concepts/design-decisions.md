# Design decisions

Sommelier makes one bet: an evaluation result is worth publishing only if the process that produced it cannot quietly go wrong. Every decision below trades convenience for that property. In each case the rejected alternative was real, usually easier, and usually what a quick fine-tuning script would do. This page states what was rejected and what the rejection buys, so you can judge whether the trade was sound.

## One config file, strictly validated

A run is driven by a single YAML file validated into a typed model, and every section of that model rejects unknown keys. Misspell `learning_rate` and the command fails with a config error naming the problem, instead of training with the default while you believe your value applied. The loader writes the fully resolved config to `config.resolved.yaml`, and the SHA-256 of that file becomes the config digest recorded in every stage manifest and checked by [the comparison gate](determinism.md).

Rejected: lenient parsing, and CLI flags as the primary interface. Lenient parsing turns a typo in a hyperparameter name into a silent fallback: the run completes, the numbers look plausible, and the config you thought you tested never ran. Flags are hard to hash and hard to replay across remote stages; a file with a digest is both. The full field list is in the [configuration reference](../reference/configuration.md).

## Exactly one tool call per example

Version 1 trains and scores exactly one call per request. Rows whose gold answer contains more than one call are dropped during `data prepare` under their own `multi_call_answer` reason in the [drop summary](data.md), and the parser accepts a single call object or a one-element array, nothing else.

Rejected: scoring multi-call plans. Scoring a plan requires alignment policy: does call order matter, is partial credit given, how do calls match when a name repeats. Each of those choices is a place where two reasonable implementations produce different numbers from the same generations, which makes the headline comparison arguable. The scope was narrowed until the measurement is not. There is also an internal consistency argument: because the parser rejects multi-call outputs, keeping multi-call golds in the data would score a perfectly faithful model as a failure.

## Tools in the system message, through the tokenizer's own template

Every example is a three-message chat: a system message carrying the configured instruction plus the tool schemas as canonical JSON, the user query, and an assistant message containing only the canonical JSON of the gold call. Rendering goes through the tokenizer's own `apply_chat_template`, loaded at the pinned `model.tokenizer_revision`; the config's `formatting.template_policy` admits exactly one value.

Rejected: hand-rolled prompt templates, and passing tool schemas through side channels instead of message text. The base model consumes rendered chat text either way, so the honest comparison is over that text, in the format the model's own instruction tuning expects; a hand-written template is a second formatting implementation that can drift from it. Keeping tools inside the messages makes the whole prompt one string, which is what lets `prompt_sha256` prove prompt identity between training and both evaluations. Canonical JSON (sorted keys, compact separators) makes equal inputs byte-equal, so those digests mean something.

## Completion-only loss with a proven boundary

Training masks every prompt token and computes loss only on the assistant target. That requires knowing exactly where the prompt ends, and Sommelier refuses to assume it. The format stage fails if the rendered `full_text` does not begin with `prompt_text`, and the [training collator](training.md) re-proves the boundary at the token level for every example, failing when the tokenizer merges tokens across it.

Rejected: falling back to full-sequence loss when the boundary cannot be proven. The fallback changes the training objective (the model spends capacity learning to reproduce prompts) without changing anything visible in the config, so two runs could train different objectives while their artifacts claim they are the same run. A failed run costs a rerun. An invisible objective change costs the comparison.

## A parser that never repairs

The [evaluation parser](evaluation.md) extracts the first balanced JSON object or array from the generation and accepts exactly `{"name": <string>, "arguments": <object>}`, or a one-element array containing such an object. Surrounding prose is tolerated; nothing inside the extracted span is repaired. Extra keys fail, empty names fail, non-object arguments fail, multi-call arrays fail, malformed JSON fails. Failures are classified (`no_json`, `invalid_json`, `invalid_shape`), counted against every metric, and the raw generations are kept for audit.

Rejected: lenient extraction with repair (fixing trailing commas, coercing shapes, retrying variants). Repair moves work from the model to the harness, by an amount you cannot measure and that differs between base and adapter. Producing valid structured output is itself a thing being measured: in the [reference run](../results/reference-run.md), valid JSON rate is 0.916 base versus 1.000 adapter, and a repairing parser would erase part of exactly that signal.

## Comparison gated on digests, not discipline

`report compare` refuses to produce a report unless the base and adapter evaluation reports agree on the config digest, split name, test split digest, ordered prompt-set digest, parser version, and decoding config, and unless their metric names match. It also re-hashes the run's own `config.resolved.yaml` to confirm both reports belong to the run it is writing into.

Rejected: trusting that two evaluations run from the same checkout "should be" comparable. Incomparability is invisible in the output: a metrics table computed from mismatched prompts looks identical to one computed from matched prompts. The gate converts the one failure mode you cannot see into one you cannot miss. [Determinism and the comparison gate](determinism.md) walks through each checked field.

## Files and manifests, not a tracking service

Every stage writes a schema-versioned manifest listing its inputs and outputs, each with path, kind, schema version, SHA-256, and size. Readers fail closed on missing or unsupported schema versions. Experiment tracking exists as an option, is disabled by default, and is a strict no-op when disabled; the [local artifacts](artifacts.md) are the authoritative record either way.

Rejected: a hosted dashboard as the source of truth. Reproduction that depends on a third-party service also depends on that service's availability, its retention policy, and your access to someone's account. Evidence for a public claim has to travel with the claim; a directory of checksummed files can be archived and audited by anyone. Schemas and layout are specified in the [artifacts reference](../reference/artifacts.md).

## Heavy dependencies stay out of the core

`import sommelier` never imports torch, cudf, transformers, or any other GPU-adjacent package. Heavy imports happen inside stage execution, an import-discipline test walks every module of the package in a clean interpreter and asserts none of the forbidden modules load, and the remote images are declared per dependency stack: a data image with the GPU dataframe stack, a training image with the quantization and adapter stack, an evaluation image that deliberately carries less, and a serving image of its own. The [remote entrypoints](../guides/remote-execution.md) wrap the same stage functions the local CLI runs.

Rejected: one image with everything, and eager imports at module top. Eager imports make every fixture test pay the GPU-stack tax and make "the tests pass locally" a much weaker statement. A single fat image hides which stage actually needs what. And remote-only stage code was rejected because the local test suite would then no longer exercise the code that runs on the GPU.

## Nothing retries or tunes itself

When training hits out-of-memory, it fails with a resource error whose message names the current values of `train.per_device_batch_size`, `train.gradient_accumulation_steps`, `train.max_sequence_length`, and `remote.gpu`, and says those are the fields to change. It does not retry with a smaller batch, and no stage anywhere retries with altered settings.

Rejected: automatic batch-size search and silent retry. Auto-tuning turns the config from a record into a suggestion: two runs with the same digest could have trained with different effective batch sizes, which changes optimization and cost with no trace in the artifacts. When a human changes the config instead, the digest changes, and the difference is part of the record. The [error model](../reference/errors.md) applies the same rule everywhere: fail with the exact fields to change, never adapt quietly.

## Serving is deliberately illustrative

`sommelier serve adapter` exists so you can inspect the adapter through a familiar request shape, not so you can deploy it. Requests must set temperature to exactly 0.0, unknown request fields are rejected (there is no streaming flag to pass), responses carry the raw text plus the parsed call and parse status from the same parser evaluation uses, and prompts are built by the same function formatting and evaluation call, so what serving shows is what was measured. The endpoint describes itself as optional and illustrative, and a docs test asserts the README says so too.

Rejected: shipping a production-shaped server. A server that looks production ready gets used that way, and this one has no authentication story of its own and no throughput or latency claims. Attaching unaudited operational qualities to a project whose entire value is audited claims would be self-defeating. The [serving guide](../guides/serving.md) states the limits.

## Release gates as code

A release runs a preflight that evaluates every gate, writes `release_preflight.json` with a status and an evidence string per gate, and only then raises if anything failed, so the evidence survives the failure. The gates check that the license and third-party notice files exist, that the configured base model and dataset appear in the notices, that the "Built with Llama" notice required for derived artifacts is present, that the operator acknowledged the base model license by setting `SOMMELIER_ACK_BASE_MODEL_LICENSE` to the base model id, that the dependency lock exists, and that a secret scan of the artifact tree comes back clean.

Rejected: a release checklist in prose. Prose gets skimmed, especially at the end of a project when the result looks good and the checklist stands between you and publishing. Machine-checked gates cannot be skimmed, and the JSON report proves they ran. Details in [licensing](../project/licensing.md), [security](../project/security.md), and the [v1.0 checklist](../release/v1.0-checklist.md).

## What these decisions cost

Strictness has a bill, and it is worth stating plainly:

- **Iteration is slower.** Every config typo is a failed command. Every schema change means updating readers, because they fail closed on versions they do not know.
- **There are more files.** A run leaves manifests, a drop summary, per-generation records, per-stage logs, and two report formats. The artifact tree is the evidence, and evidence takes space.
- **Failures are stricter.** The boundary proof, the comparison gate, and the release preflight can each kill a run or a release that would probably have been fine.

The trade was taken because the alternative failure mode is a plausible-looking wrong number. In an evaluation project, a wrong number that looks right is strictly worse than a loud failure: the loud failure costs a rerun, while the wrong number costs the project its reason to exist. Every decision on this page is the same decision made in a different place: move the failure earlier, make it louder, and leave a record.
