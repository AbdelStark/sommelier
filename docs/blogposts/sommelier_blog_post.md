# Sommelier: Teaching a Small Open Model to Pick the Right Tool

*What I learned post-training Nemotron with QLoRA on a single GPU: the numbers, the bugs, and why the whole stack being open matters more than people realize.*

---

A sommelier does one thing supremely well: from a long list, select the one right option and pair it with the right accompaniments.

That is exactly what a model must do in an agentic system. Given a request and a list of available tools, select the right tool and fill the right arguments. It is the atomic skill of the agentic era. Every plan, every multi-step workflow, every autonomous loop decomposes into this one act, repeated thousands of times. If the tool call is wrong, nothing downstream can save you. If it is right, cheap, and fast, everything compounds.

So I built [Sommelier](https://github.com/AbdelStark/sommelier): an open, reproducible pipeline that post-trains a small open-weight model to emit schema-valid tool calls, and measures honestly whether the fine-tuned model beats the base model. This post is the story of actually running it, including the part where my own evaluation design lied to me and nearly convinced me that fine-tuning made the model worse.

## The ending, first

You should know whether the rest is worth your time. `Llama-3.1-Nemotron-Nano-8B-v1`, QLoRA-tuned on 15,000 function-calling examples for about three hours on one rented L40S, evaluated greedily on 1,000 held-out prompts with the same prompts and parser for both models, parse failures counted as failures:

| Metric | Base | Adapter | Δ |
|--------|------|---------|---|
| Valid JSON rate | 0.916 | **1.000** | +0.084 |
| Function name accuracy | 0.911 | **0.996** | +0.085 |
| Argument exact match | 0.707 | **0.876** | +0.169 |
| Argument F1 | 0.757 | **0.929** | +0.172 |
| Full-call exact match | 0.705 | **0.874** | +0.169 |

The GPU bill for the successful run was about eight dollars. My mistakes along the way (failed runs, plus a serving crash-loop I let spin for three hours) cost about as much again. The [adapter](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora) and the [exact train/val/test splits](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits) are on Hugging Face, and every number above traces back to a checksummed artifact you can download.

Before anyone gets excited, the two objections I would raise myself. You could get 100% *valid JSON* out of the base model with constrained decoding; grammar-guided generation exists and works. What constrained decoding cannot give you is the jump from 0.705 to 0.874 on full-call exact match. That delta is the model getting better at choosing the right function with the right arguments, not just at emitting parseable syntax. And exact-match scoring is harsh: an argument that is semantically right but formatted differently counts as wrong, for both models equally. The absolute numbers are conservative. The deltas are the point.

Those are the numbers. How they almost came out backwards is the better story, so let me tell it in order.

## The realization that started it

It began with a correction to something I believed. I used to think Nemotron was a family of open models. Good ones, but models.

I was wrong, and the correction matters. Nemotron is a system with three layers:

1. **Open models.** The Nemotron family, with open weights on Hugging Face, from Nano models that fit on one GPU to the larger tiers.
2. **Open software.** The NVIDIA NeMo suite covering the AI agent lifecycle, including [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator), a GPU-accelerated data curation toolkit built on RAPIDS and Ray.
3. **Open recipes.** This is the part that surprised me most. The actual data pipelines NVIDIA used to build Nemotron are public. The [Nemotron-CC curation recipe](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data/curation/nemotron-cc) reproduces the Nemotron-CC dataset end to end. Not a paper describing the pipeline. The pipeline.

Open weights tell you what a model is. Open recipes tell you how it came to be, and let you rerun the process yourself. That is a different level of openness, and it changes what you can build on top.

Sommelier is my attempt to build on top, and to test the promise on the task I care most about.

## Building for the question, not the demo

The design goal was never to beat frontier models. It was to answer the question a thousand teams are asking right now: can a small open model, post-trained cheaply on your own schemas, become reliable enough to carry the tool-calling load of your agentic product? And can you prove it with numbers instead of vibes?

The shape of the thing follows from that question. Sommelier is a staged CLI: data preparation, chat formatting, baseline evaluation, QLoRA training, adapter evaluation, comparison report. MIT licensed, documented end to end, and a fixture mode that runs the entire pipeline on a laptop with no GPU and no accounts, so CI never needs hardware.

The "prove it" part shaped everything else. A comparison between a base model and a fine-tune is only as good as its controls, so the pipeline enforces them structurally rather than by convention:

- **Prompts carry digests.** Every formatted example records a SHA-256 of its rendered prompt. Base and adapter evaluation consume the stored prompt text, so they cannot accidentally rebuild prompts differently.
- **The comparison is gated.** The final report refuses to exist unless both evaluations share the same config digest, test-split digest, prompt-set digest, parser version, and decoding settings. A mismatched comparison is not a warning. It is an error.
- **The parser never repairs.** It extracts the first balanced JSON span and classifies the result: `ok`, `no_json`, `invalid_json`, or `invalid_shape`. Malformed output is a failure, counted in every denominator. Valid structure is a metric, so a lenient parser would corrupt the experiment.
- **Loss is computed only on the answer.** Training masks every prompt token and requires the prompt/target token boundary to be provable: the pipeline tokenizes prompt and full text separately and verifies the prefix property holds. If a tokenizer merges tokens across the boundary, training refuses to run rather than silently training on the prompt.

None of this is glamorous. All of it is the difference between a measurement and a demo. And it paid for itself on the very first real run, though not by preventing a disaster. By making one legible.

## The bug that taught me the most

The first real smoke run on the 8B produced a result that looked like a catastrophe: the adapter was *worse* than the base model on every metric. Valid JSON dropped from 95% to 40%. Three hours into the project, the honest conclusion appeared to be that fine-tuning hurt.

I stared at that table for longer than I want to admit before doing the obvious thing and reading the raw generations. What I found was stranger than a training failure. The adapter had learned its training format perfectly. Canonical JSON, sorted keys, compact separators, byte for byte what the gold targets looked like. The model was fine. The problem was me.

The source dataset ([Salesforce's xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)) is 52.6% *multi-call* examples, where the gold answer contains two or more tool calls. My v1 contract of one call per request, enforced by the parser, collided with that data in a way that punished the two models asymmetrically:

- The **base model** emitted multiple loose JSON objects, one after another. The parser extracted the first balanced span, which matched the first gold call. Full credit.
- The **adapter** faithfully reproduced the complete gold answer as a multi-call array. The single-call parser rejected it as `invalid_shape`. Zero.

Read that again, because it is the whole lesson in two bullets. A model could emit the byte-exact gold target and score zero, while a model that ignored half the task scored full marks. The evaluation was not measuring what I thought it was measuring, and nothing in the aggregate numbers even hinted at it.

The fix was to make the data agree with the contract: preparation now drops multi-call rows as a declared filter, with its own counted drop reason, so training targets, parser, and metrics all describe the same task. After that, the comparison became meaningful, and the adapter won cleanly.

I would tattoo the general form of this on every eval harness: **your metric can be wrong in a direction you did not anticipate, and it will look exactly like a model failure.** The only defense is keeping raw generations and actually reading them.

## What only real runs catch

The multi-call collision was the deep surprise. The GPU had shallower lessons queued up too, and they are worth listing because every one of them was invisible to a green test suite. Three hundred tests, golden prompt fixtures, property tests on split leakage: all passing, and the first three remote runs still failed.

- **Double BOS.** The rendered chat template already contains the begin-of-text token; tokenizing it again with `add_special_tokens=True` silently duplicated it. Same corruption for both models, so no comparison test caught it. But the evaluated prompt was not the trained prompt.
- **The Trainer ate my columns.** Recent `transformers` wraps custom collators in a `RemoveColumnsCollator` that strips any feature not in the model's forward signature, including the `prompt_text` and `full_text` my completion-only collator needed. `remove_unused_columns=False` is load-bearing.
- **Library defaults drift under you.** The first full run ran out of memory on a 44 GiB L40S at batch 8, and the memory telemetry suggested that gradient checkpointing (which peft's k-bit preparation used to enable) was not actually on under the current transformers and peft combination. Explicit `gradient_checkpointing=True` plus half the micro-batch, at the same effective batch size, brought peak memory to 26 GiB and, counterintuitively, faster steps: 5.4 s/step versus 8.5.
- **Real data has a longer tail than you think.** Two prompts out of 16,000 exceeded my 2,048-token training budget. The pipeline now audits every rendered sequence right after formatting, so a budget violation costs thirty seconds instead of failing training forty minutes in, after the baseline evaluation has already spent its GPU time.
- **vLLM needs nvcc.** Serving the adapter through vLLM crash-looped for hours, with the engine core dying before it could log anything: vLLM JIT-compiles kernels at startup and slim container images do not ship the CUDA toolkit. The devel base image fixed it, and a foreground-diagnostics entrypoint now exists so that class of silent failure costs one focused run.

None of these are exotic. They are what the gap between "the tests pass" and "the system works" looks like in 2026's ML stack, where the libraries under you move monthly. What the pipeline contributed is that every failure was loud. Each raised an explicit error naming the config values that caused it, instead of leaving a silently wrong number in a table.

## The run that finally worked was boring

After all of that, the successful run was almost an anticlimax. For the record, because I wish more posts included this:

| | |
|---|---|
| Base model | nvidia/Llama-3.1-Nemotron-Nano-8B-v1, NF4 4-bit, bf16 compute |
| Adapter | LoRA r=16, α=32, dropout 0.05, on q/k/v/o/gate/up/down projections |
| Data | 60,000 raw rows → 26,735 after single-call filter + dedup → 15k/1k/1k splits |
| Schedule | 2 epochs, effective batch 16, lr 2e-4 cosine, 3% warmup, 1,876 steps |
| Loss | train 1.085 → 0.042; eval 0.038 → 0.030 |
| Hardware | 1× L40S, 26.3 GiB peak, 3.05 h training, ~15 min per 1,000-prompt eval |
| Stack | torch 2.12.1, transformers 5.12.1, peft 0.19.1, bitsandbytes 0.49.2 |

QLoRA is the reason this table is boring, and boring is the achievement. The base model stays frozen in 4-bit; the trainable adapter is 168 MB. An 8-billion-parameter model post-trains on one rented GPU in an afternoon for single-digit dollars. Post-training used to be a lab capability. It is now a weekend capability, and that shifts the whole economics of building with open models.

Serving follows the same arc. The trained adapter runs on my MacBook: 18 GB of unified memory, about 33 tokens per second, which is the machine's memory bandwidth ceiling. Since the model learned to emit only the call and stop, responses land in about two seconds. For real throughput, vLLM loads the base once and registers the LoRA as a named model, so one endpoint serves both variants and an A/B is a one-word change in the request. One distribution caveat worth knowing: the adapter expects tool schemas in the source dataset's flat-parameter style. Hand it a JSON-Schema-style tool and it degrades visibly, as parse failures, because the parser refuses to pretend otherwise.

## The strategy lesson, and why I admire it

Here is the thing that kept striking me while building: NVIDIA's open strategy is good for the entire industry, and it is good business for NVIDIA, and they are completely open about both halves.

Making open models easy to adopt, publishing the curation recipes, open-sourcing the tooling that makes post-training accessible: all of it raises the standard of the whole ecosystem. More teams can build serious models. More builders can own their stack. And yes, every team that starts post-training and serving open models consumes more accelerated computing. There is no hidden agenda because none is needed; the interests genuinely align, and NVIDIA says so out loud.

I find that intellectually honest, and rare. The common pattern in our industry is openness as marketing, capability held back, strategy dressed up as altruism. The Nemotron approach is the opposite: here are the weights, here is the software, here is the literal recipe, go build, and the rising tide is the business model.

## The sovereignty angle

I care deeply about European AI sovereignty, and this experiment sharpened my view of what sovereignty means in practice. It is not a slogan. It is the concrete ability to take open weights, apply open recipes, run accessible tooling, and produce a model you control, tuned to your task, on your terms.

After this experiment I can make that concrete: sovereignty over a tool-calling model, for one task, costs about eight dollars of compute, three hours of GPU time, and a pipeline you can read in an afternoon. The weights are yours. The data transformations are declared and counted. The evaluation is reproducible from digests, and the licenses (the base model's NVIDIA Open Model License and Llama 3.1 terms, the dataset's CC-BY-4.0) are recorded and machine-checked before anything ships. Nobody can reprice it or deprecate it out from under you. Open weights plus open recipes plus accessible post-training is what practical sovereignty looks like for a European builder in 2026. The stack for it is here, and most people have not noticed how much of it is sitting on GitHub.

## What is not claimed, and what is next

This is one run, one seed, one dataset, English only, single-call by design. The report the pipeline generates says exactly that, because a claim without its boundaries is marketing. I have not measured multi-call planning, multi-turn tool use, robustness to out-of-distribution schemas beyond observing that it degrades, or variance across seeds. Next on the list: a French evaluation slice, because tool calling should work as well in French as in English, and that is a claim worth measuring rather than assuming.

If you are exploring post-training for your own product, steal anything you like from Sommelier. The digest-gated comparison and the boundary-proving collator are the parts I would take first. And if you have opinions on the evaluation design, the parser is deliberately conservative and I would genuinely enjoy the argument.

The model, in the end, learned the sommelier's job in three GPU hours: one long list, one right choice, again and again. Teaching myself to measure that honestly took the rest of the time, and it was the part worth writing about.

---

**Artifacts**

- Code, spec, and reproduction guide: https://github.com/AbdelStark/sommelier
- Adapter + evaluation evidence: https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora
- Training data (exact splits): https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits
- NeMo Curator: https://github.com/NVIDIA-NeMo/Curator
- The Nemotron-CC open recipe: https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data/curation/nemotron-cc

*Sommelier is MIT licensed. Model and dataset obligations, including the Built with Llama notice, are recorded in the repository and enforced by a release preflight.*
