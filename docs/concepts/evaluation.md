# Evaluation method

Evaluation answers one question: does the adapter produce better single JSON tool calls than the base model, under conditions identical enough that the difference is attributable to training? Every design choice here leans conservative, because an evaluation that flatters the model is worse than no evaluation. The base and adapter runs are the same code, the same stored prompts, the same parser, and the same decoding; the [comparison gate](determinism.md) refuses to compare anything else.

## Deterministic decoding, enforced

`sommelier eval run` generates exactly one completion per formatted test prompt. Prompts come only from the stored `prompt_text` of the formatted split; evaluation never rebuilds them, and every generation must carry the `prompt_sha256` the formatter recorded, or the report stage fails with an invariant violation.

Decoding requirements are validated, never coerced. A config with `eval.temperature` other than exactly 0.0, `eval.do_sample: true`, or a non-positive `eval.max_new_tokens` (default 512) fails with an `EvaluationError` that names the field to fix. Silently forcing temperature to 0.0 would mean the config on disk no longer describes what ran. Greedy decoding removes sampling variance, so a metric difference between base and adapter cannot be decoding noise. Each generation record embeds the decoding config it ran with, and report building fails if the records mix configs.

## The conservative parser

The parser (`sommelier.parser.v1`, [sommelier/evaluation/parse.py](https://github.com/AbdelStark/sommelier/blob/main/sommelier/evaluation/parse.py)) extracts the first balanced JSON object or array span from the raw generation, tracking bracket depth while skipping double-quoted strings and escape sequences. Surrounding prose is ignored; nothing inside the span is repaired. If the first candidate span never balances, the parser does not hunt for a second one.

The accepted shape is exact: an object whose keys are exactly `name` and `arguments`, with `name` a non-empty string and `arguments` a JSON object, or a single-element array containing exactly one such object. Extra keys, empty names, non-object arguments, empty arrays, and multi-call arrays all fail. Multi-call outputs failing is by design: v1 scores exactly one call per request, and the [data policy](data.md) drops multi-call golds for the same reason.

Every generation gets one of four statuses:

| Status | Meaning |
|--------|---------|
| `ok` | A schema-valid call was extracted |
| `no_json` | The text contains no opening `{` or `[` |
| `invalid_json` | An opening bracket exists but no balanced span can be extracted, or the span is not valid JSON |
| `invalid_shape` | The JSON parses but is not exactly the required call shape |

Why so strict? A lenient parser that repairs trailing commas, strips markdown fences, or picks the most plausible of several objects is answering a different question: not "can this model emit a valid tool call" but "can this model plus my repair heuristics emit one". Repair hides exactly the failure mode this project measures; producing valid structured output is itself a primary metric, and the reference run shows it carries signal (base valid JSON rate 0.916, adapter 1.000). Strictness also cannot favor either side, since base and adapter face the identical parser. Model-judged scoring was rejected for the same reason: it adds a second model to the trust chain of a pipeline whose point is that anyone can re-derive the numbers.

## Parse failures count against everything

Invariant INV-DATA-005: a parse failure is a wrong answer, not missing data. The four rate metrics use the full test count (1,000 in the reference config) as their denominator, so a generation with status `no_json` scores zero on all of them; failures are never excluded from the sample. Argument F1 applies the same principle in pooled form: a failed record contributes its gold argument pairs to the denominator and zero predicted pairs. Reports would look better if failures were dropped from the denominator, which is precisely why they are not.

## The five metrics

Each metric is stored as a value with its numerator and denominator, so no number has to be taken on faith. Exact definitions, including the argument flattening rules, are in the [metric reference](../reference/metrics.md).

| Metric | Question it answers |
|--------|---------------------|
| `valid_json_rate` | Did the output parse into a schema-valid tool call at all? |
| `function_name_accuracy` | Did the parsed call name the gold function? |
| `argument_exact_match` | Are the arguments exactly the gold arguments (canonical JSON equality)? |
| `argument_f1` | How much of the arguments was right? Micro-F1 over flattened key paths and values |
| `full_call_exact_match` | Name and arguments both exactly right, the strictest bar |

The metrics are ordered diagnostics, not redundancy: a model can emit valid JSON but name the wrong tool, or name the right tool and fumble one argument. The [reference run](../results/reference-run.md) reports all five for both models.

## Slices and the language gap

Evaluation runs once per configured `eval.slices` language: the formatted test split is partitioned by each example's `language`, every slice is evaluated with the same model, prompt policy, parser, and decoding, and a configured slice with no rows is an error rather than an empty section. The evaluation report carries one metrics block per slice plus the overall block across all slices, each with its own prompt-set digest.

The comparison report adds the measurement multilingual runs exist for: for every non-reference slice (the reference is the first configured slice, English in the v2 setup), it records the per-metric gap against the reference, once for the base model and once for the adapter. The base gap answers how much the model loses on French input before any training; the adapter gap answers how much of that loss the training closed. Because paired slices share byte-identical gold answers and tool schemas by construction, the gap isolates the query language as the only moving variable.

## Everything is re-scorable

Raw generations are always retained. `generations.<slice>.jsonl` holds one record per test prompt of that slice with the raw generated text, the language, the parsed call (or null), the parse status, the prompt digest, and the decoding config, including every failure. If you distrust the parser, you can re-run it, or your own, over `raw_text` and re-derive every metric from the file. The report stage adds its own consistency checks before scoring: the generation count must equal the slice size, every generation must reference a known example and carry its slice's language, and every prompt digest must match the formatted split.

`evaluation_report.json` records the metrics plus the identity of the conditions they were measured under: config digest, test split digest, per-slice ordered prompt-set digests, parser version, and decoding config. Those fields, plus the slice set itself, are what the comparison gate checks before it will put base and adapter numbers in the same table; the mechanism is described in [Determinism and the comparison gate](determinism.md). Adapter reports also record where the weights came from, which is how a published adapter is evaluated against a base model without retraining.

## Known sharp edges

Conservative scoring has costs, and they are stated rather than smoothed over. Exact comparison penalizes semantically equivalent forms: the string `"2"` does not match the number `2`, and two spellings of the same date do not match. Lists are compared by index, which is harsh on arguments where order does not matter. Both edges apply identically to base and adapter, so the comparison stays fair even where the absolute numbers are pessimistic. If your task needs semantic argument equivalence, that is a scoring extension, not a config flag; see [Design decisions](design-decisions.md) for the boundary.
