# Data policy

Every row of the source dataset is untrusted input. The `data prepare` stage either turns a raw row into a validated, typed example or drops it with a declared reason, and the drop counts are written into the run's artifacts. Nothing is silently discarded, and nothing downstream ever re-parses raw strings: after this stage, the rest of the [pipeline](pipeline.md) works only with data that has already proven its shape.

## Raw rows are untrusted JSON strings

A raw row (`sommelier.raw_tool_call_row.v1`) carries `query`, `tools`, and `answers`, where `tools` and `answers` are raw JSON strings exactly as they appear in [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k). The strings could be malformed, the wrong shape, or empty, so validation parses them once, checks them against typed schemas, and produces a `PreparedExample` whose `tools` is a list of tool schemas and whose `gold_calls` is a list of typed calls. The design rule: if raw strings flowed downstream, every later stage would have to re-validate untrusted JSON, and each would do it slightly differently.

## Sixteen drop reasons

Validation checks run in a fixed order and a row gets exactly the first reason that fails. The set of reasons is a closed type (`DropReason` in [sommelier/data/types.py](https://github.com/AbdelStark/sommelier/blob/main/sommelier/data/types.py)), and the drop counter is initialized from that type rather than a hand-maintained list, so adding a reason can never desync the counts.

| Reason | Dropped when |
|--------|--------------|
| `missing_query` | `query` is absent, not a string, or only whitespace |
| `missing_tools` | `tools` is absent, not a string, or only whitespace |
| `missing_answers` | `answers` is absent, not a string, or only whitespace |
| `query_too_short` | the stripped query is shorter than `data.min_query_chars` (default 10) |
| `query_too_long` | the stripped query is longer than `data.max_query_chars` (default 2000) |
| `invalid_tools_json` | `tools` does not parse as JSON |
| `invalid_tool_shape` | the tools JSON is not a list of objects, each with a non-empty string `name`, a string `description`, and an object `parameters` |
| `invalid_answers_json` | `answers` does not parse as JSON |
| `invalid_answer_shape` | the answers JSON is not a non-empty list of objects, each with a non-empty string `name` and an object `arguments` |
| `multi_call_answer` | the answers parse cleanly but contain more than one call |
| `duplicate_query` | the normalized query hash was already seen in an earlier row of the same language |
| `missing_source_example` | a paired row names a root example that was dropped, never existed, or was not selected into any split |
| `duplicate_source_example` | a second paired row names a root example that already has a pair (first one wins) |
| `pair_tools_mismatch` | a paired row's `tools` string is not byte identical to its root row's |
| `pair_answers_mismatch` | a paired row's `answers` string is not byte identical to its root row's |
| `cross_split_duplicate` | a paired row's normalized query equals a root query that sits in a different split |

The first nine are quality checks and apply to every source. `multi_call_answer` and `duplicate_query` are policy. The last five exist only for paired sources and enforce the pairing contract described below.

## The multi-call drop is a scope decision

Sommelier v1 trains and scores exactly one tool call per request. Roughly half of the xlam rows answer with two or more calls, and all of them are dropped with reason `multi_call_answer`. These rows are not bad data. They are out of scope: the [conservative parser](evaluation.md) rejects multi-call outputs, so keeping multi-call golds in the training or test data would score a model as a failure for answering faithfully. Rather than bend the parser or the metrics around them, v1 declares the boundary, filters at the earliest stage, and records the filter with its own reason so the size of the exclusion is part of the run's evidence, not a footnote.

## Dedupe before split, and why

The dedupe key is `sha256(normalize_query(query))` where normalization casefolds, strips, and collapses internal whitespace. The first row with a given key is kept; later ones are dropped as `duplicate_query`.

Deduplication runs before splitting, and the ordering is the leakage defense. If you split first and dedupe within each split, two copies of the same query can land in train and test, and the held-out metrics quietly measure memorization. Deduplicating the whole pool first makes a cross-split duplicate impossible by construction: each `query_sha256` exists once in the pool, so it can land in at most one split. The invariant (a `query_sha256` appears in exactly one split) is still re-checked after splitting by intersecting the hash sets pairwise, and any overlap fails the stage. Belt and suspenders, but the belt is the mechanism.

The limit is stated plainly: this is exact dedupe on normalized text. Two paraphrases of the same request hash differently and can still straddle train and test. Semantic deduplication was considered and deferred, because it would add an embedding model dependency to a stage whose whole value is being trivially reproducible.

## Paired sources inherit their splits

A config can declare more than one dataset source under `datasets`, one per language. Exactly one source is the root: it goes through validation, dedupe, and the seeded split exactly as described above, and nothing about that path changes when paired sources exist (the same input and seed produce the same root splits with or without them). Every other source is paired: each of its rows names a root example through `source_example_id`, and the row inherits that example's split. Paired rows are never shuffled or assigned independently, so a translated variant of a query cannot land on the other side of a split boundary from its original.

The pairing contract is enforced at prepare time, not trusted from the dataset producer:

- A paired row's `tools` and `answers` strings must be byte identical to its root row's. The gold answer is the supervision target and the scoring key; if translation changed it, the languages would no longer measure the same task. Violations drop with `pair_tools_mismatch` or `pair_answers_mismatch`.
- A root example gets at most one pair per language (`duplicate_source_example`), and a pair whose root was dropped or not selected drops with `missing_source_example`.
- A paired query that coincidentally equals a root query from another split drops with `cross_split_duplicate`; identical to the dedupe ordering argument above, this closes the last path for the same text to appear on both sides of a boundary.

After preparation, split safety is re-asserted across all languages at once: per language disjointness, globally unique example ids, every paired example in the same split as its root, and no query digest in two different splits anywhere. Any violation fails the stage.

Split files mix languages: root rows first, then each paired language in configuration order, each following the root split's example order. A paired source is allowed to arrive short (translation drops rows); the per language `split_sizes` in the drop summary make the shortfall visible instead of silent.

## Seeded shuffle, fixed-count splits

The deduplicated pool is shuffled with `random.Random(seed)` (the seed comes from `project.seed`, default 42) and sliced into fixed counts in train, validation, test order: `data.n_train`, `data.n_validation`, `data.n_test`, defaults 15,000 / 1,000 / 1,000. Fixed counts, not percentages, so that runtime and cost are predictable across dataset revisions and two runs of the reference config are the same size by definition.

If the pool is smaller than the requested total, preparation fails with a `UserInputError` (exit code 2) before writing any split file:

```text
sommelier: SOM002: insufficient valid rows: need 17000, got 12480
hint: Lower split counts or provide more valid deduplicated rows.
```

Failing beats writing a smaller split silently, because every downstream digest and metric denominator assumes the configured sizes. Exit codes are cataloged in the [error reference](../reference/errors.md).

One operational note: `sommelier data prepare --gpu` runs a cuDF coarse filter (null and length bounds) before the Python validation. Rows removed by the coarse filter never reach the drop counter, so use the default CPU path when you want a complete drop summary. `sommelier pipeline run` always uses the CPU path.

## What the reference run dropped

The drop summary (`data/drop_summary.json`, schema `sommelier.drop_summary.v2`) records, per language, the count for every reason, the pool sizes, and the final split sizes. For the [reference run](../results/reference-run.md) on the recorded dataset revision (a single-source run, so one language section):

| Stage | Rows |
|-------|------|
| Source rows read | 60,000 |
| Dropped `multi_call_answer` | 31,539 |
| Dropped for any other validation reason | 0 |
| Valid single-call rows | 28,461 |
| Dropped `duplicate_query` | 1,726 |
| Deduplicated pool | 26,735 |
| Used in splits (15,000 + 1,000 + 1,000) | 17,000 |

Two things are worth reading off this table. First, the multi-call filter is by far the largest cut: 52.6 percent of the source rows are excluded by scope, not quality, which is exactly why the drop summary exists. Second, every row that claimed to be a single-call example parsed cleanly on this revision; the nine quality reasons all count zero. The 9,735 unique rows beyond the requested 17,000 are simply unused.

The exact prepared splits from the reference run are published as [abdelstark/sommelier-xlam-single-call-splits](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits) (CC-BY-4.0), so you can [reproduce](../guides/reproduction.md) or audit the run without re-running preparation, and verify that the split digests recorded in the manifests match what you download.

## What preparation writes

```text
data/
├── train.jsonl          15,000 prepared examples
├── validation.jsonl      1,000 prepared examples
├── test.jsonl            1,000 prepared examples
└── drop_summary.json    counts per reason, pool sizes, requested sizes
```

The stage manifest, `data_manifest.json`, lands at the run root (`runs/<run_id>/data_manifest.json`) with checksums of everything above.

Each JSONL row is a `sommelier.prepared_example.v2` record carrying its `language`, its `query_sha256`, the source revision it came from, and, on paired rows, the `source_example_id` of its root example. Schemas and checksum rules are in the [artifact reference](../reference/artifacts.md); the config fields that control this stage are in the [configuration reference](../reference/configuration.md).
