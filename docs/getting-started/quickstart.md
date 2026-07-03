# Quickstart

Run the first two pipeline stages on your laptop, then read what they wrote. Fixture mode needs no GPU, no accounts, and no downloads; the point is to see the artifact discipline up close before renting hardware. [Install](installation.md) first. Every output quoted below comes from a real run of these commands.

## Why fixture mode exists

`--fixture` makes a stage run on synthetic rows instead of the real dataset. The quickstart uses it for two reasons. First, `data prepare` refuses to split unless it has at least `n_train + n_validation + n_test` valid rows, and the repo bundles only a 30-row raw fixture while the smoke config asks for 100/20/20:

```text
$ uv run sommelier data prepare --config examples/config.smoke.yaml \
    --out examples/artifacts/runs/local/data --run-id local
sommelier: SOM002: insufficient valid rows: need 140, got 30
hint: Lower split counts or provide more valid deduplicated rows.
```

Second, rendering real prompts requires the tokenizer, which requires transformers, which the base install deliberately does not have. Fixture mode synthesizes exactly the requested number of valid examples and skips the tokenizer, so the artifact plumbing runs end to end on a bare laptop. Real splits of the real dataset are prepared on the GPU host against the configured dataset revision, which is recorded in every prepared example; see [Data policy](../concepts/data.md).

## Run it

From the repo root:

```bash
uv run sommelier config validate --config examples/config.smoke.yaml
uv run sommelier data prepare --config examples/config.smoke.yaml --fixture \
  --out examples/artifacts/runs/local/data --run-id local
uv run sommelier format build --config examples/config.smoke.yaml \
  --data examples/artifacts/runs/local/data \
  --out examples/artifacts/runs/local/formatted --run-id local --fixture
```

```text
config ok: examples/config.smoke.yaml
data prepare ok: run_id=local out=examples/artifacts/runs/local/data
format build ok: run_id=local out=examples/artifacts/runs/local/formatted
```

Two things about the flags. `--out` must sit inside the config's `artifact_root` (`examples/artifacts`, resolved relative to the config file); anything else fails with `SOM202: artifact path escapes artifact root`, because artifact confinement is checked on every write. And `--run-id local` names the run directory; later stages can also infer the run ID from any path containing `/runs/<id>/`.

## What was written

```text
examples/artifacts/runs/local/
├── config.resolved.yaml        the exact config the run used, defaults filled in
├── manifest.json               run manifest: stage → stage manifest path
├── data_manifest.json          the data stage's record of itself
├── format_manifest.json        the format stage's record of itself
├── data/
│   ├── train.jsonl             100 prepared examples
│   ├── validation.jsonl         20
│   └── test.jsonl               20
└── formatted/
    ├── train.jsonl             100 formatted examples
    ├── validation.jsonl         20
    └── test.jsonl               20
```

## Read a manifest

`data_manifest.json`, trimmed to one output entry:

```json
{
  "schema_version": "sommelier.manifest.v1",
  "stage": "data",
  "run_id": "local",
  "created_at": "2026-07-03T15:53:36.333569+00:00",
  "git_commit": "d993f6668898c4c143ba14bbff0914a6d5cde254",
  "config_sha256": "df946dd6f8e08edbb66fcdc085dc9084c6e80e60f3c9281f52d76242ea7ea907",
  "dependency_lock_sha256": "efe5f944a509077ebfe36e2b9ceec1381abe9cfc31046f5e4880b3cf1e207cd9",
  "command": ["sommelier", "data", "prepare", "--config", "examples/config.smoke.yaml", "..."],
  "seed": 42,
  "inputs": [ { "path": "runs/local/config.resolved.yaml", "...": "..." } ],
  "outputs": [
    {
      "path": "runs/local/data/train.jsonl",
      "kind": "dataset_split",
      "schema_version": "sommelier.prepared_example.v1",
      "sha256": "8d2ce23f901a94275b5517a5fa5f88d04a6b9f03dbbbb3ef006fbf851ffb4151",
      "bytes": 54676
    }
  ],
  "status": "succeeded"
}
```

This is the unit of evidence the whole project is built from: which command ran, at which commit, under which config digest and dependency lockfile, and the SHA-256 of every input and output. Open `format_manifest.json` and you will find the three split files listed as its inputs with the same digests, which is how the chain stays checkable file by file. Your `created_at` and `git_commit` will differ; the split digests should not, because the records carry no timestamps and are written with sorted keys, so rerunning a stage under the same config produces byte-identical files. [Determinism](../concepts/determinism.md) explains why the pipeline insists on this.

## Read the records

One prepared example from `data/train.jsonl` (fields in on-disk order):

```json
{
  "example_id": "train-1",
  "gold_calls": [{"arguments": {"city": "Paris"}, "name": "lookup_weather"}],
  "query": "Fixture train request 1: what is the weather in Paris?",
  "query_sha256": "096420be9654fd3bd766614905fd2e811cba984b2b195df1a49c8571ce1f7c28",
  "schema_version": "sommelier.prepared_example.v1",
  "source_id": "fixture:train-1",
  "source_revision": "main",
  "split": "train",
  "tools": [{"description": "Look up weather for a city.", "name": "lookup_weather",
             "parameters": {"properties": {"city": {"type": "string"}}, "type": "object"}}]
}
```

Exactly one gold call per example: that is the v1 contract, enforced at preparation time. `query_sha256` is the query's identity for deduplication and split disjointness, so the same query can never appear in two splits.

One formatted example from `formatted/test.jsonl`, long fields elided:

```json
{
  "example_id": "test-1",
  "full_text": "[{\"content\":\"You are a tool-calling model. ...",
  "messages": [ "...system, user, and assistant messages..." ],
  "prompt_sha256": "a6888fd273c33cebeb5e1152c6e9fc2e2b308cec3fc31051986ea989fdbe7704",
  "prompt_text": "[{\"content\":\"You are a tool-calling model. ...",
  "schema_version": "sommelier.formatted_example.v1",
  "split": "test",
  "target_text": "[{\"arguments\":{\"city\":\"Paris\"},\"name\":\"lookup_weather\"}]",
  "template_policy": "tokenizer_chat_template",
  "tokenizer_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
  "tokenizer_revision": "main"
}
```

`target_text` is the canonical JSON of the gold call (sorted keys, compact separators) and nothing else: no prose, no code fences. `prompt_sha256` is the digest that later lets both evaluations prove they scored the same prompts, and lets the [comparison gate](../concepts/determinism.md) refuse a base-versus-adapter report when they did not. In fixture mode `prompt_text` is the canonical JSON of the system and user messages, because no tokenizer is loaded; the real stage renders through the tokenizer's chat template and fails if the result does not split provably into prompt followed by target (see [the pipeline](../concepts/pipeline.md), stage 2).

## What fixture mode does not show

Real preparation also writes `data/drop_summary.json`, counting how many raw rows fell to each of the eleven declared drop reasons (multi-call answers, invalid JSON, duplicate queries, and so on). Fixture rows are synthesized already valid, so there is nothing to drop and no drop summary here. [Data policy](../concepts/data.md) covers the filter and why its record matters.

## Stages that need the GPU stack

The base install cannot run generation or training, by design. Those stages import their stacks lazily and fail fast, before creating their stage directory:

```text
$ uv run sommelier eval run --config examples/config.smoke.yaml --model base \
    --data examples/artifacts/runs/local/formatted \
    --out examples/artifacts/runs/local/eval/base --run-id local
sommelier: SOM003: model generation requires the torch and transformers packages
hint: Run evaluation remotely or install the eval extra stack.
$ echo $?
3
```

Exit code 3 is `ExternalDependencyError`: the environment lacks something, your input was fine. The full mapping is in [Errors and exit codes](../reference/errors.md).

| Command | On the base install |
|---|---|
| `config validate` · `data prepare` · `data validate-fixtures` · `format build --fixture` · `report compare` | Runs |
| `format build` (real tokenizer) | Fails with SOM003: needs transformers |
| `eval run` | Fails with SOM003: needs torch and transformers (peft for `--model adapter`) |
| `train run` | Fails with SOM003: needs torch, transformers, and peft |

These stacks live in the remote Modal images, so the normal way to run the missing stages is remotely.

## Where to go next

- [Remote execution](../guides/remote-execution.md): run the real six-stage chain on a Modal GPU in smoke mode, capped at 100/20/20 examples. This is where accounts and money enter.
- [Reproduction](../guides/reproduction.md): the full reference run, from clean checkout to the comparison report, with costs stated up front.
- [The pipeline](../concepts/pipeline.md) and [Artifacts](../concepts/artifacts.md): the map of what you just ran two stages of.
- [CLI reference](../reference/cli.md): every command and flag.
