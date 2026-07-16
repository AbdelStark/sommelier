---
license: cc-by-4.0
language:
  - he
source_datasets:
  - Salesforce/xlam-function-calling-60k
pretty_name: Sommelier xLAM single-call Hebrew paired rows (Hy-MT2, sanitized)
task_categories:
  - text-generation
tags:
  - tool-calling
  - function-calling
  - hebrew
  - machine-translation
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: pairs
        path: rows.he.jsonl
---

# Sommelier xLAM single-call Hebrew paired rows (Hy-MT2, sanitized release)

This CC-BY-4.0 dataset is derived from
[`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).
Sommelier filters the source corpus to single-tool-call examples, deterministically
splits it, and machine-translates only each natural-language `query` into Hebrew.
The exact training snapshot kept tool schemas and gold answers byte-identical to
the English root. For public release, 15 GitHub-PAT-shaped substrings inherited
from the synthetic upstream corpus were replaced with `[redacted]` in the
`tools` field; queries and answers were unchanged. Every accepted row names its
English root with `source_example_id`; downstream preparation deterministically
assigns it to that root's train/validation/test split. The published JSONL
itself is exposed as one Hugging Face `pairs` split.

## How the translation was produced

The forward translator is
[`tencent/Hy-MT2-1.8B-GGUF`](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)
(Hunyuan-MT), run locally through ollama from the official `Q8_0` GGUF build with
greedy decoding (temperature 0, fixed seed). The exact local 1,908,528,192-byte
blob has SHA-256
`5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4`.
It matches `Hy-MT2-1.8B-Q8_0.gguf` at immutable repository revision
`b27182d810fa3ceb6ed04e7c324c54e35c0d209c` byte for byte.
This run did not directly load the separate `Hy-MT2-1.8B-FP8` artifact.
Because Hy-MT2 is a pure translator
that would otherwise translate argument values, each gold-bearing protected span
in the query is masked with a short ASCII sentinel before translation and
restored afterward; rows are retried with alternate sentinels and a final
unmasked pass.

Every accepted row passes Sommelier's mechanical translation audit:

- every protected span present byte-identically at token boundaries,
- Hebrew as the dominant script (at least half of unprotected letters),
- no unsafe bidirectional control characters, no Unicode replacement or
  control/format/surrogate code points,
- no reproduction of the translation instruction,
- length within the preparation budget.

`translation_summary.json` records the model, decoding settings, per-attempt
acceptance counts, per-split counts, and the mechanical drop-reason histogram.

## Accounting and immutable identity

- Selected roots: 17,000
- Accepted rows: 16,272 (95.72%)
- Accepted by root split: 14,364 train / 961 validation / 947 test
- Drops: 626 missing protected span / 48 malformed spacing / 54 wrong script
- Exact dataset revision used by training (now privately quarantined):
  `c9598f1855ca88e11a4aa31c53ad387faf467eb2`
- Exact training `rows.he.jsonl` SHA-256:
  `94f1c23022f24065edad136cf4768a21576db4d89a3e45774f7806f69adbcabf`
- Public sanitized `rows.he.jsonl` SHA-256:
  `d7f2895f2b3976cc7ea917dc85633b0140efe03737bda8e20d69ba0c449cacee`
- `translation_summary.json` SHA-256:
  `0ab7b62615dce11dc705faccf047df105f782665a0e3bbaf157dc46c995bc5ce`

The downstream preparation gate removed 86 duplicate translated queries. The
training run therefore consumed 14,286 / 955 / 945 Hebrew examples in its
train/validation/test splits.

## Public security sanitization

The upstream APIGen corpus is gated, research-oriented synthetic data, but 15
rows contained strings matching the classic `ghp_` GitHub personal-access-token
shape. Sommelier did not test those strings: synthetic-looking credentials are
not assumed harmless. The exact training snapshot was therefore quarantined,
and the public JSONL replaces all 15 matches with `[redacted]`. No row was
removed; only the `tools` field changed in those 15 rows. The deterministic
`public_sanitization.json` binds the original revision/hash, pattern classes,
replacement counts, and public output hash. Consequently, the public JSONL is
not byte-identical to the dataset used by training.

## Honesty and limitations

This is a machine-translated **survivor corpus**. Rows that failed the mechanical
audits are absent and counted. The audits are structural: they cannot prove that
a translated query preserves the intent of the English root.

**There is no human semantic review.** This dataset is deliberately distinct from
the preregistered Sommelier Hebrew v3 evidence dataset, which uses a different
forward teacher and a human-signed back-translation review. This release supports
a local, open-model tool-calling experiment; it is not a native Hebrew benchmark,
a general Hebrew assistant corpus, or production safety data.

## Files

- `rows.he.jsonl`: accepted Hebrew paired rows
  (`sommelier.raw_tool_call_row.v1`: `query`, `tools`, `answers`,
  `source_example_id`, `source_id`, `source_revision`, `schema_version`).
- `translation_summary.json`: translation identity and mechanical accounting.
- `translator_identity.json`: a post-run immutable binding from the local
  Ollama blob to the official Tencent GGUF file. It supplements rather than
  rewrites the translation summary used by training.
- `public_sanitization.json`: the closed accounting and input/output hashes for
  the credential-pattern sanitization applied before public release.

## Attribution and intended use

Attribute Salesforce for the source xLAM dataset and indicate that Sommelier
filtered, split, and machine-translated the queries with Tencent Hy-MT2. Review
the gated upstream dataset card, its research/synthetic-data limitations, and
CC-BY-4.0 terms before redistribution. Cite Zuxin Liu et al., “APIGen:
Automated Pipeline for Generating Verifiable and Diverse Function-Calling
Datasets,” arXiv:2406.18518 (2024).
