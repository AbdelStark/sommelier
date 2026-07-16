---
license: cc-by-4.0
language:
  - he
source_datasets:
  - Salesforce/xlam-function-calling-60k
pretty_name: Sommelier xLAM single-call Hebrew paired rows
---

# Sommelier xLAM single-call Hebrew paired rows

> **Publication template — full Hebrew v3 evidence is pending.** Replace this
> block with a release-specific evidence statement and remove the literal
> `REPLACE_FROM_VERIFIED_DATASET_BUNDLE` marker only after the audited rows,
> semantic gate, and publication manifest exist. The dataset publisher rejects
> an unresolved marker.

This CC-BY-4.0 dataset is derived from
`Salesforce/xlam-function-calling-60k`. Sommelier filters and selects the root
corpus, machine-translates only each natural-language query into Hebrew, and
keeps tool schemas and gold answers byte-identical. Every accepted row names
its English root with `source_example_id` and inherits that root's split.

The one-time forward teacher is the OpenAI Responses snapshot
`gpt-5.5-2026-04-23` using explicit Flex service. The public
`translation_summary.json` records the exact request/runtime identity, returned
model and tier, usage, mechanical audit/drop counts, the maximum observed
response input-token count, and a calculated public-list-price value. Full
publication binds that value to the recorded local ceiling. The ceiling is not
an invoice or provider account/project cap. The dated API name is not a
provider-weight digest or a guarantee of byte-identical regeneration.
`store=false` is not a Zero Data Retention claim. The source query and bounded
selected-tool projection were processed by OpenAI.

The release is a machine-translated survivor corpus: rows that failed the
declared structural, placeholder, Hebrew-script, bidi, or protected-span audits
are absent and counted. Before publication, a deterministic 200-row sample is
back-translated with the pinned
`Helsinki-NLP/opus-mt-tc-big-he-en` checkpoint and reviewed under Sommelier's
non-native semantic rubric. The immutable template, finalized decisions, and
whole-publication gate are included. This bounded audit does not establish
native fluency or full-corpus semantic correctness.

## Files

- `rows.he.jsonl`: accepted Hebrew paired rows.
- `translation_summary.json`: complete translation identity and accounting.
- `translation_publication.json`: canonical row/provenance digest binding.
- `translation_semantic_review_template.json`: untouched machine-selected
  review inputs and back-translations.
- `translation_semantic_review.json`: finalized review and release decision.

The private raw provider journal is intentionally not distributed. Its SHA-256
and content-free aggregate are bound by the public summary.

## Attribution and intended use

Attribute Salesforce for the source xLAM dataset and indicate that Sommelier
filtered, split, and machine-translated the queries. Review the upstream dataset
card and CC-BY-4.0 terms before redistribution. This release supports the
recorded single-tool-call research experiment; it is not a general Hebrew
assistant corpus, a native translation benchmark, or production safety data.
