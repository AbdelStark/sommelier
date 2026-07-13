# Sommelier: the paper

A technical report and position paper on the Sommelier experiment: the data flywheel thesis, the sovereign AI argument, the pipeline, and the results of the two published runs (`nemotron-8b-full-3` and `nemotron-8b-fr-full-4`).

## Build

```bash
make            # compiles main.pdf with tectonic (runs the bibliography automatically)
make figures    # regenerates figures/*.pdf from the checked-in scripts
```

Requires [tectonic](https://tectonic-typesetting.github.io) (`brew install tectonic`) and, for figures, `uv`.

## Layout

- `main.tex` is the paper. It uses the NeurIPS style file in preprint mode; this is a formatting choice, not a submission.
- `references.bib` holds only verified entries: every arXiv id and DOI was fetched programmatically before inclusion.
- `figures/gen_fig_*.py` regenerate the data figures from numbers stated in `docs/results/`. The two diagrams (flywheel, pipeline) are TikZ inside `main.tex`.
- `neurips.sty` is the conference style file; do not edit it.

## Fact discipline

Every number in the paper traces to a repo artifact, chiefly `docs/results/reference-run.md`, `docs/results/french-run.md`, `docs/concepts/`, `examples/config.full.yaml`, and the published Hugging Face repos. If you change a number here, it must change there first.
