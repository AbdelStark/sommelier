# Sommelier is here. Sovereign AI is needed.

A one-minute promotional video for [Sommelier](https://github.com/AbdelStark/sommelier),
built with [Remotion](https://remotion.dev). 1920x1080, 30 fps, ~1,756 frames
(~58.5 s), paced so every card, row, and number can actually be read.

## Narrative arc

| Scene | Beat | Content |
|-------|------|---------|
| 1 | The problem | Four fact-checked headline cards on gated frontier AI, then the hook: "Should we accept gatekeeping of intelligence?" |
| 2 | The turn | The NVIDIA Nemotron open stack: open models, open software (NeMo Curator), open recipes (the Nemotron-CC pipeline), plus the official NeMo agent-lifecycle diagram |
| 3 | The proof, part 1 | Sommelier: fine-tune a small open model into a reliable JSON tool caller, with a live tool-call demo |
| 4 | The proof, part 2 | Base vs. adapter benchmarks from the reference run (n=1,000, greedy, conservative parser), with per-metric delta badges |
| 5 | The economics | One L40S, 3 hours, ~$8 |
| 6 | The thesis | The weights are yours. Nobody can reprice it. Nobody can deprecate it. Sovereign AI in practice, marginal French/English slices nearly aligned |
| 7 | CTA | Open code, weights, data, evidence. Repo and Space links |

## Fact policy

Every claim on screen has an explicit source and evidence class:

- Benchmark metrics and runtime come from the repository README and
  `docs/results/reference-run.md` (run `nemotron-8b-full-3`). The public
  reports expose aggregate counts and checksums, but not raw generations. The
  ~$8 cost is a maintainer billing-console observation documented in
  `docs/blogposts/sommelier_blog_post.md`, not a public checksummed artifact.
  See `src/data/facts.ts` for per-number source notes.
- The opening headlines were adversarially fact-checked against primary
  sources (Anthropic announcement, OpenAI deprecation page and Verified
  Organization docs, Anthropic model-deprecations page, Llama 4
  Community License / AUP). See `src/data/headlines.ts` for citations.
- The Nemotron stack claims (NeMo Curator Apache-2.0, public
  Nemotron-CC recipe, open weights under the NVIDIA Open Model License)
  were verified against GitHub, Hugging Face, and NVIDIA's blog.
- The NeMo diagram is the repository asset `docs/img/gtcdc25-nemo-diagram.png`.

## Usage

```bash
npm install
npm run dev      # Remotion Studio (interactive preview)
npm run render   # renders out/sommelier_is_here_sovereign_ai_is_needed.mp4
```

The video is silent by design; add a music bed in your editor of choice
if the target platform benefits from one.
