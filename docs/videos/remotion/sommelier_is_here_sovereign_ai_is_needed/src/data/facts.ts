// Public metric and runtime values here trace to repository reports. The cost
// is separately labeled as a maintainer billing-console observation. Do not
// edit without re-checking the source and evidence class.
//
// Source: README.md "Reference result" + docs/results/reference-run.md
// (run nemotron-8b-full-3, n=1,000 held-out prompts, greedy decoding,
// conservative parser, parse failures count as failures).
export type Metric = {
  label: string;
  base: number; // percent
  adapter: number; // percent
};

export const METRICS: Metric[] = [
  {label: 'Valid JSON rate', base: 91.6, adapter: 100.0},
  {label: 'Full-call exact match', base: 70.5, adapter: 87.4},
  {label: 'Argument F1', base: 75.7, adapter: 92.9},
];

// Source: docs/blogposts/sommelier_blog_post.md — "about three hours on one
// rented L40S", "The GPU bill for the successful run was about eight dollars",
// training details table (1x L40S, 3.05 h training).
export const COST = {
  gpu: 'ONE L40S',
  time: '3 HOURS',
  dollars: '~$8',
};

// Source: README.md "Multilingual result" — v2 adapter en 0.870 vs fr 0.873
// (+0.3 pts), fr test slice n=879, base model gap -4.2 pts.
export const FRENCH = {
  note: 'Bilingual: marginal French–English slices differ by +0.3 pts (fr n=879, en n=1,000; not paired).',
};

// Source: README.md serve example (verbatim tool call shape the adapter emits).
export const DEMO_QUERY = 'What is the weather in Paris today?';
export const DEMO_CALL = '{"name": "lookup_weather", "arguments": {"city": "Paris"}}';

export const MODEL_LINE =
  'nvidia/Llama-3.1-Nemotron-Nano-8B-v1 · QLoRA · 15k examples';

export const EVAL_FOOTNOTE =
  'n=1,000 held-out prompts · greedy · parse failures count as failures';
