// Headline cards for the opening "gatekeeping" montage.
// Constraint: strictly factual, no invented outlets, no invented quotes.
// Every card was adversarially fact-checked against primary sources
// before rendering (July 2026). Citations:
//
// 1. anthropic.com/news/claude-fable-5-mythos-5 (Jun 9, 2026): "Claude
//    Mythos 5 is restricted to Glasswing partners ... until our broader
//    trusted access program is available."
// 2. OpenAI "Verified Organization" status (Apr 2025): government-issued
//    ID required to access some of its most advanced API models.
// 3. platform.claude.com model-deprecations page: 11 Claude models
//    retired between Jul 21, 2025 and Jun 15, 2026.
// 4. Llama 4 Acceptable Use Policy / Community License (Apr 2025):
//    rights not granted to individuals and companies domiciled in the EU.
export type Headline = {
  kicker: string; // small mono citation line
  title: string; // the punchy factual statement
};

export const HEADLINES: Headline[] = [
  {
    kicker: 'ANTHROPIC · JUN 2026',
    title: 'Most capable model: approved organizations only.',
  },
  {
    kicker: 'OPENAI · 2025',
    title: 'Government ID required to unlock top API models.',
  },
  {
    kicker: 'MODEL DEPRECATIONS · 2025–2026',
    title: '11 models retired in under a year.',
  },
  {
    kicker: 'LLAMA 4 LICENSE · 2025',
    title: 'EU companies and individuals: excluded.',
  },
];
