export const FPS = 30;

// Paced for reading: each scene holds long enough to read its densest
// element at a comfortable speed (~3 words/second plus settle time).
export const SCENE = {
  gatekeeping: 340,
  openStack: 320,
  sommelier: 230,
  numbers: 320,
  cost: 170,
  sovereignty: 260,
  cta: 200,
};

export const TRANSITION_FRAMES = 14;
const TRANSITION_COUNT = 6;

export const TOTAL_FRAMES =
  Object.values(SCENE).reduce((a, b) => a + b, 0) -
  TRANSITION_COUNT * TRANSITION_FRAMES;
