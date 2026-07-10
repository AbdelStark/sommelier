export const FPS = 30;

export const SCENE = {
  gatekeeping: 168,
  openStack: 150,
  sommelier: 112,
  numbers: 168,
  cost: 100,
  sovereignty: 140,
  cta: 144,
};

export const TRANSITION_FRAMES = 11;
const TRANSITION_COUNT = 6;

export const TOTAL_FRAMES =
  Object.values(SCENE).reduce((a, b) => a + b, 0) -
  TRANSITION_COUNT * TRANSITION_FRAMES;
