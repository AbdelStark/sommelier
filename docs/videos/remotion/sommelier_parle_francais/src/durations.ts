export const FPS = 30;

// Rythmé pour la lecture : chaque scène tient assez longtemps pour lire
// son élément le plus dense à vitesse confortable (~3 mots/seconde plus
// un temps de pose). Total ~95 s, dans la cible d'une à deux minutes.
export const SCENE = {
  question: 310,
  constat: 330,
  contrat: 400,
  resultats: 480,
  moyens: 240,
  pileOuverte: 300,
  flywheel: 340,
  souverainete: 330,
  cta: 230,
};

export const TRANSITION_FRAMES = 14;
const TRANSITION_COUNT = 8;

export const TOTAL_FRAMES =
  Object.values(SCENE).reduce((a, b) => a + b, 0) -
  TRANSITION_COUNT * TRANSITION_FRAMES;
