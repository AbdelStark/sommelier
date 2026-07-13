import {loadFont as loadArchivo} from '@remotion/google-fonts/Archivo';
import {loadFont as loadJetBrainsMono} from '@remotion/google-fonts/JetBrainsMono';

const archivo = loadArchivo('normal', {
  weights: ['500', '700', '900'],
  subsets: ['latin'],
});

const jetbrains = loadJetBrainsMono('normal', {
  weights: ['400', '700'],
  subsets: ['latin'],
});

export const fonts = {
  display: archivo.fontFamily,
  mono: jetbrains.fontFamily,
};

// Palette reprise de la première vidéo (marque Sommelier : vin, papier
// chaud, bronze pour le modèle de base) plus le vert NVIDIA pour la
// séquence consacrée à la pile ouverte.
export const colors = {
  bg: '#12100e',
  bgRaised: '#1c1712',
  ink: '#faf8f4',
  inkDim: '#a89f90',
  inkFaint: '#6f6759',
  hairline: 'rgba(250, 248, 244, 0.16)',
  wine: '#b93d58',
  wineBright: '#e06078',
  green: '#76b900',
  bronze: '#a87a35',
  paper: '#faf8f4',
};
