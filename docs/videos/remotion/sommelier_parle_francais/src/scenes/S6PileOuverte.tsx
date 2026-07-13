import React from 'react';
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {colors, fonts} from '../theme';
import {MonoKicker, Stage} from '../components/Layout';

// Vérifié en juillet 2026 (voir la première vidéo du dépôt) :
// - NeMo Curator : Apache-2.0, github.com/NVIDIA-NeMo/Curator
// - Recette Nemotron-CC : github.com/NVIDIA-NeMo/Nemotron
//   (src/nemotron/recipes/data/curation/nemotron-cc)
// - Poids Nemotron : huggingface.co/nvidia (NVIDIA Open Model License)
const COUCHES = [
  {
    index: '01',
    nom: 'MODÈLES OUVERTS',
    detail: 'Nemotron Nano 8B : les poids, téléchargeables sur Hugging Face',
  },
  {
    index: '02',
    nom: 'LOGICIELS OUVERTS',
    detail: 'NeMo Curator : curation de données sur GPU, licence Apache-2.0',
  },
  {
    index: '03',
    nom: 'RECETTES OUVERTES',
    detail: 'Le pipeline de données Nemotron-CC, publié de bout en bout',
  },
];

const Ligne: React.FC<{
  index: string;
  nom: string;
  detail: string;
  delay: number;
}> = ({index, nom, detail, delay}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({
    frame,
    fps,
    delay,
    config: {damping: 20, stiffness: 200},
  });
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'baseline',
        gap: 34,
        opacity: enter,
        transform: `translateX(${(1 - enter) * -80}px)`,
        marginBottom: 40,
      }}
    >
      <div
        style={{
          fontFamily: fonts.mono,
          fontSize: 34,
          fontWeight: 700,
          color: colors.green,
        }}
      >
        {index}
      </div>
      <div>
        <div style={{fontSize: 64, fontWeight: 900, lineHeight: 1.05}}>
          {nom}
        </div>
        <div
          style={{
            fontSize: 30,
            fontWeight: 500,
            color: colors.inkDim,
            marginTop: 6,
          }}
        >
          {detail}
        </div>
      </div>
    </div>
  );
};

export const S6PileOuverte: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const kickerIn = interpolate(frame, [0, 12], [0, 1], {
    extrapolateRight: 'clamp',
  });
  const titleIn = spring({frame, fps, delay: 8, config: {damping: 200}});
  const punchIn = spring({frame, fps, delay: 196, config: {damping: 200}});

  return (
    <Stage tint="rgba(118, 185, 0, 0.08)">
      <div style={{position: 'absolute', left: 120, top: 92, right: 120}}>
        <MonoKicker color={colors.green} style={{opacity: kickerIn}}>
          La pile ouverte NVIDIA
        </MonoKicker>
        <div
          style={{
            fontSize: 80,
            fontWeight: 900,
            lineHeight: 1.04,
            marginTop: 22,
            marginBottom: 56,
            opacity: titleIn,
            transform: `translateY(${(1 - titleIn) * 40}px)`,
          }}
        >
          Rien de tout ça n’exige un labo.
        </div>

        {COUCHES.map((c, i) => (
          <Ligne
            key={c.index}
            index={c.index}
            nom={c.nom}
            detail={c.detail}
            delay={40 + i * 38}
          />
        ))}
      </div>

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 96,
          textAlign: 'center',
          fontFamily: fonts.mono,
          fontSize: 36,
          fontWeight: 700,
          color: colors.ink,
          opacity: punchIn,
        }}
      >
        Les pelles sont à louer.{' '}
        <span style={{color: colors.green}}>Le modèle, lui, est à vous.</span>
      </div>
    </Stage>
  );
};
