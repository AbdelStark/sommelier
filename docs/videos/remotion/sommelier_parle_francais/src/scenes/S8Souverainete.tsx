import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {SOUVERAINETE_NOTE} from '../data/facts';
import {colors, fonts} from '../theme';
import {Stage} from '../components/Layout';

const LIGNES = [
  'Personne ne peut en changer le prix.',
  'Personne ne peut le retirer.',
  'Personne ne peut vous en couper l’accès.',
];

const Ligne: React.FC<{text: string; delay: number; dimAfter: number}> = ({
  text,
  delay,
  dimAfter,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, delay, config: {damping: 18, stiffness: 210}});
  const dimmed = frame >= dimAfter;
  return (
    <div
      style={{
        fontSize: 74,
        fontWeight: 900,
        lineHeight: 1.18,
        opacity: enter * (dimmed ? 0.42 : 1),
        transform: `translateY(${(1 - enter) * 50}px)`,
      }}
    >
      {text}
    </div>
  );
};

export const S8Souverainete: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const punchIn = spring({frame, fps, delay: 132, config: {damping: 16, stiffness: 200}});
  const noteIn = spring({frame, fps, delay: 176, config: {damping: 200}});

  return (
    <Stage tint="rgba(185, 61, 88, 0.10)">
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
        }}
      >
        {LIGNES.map((l, i) => (
          <Ligne key={l} text={l} delay={i * 36} dimAfter={126} />
        ))}
        <div
          style={{
            fontSize: 96,
            fontWeight: 900,
            color: colors.wineBright,
            marginTop: 34,
            opacity: punchIn,
            transform: `scale(${0.85 + punchIn * 0.15})`,
          }}
        >
          Vos poids. Vos données. Votre langue.
        </div>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 26,
            color: colors.inkFaint,
            marginTop: 46,
            opacity: noteIn,
          }}
        >
          {SOUVERAINETE_NOTE}
        </div>
      </div>
    </Stage>
  );
};
