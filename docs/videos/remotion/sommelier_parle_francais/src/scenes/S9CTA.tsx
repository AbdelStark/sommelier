import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {colors, fonts} from '../theme';
import {Stage} from '../components/Layout';

const MOTS_OUVERTS = [
  'Code ouvert.',
  'Poids ouverts.',
  'Données ouvertes.',
  'Preuves ouvertes.',
];

export const S9CTA: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const glassIn = spring({frame, fps, config: {damping: 12, stiffness: 190}});
  const nameIn = spring({frame, fps, delay: 6, config: {damping: 200}});
  const linksIn = spring({frame, fps, delay: 88, config: {damping: 200}});
  const footIn = spring({frame, fps, delay: 110, config: {damping: 200}});

  const underline = spring({
    frame,
    fps,
    delay: 20,
    config: {damping: 200},
    durationInFrames: 34,
  });

  return (
    <Stage tint="rgba(185, 61, 88, 0.14)">
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
        <div style={{fontSize: 120, transform: `scale(${glassIn})`}}>🍷</div>
        <div
          style={{
            fontSize: 108,
            fontWeight: 900,
            letterSpacing: '0.13em',
            opacity: nameIn,
            marginTop: 6,
          }}
        >
          SOMMELIER
        </div>
        <div
          style={{
            height: 8,
            width: underline * 560,
            background: colors.wine,
            borderRadius: 4,
            marginTop: 18,
            marginBottom: 42,
          }}
        />

        <div
          style={{display: 'flex', gap: '0.55em', fontSize: 44, fontWeight: 700}}
        >
          {MOTS_OUVERTS.map((w, i) => {
            const e = spring({
              frame,
              fps,
              delay: 42 + i * 11,
              config: {damping: 16, stiffness: 240},
            });
            return (
              <span
                key={w}
                style={{
                  opacity: e,
                  transform: `translateY(${(1 - e) * 30}px)`,
                  color:
                    i === MOTS_OUVERTS.length - 1
                      ? colors.wineBright
                      : colors.ink,
                }}
              >
                {w}
              </span>
            );
          })}
        </div>

        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 32,
            fontWeight: 700,
            color: colors.ink,
            marginTop: 48,
            opacity: linksIn,
          }}
        >
          github.com/AbdelStark/sommelier
        </div>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 26,
            color: colors.inkDim,
            marginTop: 12,
            opacity: linksIn,
          }}
        >
          hf.co/spaces/abdelstark/sommelier
        </div>

        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 22,
            color: colors.inkFaint,
            marginTop: 46,
            opacity: footIn,
          }}
        >
          Licence MIT · chaque chiffre remonte à un artefact vérifiable
        </div>
      </div>
    </Stage>
  );
};
