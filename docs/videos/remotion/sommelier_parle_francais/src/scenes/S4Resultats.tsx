import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {ECARTS, METRIQUES_FR, RESULTATS_NOTE} from '../data/facts';
import {colors, fonts} from '../theme';
import {Badge, Bar} from '../components/Bars';
import {MonoKicker, Stage} from '../components/Layout';

// Premier temps : les barres françaises plaident. Deuxième temps : le
// parcours de l'écart de langue prend la scène.
const BARS_OUT = 250;
const ECARTS_IN = 262;

const LigneEcart: React.FC<{
  modele: string;
  ecart: string;
  referme?: boolean;
  delay: number;
}> = ({modele, ecart, referme, delay}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, delay, config: {damping: 18, stiffness: 210}});
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        width: 1240,
        background: referme ? 'rgba(185, 61, 88, 0.14)' : colors.bgRaised,
        border: referme
          ? `2px solid ${colors.wine}`
          : `1px solid ${colors.hairline}`,
        borderRadius: 12,
        padding: '24px 40px',
        marginBottom: 22,
        opacity: enter,
        transform: `translateX(${(1 - enter) * -70}px) scale(${
          referme ? 0.98 + enter * 0.04 : 1
        })`,
        boxShadow: referme ? '0 24px 60px rgba(185,61,88,0.25)' : 'none',
      }}
    >
      <div style={{fontSize: 40, fontWeight: referme ? 900 : 700}}>
        {modele}
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 24,
        }}
      >
        {referme ? (
          <span
            style={{
              fontFamily: fonts.mono,
              fontSize: 24,
              fontWeight: 700,
              letterSpacing: '0.18em',
              color: colors.green,
            }}
          >
            TRANCHES +0,3 PT
          </span>
        ) : null}
        <span
          style={{
            fontFamily: fonts.mono,
            fontSize: 44,
            fontWeight: 700,
            color: referme ? colors.wineBright : colors.inkDim,
          }}
        >
          {ecart}
        </span>
      </div>
    </div>
  );
};

export const S4Resultats: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const headIn = spring({frame, fps, config: {damping: 200}});

  const barsOut = spring({
    frame,
    fps,
    delay: BARS_OUT,
    config: {damping: 200},
    durationInFrames: 16,
  });
  const sousTitreIn = spring({frame, fps, delay: ECARTS_IN, config: {damping: 200}});
  const punchIn = spring({
    frame,
    fps,
    delay: ECARTS_IN + 132,
    config: {damping: 16, stiffness: 220},
  });
  const footIn = spring({frame, fps, delay: ECARTS_IN + 150, config: {damping: 200}});

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, right: 120, top: 96}}>
        <MonoKicker color={colors.wineBright} style={{opacity: headIn}}>
          Le résultat
        </MonoKicker>
        <div
          style={{
            fontSize: 66,
            fontWeight: 900,
            marginTop: 20,
            marginBottom: 56,
            opacity: headIn,
            transform: `translateY(${(1 - headIn) * 30}px)`,
          }}
        >
          Le test français, avant / après.
        </div>

        <div
          style={{
            opacity: 1 - barsOut,
            transform: `translateY(${barsOut * -60}px)`,
          }}
        >
          {METRIQUES_FR.map((m, i) => {
            const delay = 24 + i * 52;
            const dernier = i === METRIQUES_FR.length - 1;
            return (
              <div key={m.label} style={{marginBottom: 38}}>
                <div
                  style={{
                    fontSize: 34,
                    fontWeight: 700,
                    marginBottom: 12,
                    color: colors.ink,
                  }}
                >
                  {m.label}
                  <Badge text={m.delta} delay={delay + 60} />
                </div>
                <div style={{display: 'flex', flexDirection: 'column', gap: 6}}>
                  <Bar
                    value={m.base}
                    color={colors.bronze}
                    label="base"
                    delay={delay}
                  />
                  <Bar
                    value={m.adapter}
                    color={colors.wine}
                    label="affiné"
                    delay={delay + 8}
                    emphasize={dernier}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: 120,
          right: 120,
          top: 320,
          opacity: sousTitreIn,
        }}
      >
        <div
          style={{
            fontSize: 52,
            fontWeight: 900,
            marginBottom: 42,
            transform: `translateY(${(1 - sousTitreIn) * 40}px)`,
          }}
        >
          Et face à l’anglais ?
        </div>
        {ECARTS.map((e, i) => (
          <LigneEcart
            key={e.modele}
            modele={e.modele}
            ecart={e.ecart}
            referme={e.referme}
            delay={ECARTS_IN + 16 + i * 34}
          />
        ))}
        <div
          style={{
            fontSize: 44,
            fontWeight: 700,
            marginTop: 44,
            opacity: punchIn,
            transform: `translateY(${(1 - punchIn) * 26}px)`,
          }}
        >
          Le français fait jeu égal.{' '}
          <span style={{color: colors.wineBright}}>Mesuré, pas promis.</span>
        </div>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 23,
            color: colors.inkFaint,
            marginTop: 40,
            opacity: footIn,
          }}
        >
          {RESULTATS_NOTE}
        </div>
      </div>
    </Stage>
  );
};
