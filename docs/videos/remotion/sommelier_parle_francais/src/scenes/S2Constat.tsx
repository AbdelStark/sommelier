import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {CONSTAT, EVAL_NOTE, MODELE_LIGNE} from '../data/facts';
import {colors, fonts} from '../theme';
import {Badge, Bar} from '../components/Bars';
import {MonoKicker, Stage} from '../components/Layout';

export const S2Constat: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const headIn = spring({frame, fps, config: {damping: 200}});
  const punchIn = spring({frame, fps, delay: 168, config: {damping: 200}});
  const footIn = spring({frame, fps, delay: 200, config: {damping: 200}});

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, right: 120, top: 120}}>
        <MonoKicker color={colors.wineBright} style={{opacity: headIn}}>
          Même modèle, même tâche, deux langues
        </MonoKicker>
        <div
          style={{
            fontSize: 72,
            fontWeight: 900,
            marginTop: 20,
            marginBottom: 70,
            opacity: headIn,
            transform: `translateY(${(1 - headIn) * 30}px)`,
          }}
        >
          En français, le même modèle perd 4,2 points.
        </div>

        <div style={{marginBottom: 44}}>
          <div
            style={{
              fontSize: 34,
              fontWeight: 700,
              marginBottom: 14,
              color: colors.ink,
            }}
          >
            Appel complet exact
            <Badge text={CONSTAT.ecart} delay={120} />
          </div>
          <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
            <Bar
              value={CONSTAT.en}
              color={colors.bronze}
              label="anglais"
              delay={30}
            />
            <Bar
              value={CONSTAT.frSlice}
              color={colors.wine}
              label="français"
              delay={44}
              emphasize
            />
          </div>
        </div>

        <div
          style={{
            fontSize: 44,
            fontWeight: 700,
            color: colors.inkDim,
            marginTop: 66,
            opacity: punchIn,
            transform: `translateY(${(1 - punchIn) * 26}px)`,
          }}
        >
          Une taxe silencieuse,{' '}
          <span style={{color: colors.ink}}>sur chaque requête.</span>
        </div>

        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 23,
            color: colors.inkFaint,
            marginTop: 78,
            opacity: footIn,
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>{EVAL_NOTE}</span>
          <span>{MODELE_LIGNE.split(' · ')[0]}</span>
        </div>
      </div>
    </Stage>
  );
};
