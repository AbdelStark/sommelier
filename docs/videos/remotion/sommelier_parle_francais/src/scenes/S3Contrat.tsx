import React from 'react';
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {CONTRAT_NOTE, DEMO_EN, DEMO_FR, DEMO_GOLD} from '../data/facts';
import {colors, fonts} from '../theme';
import {MonoKicker, Stage} from '../components/Layout';

const TYPE_START = 70;

const Terminal: React.FC<{
  header: string;
  query: string;
  typedChars: number;
  goldVisible: number;
  delay: number;
}> = ({header, query, typedChars, goldVisible, delay}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, delay, config: {damping: 200}});
  return (
    <div
      style={{
        width: 830,
        background: colors.bgRaised,
        border: `1px solid ${colors.hairline}`,
        borderRadius: 12,
        padding: '26px 34px',
        fontFamily: fonts.mono,
        fontSize: 24,
        lineHeight: 1.75,
        opacity: enter,
        transform: `translateY(${(1 - enter) * 46}px)`,
        boxShadow: '0 30px 70px rgba(0,0,0,0.55)',
      }}
    >
      <div
        style={{
          fontSize: 20,
          fontWeight: 700,
          letterSpacing: '0.22em',
          color: colors.inkFaint,
          marginBottom: 14,
        }}
      >
        {header}
      </div>
      <div style={{color: colors.inkDim, minHeight: 84}}>
        <span style={{color: colors.inkFaint}}>user ›</span>{' '}
        {query.slice(0, typedChars)}
      </div>
      <div
        style={{
          color: colors.ink,
          fontWeight: 700,
          whiteSpace: 'pre-wrap',
          opacity: goldVisible,
        }}
      >
        <span style={{color: colors.wineBright}}>gold ›</span> {DEMO_GOLD}
      </div>
    </div>
  );
};

export const S3Contrat: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const headIn = spring({frame, fps, config: {damping: 200}});

  const typedEn = Math.round(
    interpolate(frame, [TYPE_START, TYPE_START + 40], [0, DEMO_EN.length], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }),
  );
  const typedFr = Math.round(
    interpolate(
      frame,
      [TYPE_START + 12, TYPE_START + 56],
      [0, DEMO_FR.length],
      {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
    ),
  );
  const goldIn = spring({
    frame,
    fps,
    delay: TYPE_START + 66,
    config: {damping: 200},
  });
  const sealIn = spring({
    frame,
    fps,
    delay: TYPE_START + 108,
    config: {damping: 16, stiffness: 240},
  });
  const noteIn = spring({
    frame,
    fps,
    delay: TYPE_START + 150,
    config: {damping: 200},
  });

  return (
    <Stage tint="rgba(185, 61, 88, 0.10)">
      <div style={{position: 'absolute', left: 120, right: 120, top: 100}}>
        <MonoKicker color={colors.wineBright} style={{opacity: headIn}}>
          La méthode · le contrat de traduction
        </MonoKicker>
        <div
          style={{
            fontSize: 68,
            fontWeight: 900,
            marginTop: 20,
            opacity: headIn,
            transform: `translateY(${(1 - headIn) * 30}px)`,
          }}
        >
          Une seule variable : la langue de la requête.
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: 120,
          right: 120,
          top: 340,
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <Terminal
          header="REQUÊTE · ANGLAIS"
          query={DEMO_EN}
          typedChars={typedEn}
          goldVisible={goldIn}
          delay={26}
        />
        <Terminal
          header="REQUÊTE · FRANÇAIS"
          query={DEMO_FR}
          typedChars={typedFr}
          goldVisible={goldIn}
          delay={40}
        />
      </div>

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 168,
          textAlign: 'center',
          opacity: sealIn,
          transform: `scale(${0.9 + sealIn * 0.1})`,
        }}
      >
        <span
          style={{
            display: 'inline-block',
            fontFamily: fonts.mono,
            fontSize: 30,
            fontWeight: 700,
            color: colors.green,
            background: 'rgba(118, 185, 0, 0.10)',
            border: `1px solid rgba(118, 185, 0, 0.5)`,
            borderRadius: 8,
            padding: '10px 26px',
          }}
        >
          Schémas et réponses gold : identiques à l’octet près ✓
        </span>
      </div>

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 100,
          textAlign: 'center',
          fontFamily: fonts.mono,
          fontSize: 24,
          color: colors.inkFaint,
          opacity: noteIn,
        }}
      >
        {CONTRAT_NOTE}
      </div>
    </Stage>
  );
};
