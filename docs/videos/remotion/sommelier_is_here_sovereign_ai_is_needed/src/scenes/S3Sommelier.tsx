import React from 'react';
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {DEMO_CALL, DEMO_QUERY} from '../data/facts';
import {colors, fonts} from '../theme';
import {Stage} from '../components/Layout';

const TYPE_START = 36;

export const S3Sommelier: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const glassIn = spring({frame, fps, config: {damping: 12, stiffness: 180}});
  const nameIn = spring({frame, fps, delay: 6, config: {damping: 200}});
  const tagIn = spring({frame, fps, delay: 18, config: {damping: 200}});
  const termIn = spring({frame, fps, delay: 36, config: {damping: 200}});

  // Typewriter: query first, then the tool call.
  const queryChars = Math.round(
    interpolate(frame, [TYPE_START, TYPE_START + 20], [0, DEMO_QUERY.length], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }),
  );
  const callChars = Math.round(
    interpolate(
      frame,
      [TYPE_START + 26, TYPE_START + 50],
      [0, DEMO_CALL.length],
      {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
    ),
  );
  const statusIn = spring({
    frame,
    fps,
    delay: TYPE_START + 52,
    config: {damping: 16, stiffness: 260},
  });

  const letterSpacing = interpolate(nameIn, [0, 1], [0.5, 0.14]);

  return (
    <Stage tint="rgba(185, 61, 88, 0.12)">
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 132,
          textAlign: 'center',
        }}
      >
        <div
          style={{
            fontSize: 110,
            transform: `scale(${glassIn})`,
            display: 'inline-block',
          }}
        >
          🍷
        </div>
        <div
          style={{
            fontSize: 100,
            fontWeight: 900,
            letterSpacing: `${letterSpacing}em`,
            opacity: nameIn,
            marginTop: 4,
          }}
        >
          SOMMELIER
        </div>
        <div
          style={{
            fontSize: 36,
            fontWeight: 500,
            color: colors.inkDim,
            marginTop: 18,
            opacity: tagIn,
            transform: `translateY(${(1 - tagIn) * 30}px)`,
          }}
        >
          Fine-tune a small open model into a reliable{' '}
          <span style={{color: colors.wineBright, fontWeight: 700}}>
            JSON tool caller
          </span>
          .
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: '50%',
          bottom: 120,
          transform: `translateX(-50%) translateY(${(1 - termIn) * 60}px)`,
          opacity: termIn,
          width: 1180,
          background: colors.bgRaised,
          border: `1px solid ${colors.hairline}`,
          borderRadius: 12,
          padding: '30px 40px',
          fontFamily: fonts.mono,
          fontSize: 30,
          lineHeight: 1.7,
          textAlign: 'left',
          boxShadow: '0 30px 70px rgba(0,0,0,0.55)',
        }}
      >
        <div style={{color: colors.inkDim}}>
          <span style={{color: colors.inkFaint}}>user ›</span>{' '}
          {DEMO_QUERY.slice(0, queryChars)}
        </div>
        <div style={{color: colors.ink, fontWeight: 700}}>
          <span style={{color: colors.wineBright}}>model ›</span>{' '}
          {DEMO_CALL.slice(0, callChars)}
        </div>
        <div
          style={{
            color: colors.green,
            fontWeight: 700,
            opacity: statusIn,
            transform: `scale(${0.9 + statusIn * 0.1})`,
            transformOrigin: 'left center',
          }}
        >
          parse_status: ok ✓
        </div>
      </div>
    </Stage>
  );
};
