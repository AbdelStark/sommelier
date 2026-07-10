import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {EVAL_FOOTNOTE, METRICS, MODEL_LINE} from '../data/facts';
import {colors, fonts} from '../theme';
import {MonoKicker, Stage} from '../components/Layout';

const BAR_MAX = 1080; // px width for 100%

const Bar: React.FC<{
  value: number;
  color: string;
  label: string;
  delay: number;
  emphasize?: boolean;
}> = ({value, color, label, delay, emphasize}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const progress = spring({
    frame,
    fps,
    delay,
    config: {damping: 200},
    durationInFrames: 34,
  });
  const width = (value / 100) * BAR_MAX * progress;
  const shown = value * progress;

  return (
    <div style={{display: 'flex', alignItems: 'center', gap: 24, height: 44}}>
      <div
        style={{
          fontFamily: fonts.mono,
          fontSize: 24,
          fontWeight: 700,
          color: colors.inkFaint,
          width: 120,
          textAlign: 'right',
        }}
      >
        {label}
      </div>
      <div
        style={{
          width,
          height: emphasize ? 30 : 22,
          background: color,
          borderRadius: 4,
        }}
      />
      <div
        style={{
          fontFamily: fonts.mono,
          fontSize: emphasize ? 34 : 27,
          fontWeight: 700,
          color: emphasize ? colors.ink : colors.inkDim,
        }}
      >
        {shown.toFixed(1)}%
      </div>
    </div>
  );
};

export const S4Numbers: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const headIn = spring({frame, fps, config: {damping: 200}});
  const footIn = spring({frame, fps, delay: 104, config: {damping: 200}});

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, right: 120, top: 96}}>
        <MonoKicker color={colors.wineBright} style={{opacity: headIn}}>
          Measured, not vibes
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
          Base model vs. an $8 fine-tune.
        </div>

        {METRICS.map((m, i) => {
          const delay = 14 + i * 22;
          return (
            <div key={m.label} style={{marginBottom: 40}}>
              <div
                style={{
                  fontSize: 34,
                  fontWeight: 700,
                  marginBottom: 12,
                  color: colors.ink,
                }}
              >
                {m.label}
              </div>
              <div
                style={{display: 'flex', flexDirection: 'column', gap: 6}}
              >
                <Bar
                  value={m.base}
                  color={colors.bronze}
                  label="base"
                  delay={delay}
                />
                <Bar
                  value={m.adapter}
                  color={colors.wine}
                  label="tuned"
                  delay={delay + 8}
                  emphasize
                />
              </div>
            </div>
          );
        })}

        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 24,
            color: colors.inkFaint,
            opacity: footIn,
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>{EVAL_FOOTNOTE}</span>
          <span>{MODEL_LINE}</span>
        </div>
      </div>
    </Stage>
  );
};
