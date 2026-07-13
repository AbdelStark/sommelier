import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {pct} from '../data/facts';
import {colors, fonts} from '../theme';

const BAR_MAX = 1040; // largeur en px pour 100 %

export const Bar: React.FC<{
  value: number;
  color: string;
  label: string;
  delay: number;
  emphasize?: boolean;
  labelWidth?: number;
}> = ({value, color, label, delay, emphasize, labelWidth = 150}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const progress = spring({
    frame,
    fps,
    delay,
    config: {damping: 200},
    durationInFrames: 42,
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
          width: labelWidth,
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
        {pct(shown)}
      </div>
    </div>
  );
};

export const Badge: React.FC<{
  text: string;
  delay: number;
  color?: string;
}> = ({text, delay, color = colors.wineBright}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const pop = spring({
    frame,
    fps,
    delay,
    config: {damping: 14, stiffness: 240},
  });
  return (
    <span
      style={{
        display: 'inline-block',
        fontFamily: fonts.mono,
        fontSize: 26,
        fontWeight: 700,
        color,
        background: 'rgba(185, 61, 88, 0.16)',
        border: `1px solid ${colors.wine}`,
        borderRadius: 6,
        padding: '4px 14px',
        marginLeft: 22,
        verticalAlign: 'middle',
        opacity: pop,
        transform: `scale(${0.7 + pop * 0.3})`,
      }}
    >
      {text}
    </span>
  );
};
