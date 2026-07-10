import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {FRENCH} from '../data/facts';
import {colors, fonts} from '../theme';
import {Stage} from '../components/Layout';

const LINES = [
  'The weights are yours.',
  'Nobody can reprice it.',
  'Nobody can deprecate it.',
];

const Line: React.FC<{text: string; delay: number; dimAfter: number}> = ({
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
        fontSize: 82,
        fontWeight: 900,
        lineHeight: 1.16,
        opacity: enter * (dimmed ? 0.42 : 1),
        transform: `translateY(${(1 - enter) * 50}px)`,
      }}
    >
      {text}
    </div>
  );
};

export const S6Sovereignty: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const punchIn = spring({frame, fps, delay: 76, config: {damping: 16, stiffness: 200}});
  const noteIn = spring({frame, fps, delay: 100, config: {damping: 200}});

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
        {LINES.map((l, i) => (
          <Line key={l} text={l} delay={i * 22} dimAfter={78} />
        ))}
        <div
          style={{
            fontSize: 92,
            fontWeight: 900,
            color: colors.wineBright,
            marginTop: 30,
            opacity: punchIn,
            transform: `scale(${0.85 + punchIn * 0.15})`,
          }}
        >
          That is sovereign AI, in practice.
        </div>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 26,
            color: colors.inkFaint,
            marginTop: 44,
            opacity: noteIn,
          }}
        >
          {FRENCH.note}
        </div>
      </div>
    </Stage>
  );
};
