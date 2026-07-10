import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {HEADLINES} from '../data/headlines';
import {colors, fonts} from '../theme';
import {Stage} from '../components/Layout';

const CARD_STAGGER = 22;
const QUESTION_START = 96;

const Card: React.FC<{
  index: number;
  kicker: string;
  title: string;
}> = ({index, kicker, title}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const enter = spring({
    frame,
    fps,
    delay: index * CARD_STAGGER,
    config: {damping: 16, stiffness: 260},
  });

  // The pile dims as the question takes over.
  const dim = interpolate(
    frame,
    [QUESTION_START - 6, QUESTION_START + 10],
    [1, 0.13],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );

  const offsets = [
    {x: -300, y: -300, r: -2.2},
    {x: 265, y: -105, r: 1.6},
    {x: -225, y: 95, r: -1.2},
    {x: 215, y: 295, r: 2.0},
  ];
  const o = offsets[index % offsets.length];

  return (
    <div
      style={{
        position: 'absolute',
        left: '50%',
        top: '50%',
        transform: `translate(calc(-50% + ${o.x}px), calc(-50% + ${o.y}px)) rotate(${o.r}deg) scale(${0.92 + enter * 0.08})`,
        opacity: enter * dim,
        width: 940,
        background: colors.bgRaised,
        border: `1px solid ${colors.hairline}`,
        borderLeft: `6px solid ${colors.wine}`,
        padding: '26px 40px 30px',
        boxShadow: '0 30px 70px rgba(0,0,0,0.55)',
      }}
    >
      <div
        style={{
          fontFamily: fonts.mono,
          fontSize: 22,
          fontWeight: 700,
          letterSpacing: '0.22em',
          color: colors.inkFaint,
          marginBottom: 14,
        }}
      >
        ● {kicker}
      </div>
      <div
        style={{
          fontSize: 42,
          fontWeight: 900,
          lineHeight: 1.12,
          color: colors.ink,
        }}
      >
        {title}
      </div>
    </div>
  );
};

const QuestionWord: React.FC<{
  word: string;
  delay: number;
  accent?: boolean;
}> = ({word, delay, accent}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({
    frame,
    fps,
    delay,
    config: {damping: 18, stiffness: 220},
  });
  return (
    <span
      style={{
        display: 'inline-block',
        opacity: enter,
        transform: `translateY(${(1 - enter) * 46}px)`,
        color: accent ? colors.wineBright : colors.ink,
        marginRight: '0.28em',
      }}
    >
      {word}
    </span>
  );
};

export const S1Gatekeeping: React.FC = () => {
  const frame = useCurrentFrame();

  const questionVisible = frame >= QUESTION_START;
  const line1 = ['Should', 'we', 'accept'];
  const line2 = ['gatekeeping', 'of', 'intelligence?'];

  return (
    <Stage tint="rgba(185, 61, 88, 0.10)">
      {HEADLINES.map((h, i) => (
        <Card key={h.title} index={i} kicker={h.kicker} title={h.title} />
      ))}

      {questionVisible ? (
        <AbsoluteFill
          style={{
            justifyContent: 'center',
            alignItems: 'center',
            textAlign: 'center',
          }}
        >
          <div style={{fontSize: 96, fontWeight: 900, lineHeight: 1.08}}>
            <div>
              {line1.map((w, i) => (
                <QuestionWord key={w} word={w} delay={QUESTION_START + i * 4} />
              ))}
            </div>
            <div>
              {line2.map((w, i) => (
                <QuestionWord
                  key={w}
                  word={w}
                  delay={QUESTION_START + 12 + i * 5}
                  accent={i === 0}
                />
              ))}
            </div>
          </div>
        </AbsoluteFill>
      ) : null}
    </Stage>
  );
};
