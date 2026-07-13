import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {colors, fonts} from '../theme';
import {MonoKicker, Stage} from '../components/Layout';

const QUESTION_START = 150;
const MESURE_START = 236;

const Statement: React.FC<{
  text: string;
  delay: number;
  dimAfter: number;
}> = ({text, delay, dimAfter}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, delay, config: {damping: 18, stiffness: 210}});
  const dim = interpolate(frame, [dimAfter - 4, dimAfter + 12], [1, 0.3], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <div
      style={{
        fontSize: 84,
        fontWeight: 900,
        lineHeight: 1.16,
        opacity: enter * dim,
        transform: `translateY(${(1 - enter) * 50}px)`,
      }}
    >
      {text}
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
  const enter = spring({frame, fps, delay, config: {damping: 18, stiffness: 220}});
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

export const S1Question: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const kickerIn = interpolate(frame, [0, 12], [0, 1], {
    extrapolateRight: 'clamp',
  });
  const mesureIn = spring({
    frame,
    fps,
    delay: MESURE_START,
    config: {damping: 200},
  });

  const words = ['Combien', 'ça', 'coûte ?'];

  return (
    <Stage tint="rgba(185, 61, 88, 0.10)">
      <div
        style={{
          position: 'absolute',
          left: 120,
          top: 96,
          opacity: kickerIn,
        }}
      >
        <MonoKicker color={colors.wineBright}>
          Sommelier · le run français
        </MonoKicker>
      </div>

      <AbsoluteFill
        style={{
          justifyContent: 'center',
          alignItems: 'center',
          textAlign: 'center',
          flexDirection: 'column',
          gap: 8,
        }}
      >
        <Statement
          text="Vos clients parlent français."
          delay={16}
          dimAfter={QUESTION_START}
        />
        <Statement
          text="Vos agents raisonnent en anglais."
          delay={62}
          dimAfter={QUESTION_START}
        />

        <div
          style={{
            fontSize: 110,
            fontWeight: 900,
            lineHeight: 1.08,
            marginTop: 46,
          }}
        >
          {words.map((w, i) => (
            <QuestionWord
              key={w}
              word={w}
              delay={QUESTION_START + i * 7}
              accent={i === words.length - 1}
            />
          ))}
        </div>

        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 30,
            fontWeight: 700,
            color: colors.inkDim,
            marginTop: 48,
            opacity: mesureIn,
            transform: `translateY(${(1 - mesureIn) * 26}px)`,
          }}
        >
          Alors on a mesuré.
        </div>
      </AbsoluteFill>
    </Stage>
  );
};
