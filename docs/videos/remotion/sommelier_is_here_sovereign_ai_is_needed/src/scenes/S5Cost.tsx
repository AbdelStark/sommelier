import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {COST} from '../data/facts';
import {colors} from '../theme';
import {Stage} from '../components/Layout';

const Item: React.FC<{
  text: string;
  delay: number;
  accent?: boolean;
}> = ({text, delay, accent}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({
    frame,
    fps,
    delay,
    config: {damping: 13, stiffness: 240},
  });
  return (
    <div
      style={{
        fontSize: accent ? 190 : 120,
        fontWeight: 900,
        color: accent ? colors.wineBright : colors.ink,
        opacity: enter,
        transform: `scale(${0.7 + enter * 0.3})`,
        lineHeight: 1.02,
      }}
    >
      {text}
    </div>
  );
};

export const S5Cost: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const subIn = spring({frame, fps, delay: 56, config: {damping: 200}});

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
          gap: 6,
        }}
      >
        <Item text={COST.gpu} delay={0} />
        <Item text={COST.time} delay={14} />
        <Item text={COST.dollars} delay={30} accent />
        <div
          style={{
            fontSize: 40,
            fontWeight: 500,
            color: colors.inkDim,
            marginTop: 34,
            opacity: subIn,
            transform: `translateY(${(1 - subIn) * 30}px)`,
          }}
        >
          Post-training was a lab capability.{' '}
          <span style={{color: colors.ink, fontWeight: 700}}>
            Now it is a weekend.
          </span>
        </div>
      </div>
    </Stage>
  );
};
