import React from 'react';
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {FLYWHEEL_ETAPES} from '../data/facts';
import {colors, fonts} from '../theme';
import {MonoKicker, Stage} from '../components/Layout';

const CX = 960;
const CY = 545;
const R = 235; // rayon de l'anneau
const LABEL_R = 330; // les étiquettes vivent au-delà de l'anneau
const SWEEP_START = 50;
const SWEEP_END = 220;

const Node: React.FC<{
  label: string;
  index: number;
}> = ({label, index}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const angle = ((-90 + index * 72) * Math.PI) / 180;
  const x = CX + LABEL_R * Math.cos(angle);
  const y = CY + LABEL_R * Math.sin(angle);

  const enter = spring({
    frame,
    fps,
    delay: 10 + index * 10,
    config: {damping: 200},
  });
  // L'étiquette s'allume quand le balayage de l'anneau atteint son angle.
  const activation = SWEEP_START + index * ((SWEEP_END - SWEEP_START) / 5);
  const active = frame >= activation;
  const bump = spring({
    frame,
    fps,
    delay: activation,
    config: {damping: 13, stiffness: 260},
  });

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        transform: `translate(-50%, -50%) scale(${
          0.94 + enter * 0.06 + (active ? bump * 0.06 : 0)
        })`,
        opacity: enter,
        width: 300,
        textAlign: 'center',
        fontFamily: fonts.mono,
        fontSize: 22,
        fontWeight: 700,
        lineHeight: 1.35,
        letterSpacing: '0.06em',
        color: active ? colors.ink : colors.inkFaint,
        background: active ? 'rgba(185, 61, 88, 0.18)' : colors.bgRaised,
        border: active
          ? `2px solid ${colors.wine}`
          : `1px solid ${colors.hairline}`,
        borderRadius: 12,
        padding: '13px 16px',
        boxShadow: active ? '0 18px 44px rgba(185,61,88,0.28)' : 'none',
      }}
    >
      {label}
    </div>
  );
};

export const S7Flywheel: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const kickerIn = interpolate(frame, [0, 12], [0, 1], {
    extrapolateRight: 'clamp',
  });
  const ringIn = spring({frame, fps, delay: 6, config: {damping: 200}});
  const sweep = interpolate(frame, [SWEEP_START, SWEEP_END], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const loopIn = spring({frame, fps, delay: 226, config: {damping: 200}});
  const captionIn = spring({frame, fps, delay: 240, config: {damping: 200}});
  const sousNoteIn = spring({frame, fps, delay: 264, config: {damping: 200}});

  return (
    <Stage tint="rgba(185, 61, 88, 0.10)">
      <div style={{position: 'absolute', left: 120, top: 92}}>
        <MonoKicker color={colors.wineBright} style={{opacity: kickerIn}}>
          Le data flywheel, en vrai
        </MonoKicker>
      </div>

      <svg
        width={1920}
        height={1080}
        style={{position: 'absolute', inset: 0, opacity: ringIn}}
      >
        <circle
          cx={CX}
          cy={CY}
          r={R}
          fill="none"
          stroke={colors.hairline}
          strokeWidth={3}
        />
        <circle
          cx={CX}
          cy={CY}
          r={R}
          fill="none"
          stroke={colors.wine}
          strokeWidth={6}
          strokeLinecap="round"
          pathLength={1}
          strokeDasharray={`${sweep} 1`}
          transform={`rotate(-90 ${CX} ${CY})`}
        />
      </svg>

      {FLYWHEEL_ETAPES.map((label, i) => (
        <Node key={label} label={label} index={i} />
      ))}

      <div
        style={{
          position: 'absolute',
          left: CX,
          top: CY,
          transform: 'translate(-50%, -50%)',
          fontFamily: fonts.mono,
          fontSize: 28,
          fontWeight: 700,
          color: colors.green,
          opacity: loopIn,
        }}
      >
        tour suivant ↻
      </div>

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 130,
          textAlign: 'center',
          fontSize: 46,
          fontWeight: 900,
          opacity: captionIn,
          transform: `translateY(${(1 - captionIn) * 26}px)`,
        }}
      >
        Chaque tour produit les données du suivant.
      </div>
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 76,
          textAlign: 'center',
          fontFamily: fonts.mono,
          fontSize: 24,
          color: colors.inkFaint,
          opacity: sousNoteIn,
        }}
      >
        le v2 a tourné sur les artefacts publiés du v1 · le tour suivant peut
        partir d’ici
      </div>
    </Stage>
  );
};
