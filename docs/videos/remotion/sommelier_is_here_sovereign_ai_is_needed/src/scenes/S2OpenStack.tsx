import React from 'react';
import {
  Img,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {colors, fonts} from '../theme';
import {MonoKicker, Stage} from '../components/Layout';

// Verified July 2026:
// - NeMo Curator: Apache-2.0, github.com/NVIDIA-NeMo/Curator
// - Nemotron-CC recipe: github.com/NVIDIA-NeMo/Nemotron
//   (src/nemotron/recipes/data/curation/nemotron-cc)
// - Nemotron weights: huggingface.co/nvidia (NVIDIA Open Model License)
const LAYERS = [
  {
    index: '01',
    name: 'OPEN MODELS',
    detail: 'Nemotron weights, downloadable on Hugging Face',
  },
  {
    index: '02',
    name: 'OPEN SOFTWARE',
    detail: 'NeMo Curator: GPU-accelerated data curation, Apache-2.0',
  },
  {
    index: '03',
    name: 'OPEN RECIPES',
    detail: 'The Nemotron-CC data pipeline itself, published end to end',
  },
];

// The rows make their case, then hand the stage to the diagram beat.
const ROWS_OUT = 196;
const DIAGRAM_IN = 206;

const Row: React.FC<{
  index: string;
  name: string;
  detail: string;
  delay: number;
}> = ({index, name, detail, delay}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({
    frame,
    fps,
    delay,
    config: {damping: 20, stiffness: 200},
  });
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'baseline',
        gap: 34,
        opacity: enter,
        transform: `translateX(${(1 - enter) * -80}px)`,
        marginBottom: 34,
      }}
    >
      <div
        style={{
          fontFamily: fonts.mono,
          fontSize: 34,
          fontWeight: 700,
          color: colors.green,
        }}
      >
        {index}
      </div>
      <div>
        <div style={{fontSize: 64, fontWeight: 900, lineHeight: 1.05}}>
          {name}
        </div>
        <div
          style={{
            fontSize: 30,
            fontWeight: 500,
            color: colors.inkDim,
            marginTop: 6,
          }}
        >
          {detail}
        </div>
      </div>
    </div>
  );
};

export const S2OpenStack: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const kickerIn = interpolate(frame, [0, 12], [0, 1], {
    extrapolateRight: 'clamp',
  });
  const titleIn = spring({frame, fps, delay: 8, config: {damping: 200}});

  const rowsOut = spring({
    frame,
    fps,
    delay: ROWS_OUT,
    config: {damping: 200},
    durationInFrames: 16,
  });
  const diagramIn = spring({
    frame,
    fps,
    delay: DIAGRAM_IN,
    config: {damping: 22, stiffness: 160},
  });
  const punchIn = spring({frame, fps, delay: DIAGRAM_IN + 24, config: {damping: 200}});

  return (
    <Stage tint="rgba(118, 185, 0, 0.08)">
      <div style={{position: 'absolute', left: 120, top: 92, right: 120}}>
        <MonoKicker color={colors.green} style={{opacity: kickerIn}}>
          There is another way
        </MonoKicker>
        <div
          style={{
            fontSize: 84,
            fontWeight: 900,
            lineHeight: 1.04,
            marginTop: 22,
            marginBottom: 56,
            opacity: titleIn,
            transform: `translateY(${(1 - titleIn) * 40}px)`,
          }}
        >
          The open stack is already here.
        </div>

        <div
          style={{
            opacity: 1 - rowsOut,
            transform: `translateY(${rowsOut * -60}px)`,
          }}
        >
          {LAYERS.map((l, i) => (
            <Row
              key={l.index}
              index={l.index}
              name={l.name}
              detail={l.detail}
              delay={40 + i * 35}
            />
          ))}
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: 150,
          right: 150,
          top: 350,
          opacity: diagramIn,
          transform: `translateY(${(1 - diagramIn) * 110}px)`,
        }}
      >
        <div
          style={{
            background: colors.paper,
            borderRadius: 10,
            padding: '26px 34px',
            boxShadow: '0 24px 60px rgba(0,0,0,0.5)',
          }}
        >
          <Img
            src={staticFile('gtcdc25-nemo-diagram.png')}
            style={{width: '100%', display: 'block'}}
          />
        </div>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 36,
            fontWeight: 700,
            color: colors.ink,
            marginTop: 42,
            textAlign: 'center',
            opacity: punchIn,
          }}
        >
          Not a paper describing the pipeline.{' '}
          <span style={{color: colors.green}}>The pipeline.</span>
        </div>
      </div>
    </Stage>
  );
};
