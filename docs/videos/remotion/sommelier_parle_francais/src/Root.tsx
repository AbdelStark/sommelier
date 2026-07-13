import React from 'react';
import {Composition} from 'remotion';
import {FPS, TOTAL_FRAMES} from './durations';
import {Video} from './Video';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="SommelierFrancais"
      component={Video}
      durationInFrames={TOTAL_FRAMES}
      fps={FPS}
      width={1920}
      height={1080}
    />
  );
};
