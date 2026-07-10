import React from 'react';
import {TransitionSeries, linearTiming} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';
import {slide} from '@remotion/transitions/slide';
import {SCENE, TRANSITION_FRAMES} from './durations';
import {S1Gatekeeping} from './scenes/S1Gatekeeping';
import {S2OpenStack} from './scenes/S2OpenStack';
import {S3Sommelier} from './scenes/S3Sommelier';
import {S4Numbers} from './scenes/S4Numbers';
import {S5Cost} from './scenes/S5Cost';
import {S6Sovereignty} from './scenes/S6Sovereignty';
import {S7CTA} from './scenes/S7CTA';

const timing = linearTiming({durationInFrames: TRANSITION_FRAMES});

export const Video: React.FC = () => {
  return (
    <TransitionSeries>
      <TransitionSeries.Sequence durationInFrames={SCENE.gatekeeping}>
        <S1Gatekeeping />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.openStack}>
        <S2OpenStack />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={slide({direction: 'from-right'})}
        timing={timing}
      />

      <TransitionSeries.Sequence durationInFrames={SCENE.sommelier}>
        <S3Sommelier />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.numbers}>
        <S4Numbers />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.cost}>
        <S5Cost />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.sovereignty}>
        <S6Sovereignty />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.cta}>
        <S7CTA />
      </TransitionSeries.Sequence>
    </TransitionSeries>
  );
};
