import React from 'react';
import {TransitionSeries, linearTiming} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';
import {slide} from '@remotion/transitions/slide';
import {SCENE, TRANSITION_FRAMES} from './durations';
import {S1Question} from './scenes/S1Question';
import {S2Constat} from './scenes/S2Constat';
import {S3Contrat} from './scenes/S3Contrat';
import {S4Resultats} from './scenes/S4Resultats';
import {S5Moyens} from './scenes/S5Moyens';
import {S6PileOuverte} from './scenes/S6PileOuverte';
import {S7Flywheel} from './scenes/S7Flywheel';
import {S8Souverainete} from './scenes/S8Souverainete';
import {S9CTA} from './scenes/S9CTA';

const timing = linearTiming({durationInFrames: TRANSITION_FRAMES});

export const Video: React.FC = () => {
  return (
    <TransitionSeries>
      <TransitionSeries.Sequence durationInFrames={SCENE.question}>
        <S1Question />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.constat}>
        <S2Constat />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.contrat}>
        <S3Contrat />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.resultats}>
        <S4Resultats />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.moyens}>
        <S5Moyens />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={slide({direction: 'from-right'})}
        timing={timing}
      />

      <TransitionSeries.Sequence durationInFrames={SCENE.pileOuverte}>
        <S6PileOuverte />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.flywheel}>
        <S7Flywheel />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.souverainete}>
        <S8Souverainete />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={fade()} timing={timing} />

      <TransitionSeries.Sequence durationInFrames={SCENE.cta}>
        <S9CTA />
      </TransitionSeries.Sequence>
    </TransitionSeries>
  );
};
