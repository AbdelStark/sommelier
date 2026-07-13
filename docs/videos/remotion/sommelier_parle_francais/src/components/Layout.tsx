import React from 'react';
import {AbsoluteFill} from 'remotion';
import {colors, fonts} from '../theme';

// Scène partagée : quasi-noir chaud, grille de points discrète, vignettage.
export const Stage: React.FC<{
  children: React.ReactNode;
  tint?: string;
}> = ({children, tint}) => {
  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.bg,
        backgroundImage: `radial-gradient(rgba(250,248,244,0.05) 1px, transparent 1px)`,
        backgroundSize: '28px 28px',
        fontFamily: fonts.display,
        color: colors.ink,
        overflow: 'hidden',
      }}
    >
      {tint ? (
        <AbsoluteFill
          style={{
            background: `radial-gradient(ellipse 90% 70% at 50% 30%, ${tint}, transparent 70%)`,
          }}
        />
      ) : null}
      <AbsoluteFill
        style={{
          boxShadow: 'inset 0 0 340px rgba(0,0,0,0.75)',
        }}
      />
      {children}
    </AbsoluteFill>
  );
};

export const MonoKicker: React.FC<{
  children: React.ReactNode;
  color?: string;
  style?: React.CSSProperties;
}> = ({children, color = colors.inkDim, style}) => {
  return (
    <div
      style={{
        fontFamily: fonts.mono,
        fontSize: 26,
        fontWeight: 700,
        letterSpacing: '0.32em',
        textTransform: 'uppercase',
        color,
        ...style,
      }}
    >
      {children}
    </div>
  );
};
