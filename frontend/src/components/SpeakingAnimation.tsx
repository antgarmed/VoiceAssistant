// frontend/src/components/SpeakingAnimation.tsx
import React from 'react';
import { useAppStore, AppState } from '../lib/store';
import './SpeakingAnimation.css';

export const SpeakingAnimation: React.FC = () => {
  const currentAppState = useAppStore((state) => state.appState);
  const isActive = currentAppState === AppState.SPEAKING;

  return (
    <div className={`animation-container ${isActive ? 'active' : ''}`}>
      {isActive && (
        <div className="speaking-bars">
          <span></span>
          <span></span>
          <span></span>
          <span></span>
        </div>
      )}
    </div>
  );
};