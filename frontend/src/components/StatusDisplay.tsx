// frontend/src/components/StatusDisplay.tsx
import React from 'react';
import { useAppStore, AppState } from '../lib/store';
import './StatusDisplay.css';

export const StatusDisplay: React.FC = () => {
  const currentAppState = useAppStore((state) => state.appState);

  const getStatusInfo = (state: AppState): { text: string; color: string } => {
    switch (state) {
      case AppState.LISTENING: return { text: 'Listening...', color: 'var(--listening-color)' };
      case AppState.THINKING: return { text: 'Thinking...', color: 'var(--thinking-color)' };
      case AppState.SPEAKING: return { text: 'Speaking...', color: 'var(--speaking-color)' };
      case AppState.IDLE: return { text: 'Idle', color: 'grey' };
      default: return { text: 'Unknown', color: 'grey' };
    }
  };

  const { text, color } = getStatusInfo(currentAppState);

  return (
    <div className="status-indicator">
      <span className="status-light" style={{ backgroundColor: color }}></span>
      <span className="status-text">{text}</span>
    </div>
  );
};