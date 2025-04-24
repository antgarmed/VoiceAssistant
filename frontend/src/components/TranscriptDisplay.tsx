// frontend/src/components/TranscriptDisplay.tsx
import React, { useRef, useEffect } from 'react';
import { useAppStore } from '../lib/store';
import './TranscriptDisplay.css';

export const TranscriptDisplay: React.FC = () => {
  const userTranscript = useAppStore((state) => state.userTranscript);
  const assistantResponse = useAppStore((state) => state.assistantResponse);
  const containerRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [userTranscript, assistantResponse]);

  return (
    <div className="transcript-container" ref={containerRef}>
      {!userTranscript && !assistantResponse ? (
        <div className="placeholder">Press Record and start speaking...</div>
      ) : (
        <div className="conversation-log">
          {userTranscript && (
            <div className="message user-message">
              <strong>You:</strong> {userTranscript}
            </div>
          )}
          {assistantResponse && (
            <div className="message assistant-message">
              <strong>Astra:</strong> {assistantResponse}
            </div>
          )}
        </div>
      )}
    </div>
  );
};