// frontend/src/App.tsx
import React from 'react';
import { useAppStore } from './lib/store';
import { MicrophoneButton } from './components/MicrophoneButton';
import { StatusDisplay } from './components/StatusDisplay';
import { TranscriptDisplay } from './components/TranscriptDisplay';
import { SpeakingAnimation } from './components/SpeakingAnimation';
import './App.css'; // App specific styles

function App() {
  const errorMessage = useAppStore((state) => state.errorMessage);

  return (
    <main className="app-container">
      <h1>Astra Voice Assistant</h1>

      <StatusDisplay />

      <SpeakingAnimation />

      <TranscriptDisplay />

      {/* Display error messages */}
      {errorMessage && (
        <div className="error-message">Error: {errorMessage}</div>
      )}

      <MicrophoneButton />
    </main>
  );
}

export default App;