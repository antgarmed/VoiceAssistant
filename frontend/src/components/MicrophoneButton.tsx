// frontend/src/components/MicrophoneButton.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useAppStore, AppState } from '../lib/store';
import { startRecording, stopRecording } from '../lib/audioUtils';
import { sendAudioToBackend, createAudioPlayer, resetBackendHistory } from '../lib/api';

// Import SVGs as components or use inline SVG
import MicIcon from './icons/MicIcon';
import StopIcon from './icons/StopIcon';
import ResetIcon from './icons/ResetIcon';
import './MicrophoneButton.css'; // Import CSS

export const MicrophoneButton: React.FC = () => {
  const {
    appState,
    isRecording,
    setAppState,
    setUserTranscript,
    setAssistantResponse,
    setAudioPlayer,
    setError,
    resetConversationState,
    setErrorMessage,
  } = useAppStore();

  // Local state for the player instance to manage cleanup
  const [currentPlayer, setCurrentPlayer] = useState<HTMLAudioElement | null>(null);
  const [playerObjectUrl, setPlayerObjectUrl] = useState<string | null>(null);

  // Cleanup function for audio player URL
  const cleanupPlayer = useCallback(() => {
    if (currentPlayer) {
      currentPlayer.pause();
      setCurrentPlayer(null);
      setAudioPlayer(null); // Update global store
    }
    if (playerObjectUrl) {
      URL.revokeObjectURL(playerObjectUrl);
      console.log("Revoked Object URL:", playerObjectUrl);
      setPlayerObjectUrl(null);
    }
  }, [currentPlayer, playerObjectUrl, setAudioPlayer]);


  useEffect(() => {
    // Ensure cleanup happens when the component unmounts
    return () => {
      console.log("MicrophoneButton unmounting: Cleaning up player and stream...");
      cleanupPlayer();
      stopMediaStream(); // Ensure mic stream is stopped if active (needs export from audioUtils or better state mgmt)
    };
  }, [cleanupPlayer]);


  const playAssistantAudio = (audioB64: string) => {
    cleanupPlayer(); // Clean up previous player first

    const newPlayer = createAudioPlayer(audioB64);
    if (newPlayer && newPlayer.src) {
      setCurrentPlayer(newPlayer);
      setAudioPlayer(newPlayer); // Update global store
      setPlayerObjectUrl(newPlayer.src); // Store the blob URL

      newPlayer.play()
        .then(() => {
          console.log("Audio playback started.");
          setAppState(AppState.SPEAKING);
        })
        .catch(error => {
          console.error("Error playing audio:", error);
          setError("Could not play assistant response audio.");
          cleanupPlayer(); // Cleanup on error
        });

      newPlayer.onended = () => {
        console.log("Audio playback finished.");
        setAppState(AppState.IDLE);
        cleanupPlayer(); // Clean up after playback
      };

      newPlayer.onerror = (e) => {
        console.error("Audio playback error event:", e);
        setError("Error during audio playback.");
        setAppState(AppState.IDLE);
        cleanupPlayer();
      };
    } else {
      setAppState(AppState.IDLE);
    }
  };

  const toggleRecording = async () => {
    setErrorMessage(''); // Clear previous errors

    if (isRecording) {
      console.log('Stopping recording...');
      setAppState(AppState.THINKING);
      const audioBlob = await stopRecording();

      if (audioBlob && audioBlob.size > 100) {
        console.log('Sending audio to backend...');
        const response = await sendAudioToBackend(audioBlob);

        if (response) {
          console.log('Backend response received:', response);
          setUserTranscript(response.user_transcript || '');
          setAssistantResponse(response.assistant_response || '');
          playAssistantAudio(response.assistant_audio_b64);
        } else {
          console.log('Backend call failed or returned null.');
          // Error state set by api.ts
        }
      } else {
        console.log('No valid audio blob captured or recording too short.');
        resetConversationState(); // Reset fully if no audio
        if (!useAppStore.getState().errorMessage) { // Check if specific mic error already set
           setError('Recording was too short or failed.');
        }
         setAppState(AppState.IDLE);
      }
    } else {
      await handleReset(); // Clear state before starting
      console.log('Starting recording...');
      const success = await startRecording();
      if (!success) {
        console.error('Failed to start recording.');
        // Error message set by startRecording
      }
    }
  };

  const handleReset = async () => {
    console.log("Resetting conversation state...");
    cleanupPlayer(); // Clean up any active audio player
    resetConversationState(); // Reset frontend state
    await resetBackendHistory(); // Reset backend history
  };

  const isDisabled = appState === AppState.THINKING || appState === AppState.SPEAKING;

  return (
    <div className="controls">
      <button
        onClick={toggleRecording}
        disabled={isDisabled}
        className={`mic-button ${isRecording ? 'recording' : ''}`}
      >
        {isRecording ? (
          <>
            <StopIcon /> Stop
          </>
        ) : appState === AppState.THINKING ? (
          <>
            <div className="spinner"></div> Thinking...
          </>
        ) : appState === AppState.SPEAKING ? (
           <>
             <div className="spinner"></div> Speaking...
           </>
        ) : (
          <>
            <MicIcon /> Record
          </>
        )}
      </button>

      <button onClick={handleReset} className="reset-button" title="Reset Conversation"
              disabled={isDisabled || isRecording}>
         <ResetIcon />
      </button>
    </div>
  );
};

// Helper function (if not exporting stopMediaStream from audioUtils)
const stopMediaStream = () => {
    // Implementation copied/adapted from audioUtils if needed locally
    console.warn("stopMediaStream called from MicrophoneButton - consider centralizing state/cleanup")
};