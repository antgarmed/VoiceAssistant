// frontend/src/components/MicrophoneButton.tsx
import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useAppStore, AppState } from '../lib/store';
import {
  startRecording,
  stopRecording,
  stopMediaStream,
  cleanupAudio,
  handleIncomingAudioChunk,
} from '../lib/audioUtils';
import { sendAudioToBackend, createTTSWebSocket, resetBackendHistory } from '../lib/api';

import MicIcon from './icons/MicIcon';
import StopIcon from './icons/StopIcon';
import ResetIcon from './icons/ResetIcon';
import './MicrophoneButton.css';

export const MicrophoneButton: React.FC = () => {
  const {
    appState,
    isRecording,
    setAppState,
    setUserTranscript,
    setAssistantResponse,
    setError,
    resetConversationState,
    setErrorMessage,
  } = useAppStore();

  const [ttsWebSocket, setTtsWebSocket] = useState<WebSocket | null>(null);
  
  // Add a ref to track the current utterance ID
  const currentUtteranceIdRef = useRef<string | null>(null);

  useEffect(() => {
    // This effect runs on unmount or when ttsWebSocket changes
    return () => {
      console.log('MicrophoneButton unmount or ttsWebSocket change: closing WebSocket only');
      if (ttsWebSocket && ttsWebSocket.readyState === WebSocket.OPEN) {
        ttsWebSocket.close();
      }
      // Don't call cleanupAudio() here - let audio playback finish naturally
    };
  }, [ttsWebSocket]);

  const startStreamingPlayback = (text: string) => {
    if (!text) {
      console.warn('No text to stream.');
      setAppState(AppState.IDLE);
      return;
    }

    // Always clean up previous audio context before starting a new one
    cleanupAudio();

    // Close previous WebSocket if it exists
    if (ttsWebSocket && ttsWebSocket.readyState === WebSocket.OPEN) {
      ttsWebSocket.close();
      setTtsWebSocket(null);
    }

    const utteranceId = uuidv4();
    currentUtteranceIdRef.current = utteranceId;
    
    // Create and configure the new WebSocket
    const socket = createTTSWebSocket(
      utteranceId,
      text,
      (audioChunk) => {
        // Only process chunks if this is still the active utterance
        if (currentUtteranceIdRef.current === utteranceId) {
          handleIncomingAudioChunk(audioChunk);
        }
      },
      () => {
        // Server sent "stream_end" - close THIS SPECIFIC WebSocket immediately
        console.log(`Streaming ended for utterance: ${utteranceId}`);
        if (currentUtteranceIdRef.current === utteranceId && socket.readyState === WebSocket.OPEN) {
          // Use the direct socket reference, not the possibly stale ttsWebSocket state
          socket.close();
          setTtsWebSocket(null);
          // Note: Do NOT setAppState(AppState.IDLE) here
          // This will be handled by the AudioContext's onended handler
        }
      },
      (error) => {
        console.error(`TTS stream error:`, error);
        setError(`TTS streaming error.`);
        if (currentUtteranceIdRef.current === utteranceId) {
          currentUtteranceIdRef.current = null;
          if (socket.readyState === WebSocket.OPEN) {
            socket.close();
          }
          setTtsWebSocket(null);
          cleanupAudio();
          setAppState(AppState.IDLE);
        }
      }
    );

    setTtsWebSocket(socket);
    setAppState(AppState.SPEAKING);
  };

  const toggleRecording = async () => {
    setErrorMessage('');

    // If speaking, stop the current speech
    if (appState === AppState.SPEAKING) {
      console.log('Stopping speech...');
      currentUtteranceIdRef.current = null;
      
      if (ttsWebSocket && ttsWebSocket.readyState === WebSocket.OPEN) {
        ttsWebSocket.close();
        setTtsWebSocket(null);
      }
      
      // Don't stop the media stream here, only clean up audio
      cleanupAudio(); // This will properly clean up the audio
      setAppState(AppState.IDLE);
      return;
    }

    if (isRecording) {
      console.log('Stopping recording...');
      const blob = await stopRecording();
      if (blob && blob.size > 100) {
        setAppState(AppState.THINKING);
        try {
          const response = await sendAudioToBackend(blob);
          if (response?.assistant_response) {
            setUserTranscript(response.user_transcript || '');
            setAssistantResponse(response.assistant_response);
            startStreamingPlayback(response.assistant_response);
          } else {
            setError('Assistant gave no response.');
            setAppState(AppState.IDLE);
          }
        } catch (e) {
          console.error('Error sending audio:', e);
          setAppState(AppState.IDLE);
        }
      } else {
        console.warn('Invalid or empty audio.');
        setAppState(AppState.IDLE);
      }
    } else {
      console.log('Starting recording...');
      await startRecording();
    }
  };

  const handleReset = async () => {
    console.log('Resetting conversation...');
    
    // Reset all flags
    currentUtteranceIdRef.current = null;
    
    // Correct order: stop media stream first, then clean up audio
    stopMediaStream();
    cleanupAudio();
    
    if (ttsWebSocket && ttsWebSocket.readyState === WebSocket.OPEN) {
      ttsWebSocket.close();
      setTtsWebSocket(null);
    }
    
    resetConversationState();
    await resetBackendHistory();
    setAppState(AppState.IDLE);
  };

  const isThinking = appState === AppState.THINKING;
  const isSpeaking = appState === AppState.SPEAKING;
  const isDisabled = isThinking; // disable recording when thinking

  return (
    <div className="controls">
      <button
        onClick={toggleRecording}
        disabled={isDisabled}
        className={`mic-button ${isRecording ? 'recording' : ''} ${isSpeaking ? 'speaking' : ''}`}
      >
        {isRecording ? (
          <>
            <StopIcon /> Stop Recording
          </>
        ) : isThinking ? (
          <>
            <div className="spinner"></div> Thinking...
          </>
        ) : isSpeaking ? (
          <>
            <ResetIcon /> Stop Speaking
          </>
        ) : (
          <>
            <MicIcon /> Record
          </>
        )}
      </button>

      <button
        onClick={handleReset}
        disabled={isDisabled || isRecording || isSpeaking}
        className="reset-button"
        title="Reset Conversation"
      >
        <ResetIcon />
      </button>
    </div>
  );
};
