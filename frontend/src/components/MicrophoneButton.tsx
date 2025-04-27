// frontend/src/components/MicrophoneButton.tsx
import React, { useState, useEffect } from 'react';
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

  useEffect(() => {
    return () => {
      console.log('MicrophoneButton unmount: cleaning up resources.');
      ttsWebSocket?.close();
      stopMediaStream();
      cleanupAudio();
    };
  }, [ttsWebSocket]);

  const startStreamingPlayback = (text: string) => {
    if (!text) {
      console.warn('No text to stream.');
      setAppState(AppState.IDLE);
      return;
    }

    cleanupAudio();

    if (ttsWebSocket) {
      ttsWebSocket.close();
      setTtsWebSocket(null);
    }

    const utteranceId = uuidv4();
    const socket = createTTSWebSocket(
      utteranceId,
      text,
      (audioChunk) => handleIncomingAudioChunk(audioChunk),
      () => {
        console.log(`Streaming ended for utterance: ${utteranceId}`);
        setTtsWebSocket(null);
        setAppState(AppState.IDLE);
      },
      (error) => {
        console.error(`TTS stream error:`, error);
        setError(`TTS streaming error.`);
        cleanupAudio();
        setTtsWebSocket(null);
        setAppState(AppState.IDLE);
      }
    );

    setTtsWebSocket(socket);
    setAppState(AppState.SPEAKING);
  };

  const toggleRecording = async () => {
    setErrorMessage('');

    if (ttsWebSocket) {
      ttsWebSocket.close();
      setTtsWebSocket(null);
      cleanupAudio();
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
    stopMediaStream();
    cleanupAudio();
    if (ttsWebSocket) {
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
  const showStopButton = isRecording || isSpeaking;

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
