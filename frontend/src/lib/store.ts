// frontend/src/lib/store.ts
import { create } from 'zustand';

export enum AppState {
  IDLE = 'idle',
  LISTENING = 'listening',
  THINKING = 'thinking',
  SPEAKING = 'speaking',
}

interface AppStoreState {
  appState: AppState;
  userTranscript: string;
  assistantResponse: string;
  isRecording: boolean;
  errorMessage: string;
  audioPlayer: HTMLAudioElement | null; // Store the player instance

  // Actions
  setAppState: (state: AppState) => void;
  setUserTranscript: (text: string) => void;
  setAssistantResponse: (text: string) => void;
  setIsRecording: (recording: boolean) => void;
  setErrorMessage: (message: string) => void;
  setAudioPlayer: (player: HTMLAudioElement | null) => void;
  resetConversationState: () => void;
  setError: (message: string, revertToState?: AppState) => void;
}

export const useAppStore = create<AppStoreState>((set, get) => ({
  // Initial state
  appState: AppState.IDLE,
  userTranscript: '',
  assistantResponse: '',
  isRecording: false,
  errorMessage: '',
  audioPlayer: null,

  // Actions to update state
  setAppState: (state) => set({ appState: state }),
  setUserTranscript: (text) => set({ userTranscript: text }),
  setAssistantResponse: (text) => set({ assistantResponse: text }),
  setIsRecording: (recording) => set({ isRecording: recording }),
  setErrorMessage: (message) => set({ errorMessage: message }),
  setAudioPlayer: (player) => set({ audioPlayer: player }),

  resetConversationState: () => {
    // Stop any ongoing playback before resetting
    const player = get().audioPlayer;
    if (player) {
        player.pause();
        if (player.src && player.src.startsWith('blob:')) {
             URL.revokeObjectURL(player.src); // Clean up blob URL
        }
        player.src = ''; // Clear source
    }
    set({
      userTranscript: '',
      assistantResponse: '',
      errorMessage: '',
      appState: AppState.IDLE,
      isRecording: false,
      audioPlayer: null,
    });
  },

  setError: (message, revertToState = AppState.IDLE) => {
    set({
      errorMessage: message,
      appState: revertToState,
      isRecording: false, // Ensure recording stops on error
      // Consider clearing other states if needed
      // userTranscript: '',
      // assistantResponse: '',
    });
  },
}));