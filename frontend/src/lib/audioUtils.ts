// frontend/src/lib/audioUtils.ts
import { useAppStore, AppState } from './store'; // Import Zustand store

let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let audioStream: MediaStream | null = null;

// Get actions from the store instance
const { setIsRecording, setError, setAppState } = useAppStore.getState();

/**
 * Requests microphone access and starts recording.
 * @returns True if recording started successfully, false otherwise.
 */
export async function startRecording(): Promise<boolean> {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log("Microphone access granted.");

      const options = getSupportedMimeTypeOptions();
      console.log("Using recorder options:", options);

      mediaRecorder = new MediaRecorder(audioStream, options);
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        console.log("MediaRecorder stopped.");
        setIsRecording(false);
        // stopMediaStream(); // Stop stream in stopRecording logic after processing
      };

      mediaRecorder.onerror = (event: Event) => {
        // Type cast event to MediaRecorderErrorEvent if needed, or access error directly
        const error = (event as any).error as DOMException; // Basic type assertion
        console.error("MediaRecorder error:", error);
        setError(`Microphone recording error: ${error?.name} - ${error?.message}`);
        stopMediaStream();
        setIsRecording(false);
        setAppState(AppState.IDLE);
      };

      mediaRecorder.start();
      console.log("MediaRecorder started.");
      setIsRecording(true);
      setAppState(AppState.LISTENING);
      return true;

    } catch (err: any) { // Catch as 'any' or 'unknown'
      console.error("Error accessing microphone:", err);
      let message = "Could not access microphone.";
       if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
         message = "Microphone permission denied. Please allow access.";
       } else if (err?.name === 'NotFoundError' || err?.name === 'DevicesNotFoundError') {
         message = "No microphone found.";
       } else {
         message = `Microphone access error: ${err?.name}`;
       }
      setError(message);
      setIsRecording(false);
      setAppState(AppState.IDLE);
      return false;
    }
  } else {
    setError("Your browser does not support microphone access (getUserMedia).");
    setIsRecording(false);
    setAppState(AppState.IDLE);
    return false;
  }
}

/**
 * Stops the current recording and returns the audio data as a Blob.
 * @returns The recorded audio Blob or null if not recording or error.
 */
export function stopRecording(): Promise<Blob | null> {
  return new Promise((resolve) => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      const currentAppState = useAppStore.getState().appState; // Get current state

      mediaRecorder.onstop = () => {
        console.log("MediaRecorder stopped callback, processing chunks...");
        setIsRecording(false);
        stopMediaStream(); // Stop the track now

        if (audioChunks.length === 0) {
          console.warn("No audio chunks recorded.");
          setError("No audio was recorded.");
          resolve(null);
          return;
        }

        const mimeType = mediaRecorder?.mimeType || 'audio/wav';
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        console.log(`Created audio Blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);
        audioChunks = [];
        resolve(audioBlob);
      };

      mediaRecorder.stop();
      console.log("Requested MediaRecorder stop.");
      // Transition state only if currently listening
      if (currentAppState === AppState.LISTENING) {
        setAppState(AppState.THINKING);
      }

    } else {
      console.warn("Stop recording called but not currently recording.");
      setIsRecording(false); // Ensure state consistency
      // Resolve with null if chunks are empty or state is weird
      resolve(null);
    }
  });
}

/** Stops the microphone audio stream tracks. */
function stopMediaStream() {
  if (audioStream) {
    audioStream.getTracks().forEach(track => {
      track.stop();
      console.log("Audio track stopped.");
    });
    audioStream = null;
  }
}

/** Checks for supported MIME types for MediaRecorder. */
function getSupportedMimeTypeOptions(): MediaRecorderOptions {
    const types = ['audio/wav', 'audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/ogg'];
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) {
            console.log(`Using MIME type: ${type}`);
            return { mimeType: type };
        }
    }
    console.warn("No preferred MIME type supported, using browser default.");
    return {};
}