// frontend\src\lib\audioUtils.ts
import { useAppStore, AppState } from './store';

let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let audioStream: MediaStream | null = null;

// Streaming playback variables
let audioCtx: AudioContext | null = null;
let audioQueue: AudioBuffer[] = [];
let isPlayingStream = false;
let scheduledEndTime = 0;
const PLAYBACK_START_BUFFER_SECONDS = 0.5;
const MIN_CHUNK_DURATION_TO_SCHEDULE = 0.05;

const { setIsRecording, setError, setAppState } = useAppStore.getState();

// === Recording Functions ===

export async function startRecording(): Promise<boolean> {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log("Microphone access granted.");

      const options = getSupportedMimeTypeOptions();
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
      };

      mediaRecorder.onerror = (event: Event) => {
        const error = (event as any).error as DOMException;
        console.error("MediaRecorder error:", error);
        setError(`Microphone recording error: ${error?.name} - ${error?.message}`);
        cleanupRecordingResources();
        setAppState(AppState.IDLE);
      };

      mediaRecorder.start();
      console.log("MediaRecorder started.");
      setIsRecording(true);
      setAppState(AppState.LISTENING);
      return true;

    } catch (err: any) {
      console.error("Error accessing microphone:", err);
      setError("Microphone access error.");
      cleanupRecordingResources();
      setAppState(AppState.IDLE);
      return false;
    }
  } else {
    setError("Browser does not support getUserMedia.");
    setIsRecording(false);
    setAppState(AppState.IDLE);
    return false;
  }
}

export function stopRecording(): Promise<Blob | null> {
  return new Promise((resolve) => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.onstop = () => {
        console.log("Processing recorded chunks...");
        cleanupRecordingResources();

        if (audioChunks.length === 0) {
          resolve(null);
          return;
        }

        const mimeType = mediaRecorder?.mimeType ?? 'audio/wav';
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        audioChunks = [];
        resolve(audioBlob);
      };
      mediaRecorder.stop();
    } else {
      cleanupRecordingResources();
      resolve(null);
    }
  });
}

export function stopMediaStream(): boolean {
  let stopped = false;
  if (audioStream) {
    audioStream.getTracks().forEach(track => {
      track.stop();
      stopped = true;
    });
    audioStream = null;
  }
  mediaRecorder = null;
  if (useAppStore.getState().isRecording) {
    setIsRecording(false);
  }
  return stopped;
}

function cleanupRecordingResources(): boolean {
  mediaRecorder = null;
  return stopMediaStream();
}

function getSupportedMimeTypeOptions(): MediaRecorderOptions {
  const types = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/ogg'];
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return { mimeType: type };
    }
  }
  return {};
}

// === Streaming Playback Functions ===

export function isAudioPlaybackActive(): boolean {
  return isPlayingStream && audioQueue.length > 0;
}

function getAudioContext(): AudioContext | null {
  if (!audioCtx || audioCtx.state === "closed") {
    audioCtx = new AudioContext();
    audioQueue = [];
    isPlayingStream = false;
    scheduledEndTime = 0;
  }
  return audioCtx;
}

export async function handleIncomingAudioChunk(chunkBytes: Uint8Array) {
  const ctx = getAudioContext();
  if (!ctx) return;

  try {
    const decodedBuffer = await ctx.decodeAudioData(chunkBytes.buffer.slice(chunkBytes.byteOffset, chunkBytes.byteOffset + chunkBytes.byteLength));

    if (decodedBuffer.duration < MIN_CHUNK_DURATION_TO_SCHEDULE) {
      return;
    }

    audioQueue.push(decodedBuffer);
    console.log(`Added chunk to queue. Queue length: ${audioQueue.length}, chunk duration: ${decodedBuffer.duration.toFixed(2)}s`);

    if (!isPlayingStream && totalBufferedSeconds() >= PLAYBACK_START_BUFFER_SECONDS) {
      playNextChunkFromQueue();
    }
  } catch (err) {
    console.error("Audio chunk decode error:", err);
  }
}

function totalBufferedSeconds(): number {
  return audioQueue.reduce((sum, buf) => sum + buf.duration, 0);
}

function playNextChunkFromQueue() {
  const ctx = getAudioContext();
  if (!ctx || audioQueue.length === 0) {
    return;
  }

  if (!isPlayingStream) {
    isPlayingStream = true;
    setAppState(AppState.SPEAKING);
    scheduledEndTime = ctx.currentTime;
    console.log('Started audio playback stream');
  }

  const bufferToPlay = audioQueue[0];
  const startTime = Math.max(ctx.currentTime, scheduledEndTime);

  const source = ctx.createBufferSource();
  source.buffer = bufferToPlay;
  source.connect(ctx.destination);

  source.onended = () => {
    audioQueue.shift();
    console.log(`Buffer finished playing. Remaining buffers: ${audioQueue.length}`);
    
    if (audioQueue.length > 0) {
      // If there are more buffers, continue playing
      playNextChunkFromQueue();
    } else {
      // This is truly the end of playback - now we can clean up
      console.log('All buffers done—closing AudioContext');
      isPlayingStream = false;
      
      // This is the ONLY place we set IDLE after playback
      setAppState(AppState.IDLE);
      
      // Clean up audio context now that playback is truly complete
      if (audioCtx && audioCtx.state !== 'closed') {
        audioCtx.close().catch((e) => console.warn("Error closing AudioContext:", e));
        audioCtx = null;
      }
    }
  };

  source.start(startTime);
  scheduledEndTime = startTime + bufferToPlay.duration;
  
  // Log remaining audio length for debugging
  console.log(`Playing buffer: ${bufferToPlay.duration.toFixed(2)}s, remaining buffers: ${audioQueue.length-1}, total scheduled time: ${(scheduledEndTime - ctx.currentTime).toFixed(2)}s`);
}

export function cleanupAudio() {
  console.log(`Cleaning up audio. Queue length before cleanup: ${audioQueue?.length || 0}`);
  audioQueue = [];
  if (audioCtx) {
    if (audioCtx.state !== 'closed') {
      audioCtx.close().catch((e) => console.warn("Error closing AudioContext:", e));
    }
    audioCtx = null;
  }
  isPlayingStream = false;
  scheduledEndTime = 0;
  
  // REMOVED setAppState(AppState.IDLE) from here 
  // to have a single source of truth for state transitions
}
