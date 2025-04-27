// frontend/src/lib/api.ts
import { useAppStore } from './store';
const ORCHESTRATOR_BASE_URL = 'http://localhost:5000';
const WS_TTS_URL = 'ws://localhost:5003/synthesize_stream';

const { setError } = useAppStore.getState();

/**
 * Send recorded audio to orchestrator /assist endpoint.
 * Expects only text response (no audio).
 */
export async function sendAudioToBackend(audioBlob: Blob): Promise<any | null> {
  const url = `${ORCHESTRATOR_BASE_URL}/assist`;
  console.log(`Sending audio (${(audioBlob.size / 1024).toFixed(2)} KB) to ${url}`);
  const formData = new FormData();
  formData.append('audio', audioBlob, `recording-${Date.now()}.wav`);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000);

  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    const contentType = response.headers.get('content-type');
    let responseData: any;

    if (contentType?.includes('application/json')) {
      responseData = await response.json();
    } else {
      responseData = await response.text();
    }

    if (!response.ok) {
      throw new Error(`[${response.status}] ${responseData?.detail || 'Unexpected error'}`);
    }

    if (typeof responseData !== 'object' || !responseData.assistant_response) {
      console.error('Invalid orchestrator response:', responseData);
      throw new Error('Invalid orchestrator response.');
    }

    console.log('Received assistant response from orchestrator.');
    return {
      user_transcript: responseData.user_transcript,
      assistant_response: responseData.assistant_response,
    };
  } catch (error: any) {
    clearTimeout(timeoutId);
    console.error('sendAudioToBackend error:', error);
    let friendlyMessage = "Assistant communication failed.";
    if (error.name === 'AbortError') friendlyMessage = "Assistant timeout.";
    setError(friendlyMessage);
    return null;
  }
}

/**
 * Create WebSocket connection to TTS service for streaming.
 */
export function createTTSWebSocket(
  utteranceId: string,
  text: string,
  onChunk: (audioChunk: Uint8Array) => void,
  onEnd: () => void,
  onError: (error: any) => void
): WebSocket {
  const ws = new WebSocket(WS_TTS_URL);
  console.log(`Opening TTS WebSocket to ${WS_TTS_URL}`);

  ws.onopen = () => {
    console.log(`WebSocket opened. Sending generate_chunk.`);
    ws.send(JSON.stringify({
      type: 'generate_chunk',
      text_chunk: text,
      utterance_id: utteranceId,
    }));
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'audio_chunk') {
        const raw = Uint8Array.from(atob(msg.audio_b64), c => c.charCodeAt(0));
        onChunk(raw);
      } else if (msg.type === 'stream_end') {
        console.log('Stream end received.');
        onEnd();
      } else if (msg.type === 'error') {
        console.error('TTS Error:', msg.message);
        onError(new Error(msg.message));
      }
    } catch (e) {
      console.error('TTS message parse error:', event.data, e);
      onError(new Error('Invalid TTS WebSocket message.'));
    }
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    onError(new Error('WebSocket connection error'));
  };

  ws.onclose = (e) => {
    console.log(`WebSocket closed: code=${e.code}, reason=${e.reason}`);
  };

  return ws;
}

/**
 * Reset conversation history on backend.
 */
export async function resetBackendHistory(): Promise<boolean> {
  const url = `${ORCHESTRATOR_BASE_URL}/reset_history`;
  console.log(`Resetting backend history.`);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10_000);

  try {
    const response = await fetch(url, {
      method: 'POST',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Failed to reset: ${response.status}`);
    }

    console.log('Backend history reset successful.');
    return true;
  } catch (error: any) {
    clearTimeout(timeoutId);
    console.error('resetBackendHistory error:', error);
    setError('Failed to reset history.');
    return false;
  }
}
