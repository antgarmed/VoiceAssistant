// frontend/src/lib/api.ts
// REMOVED: import { fetch as tauriFetch, Body } from '@tauri-apps/api/http'; // No longer needed for external calls with assetProtocol
import { Buffer } from 'buffer';
import { useAppStore } from './store'; // Import Zustand store hook/actions

const ORCHESTRATOR_BASE_URL = 'http://localhost:5000'; // Keep consistent

// Get the setError action from the store
const { setError } = useAppStore.getState();

/**
 * Sends audio blob to the orchestrator backend /assist endpoint using standard fetch.
 * @param audioBlob The recorded audio data.
 * @returns The response object from the backend or null on error.
 */
export async function sendAudioToBackend(audioBlob: Blob): Promise<any | null> {
  const url = `${ORCHESTRATOR_BASE_URL}/assist`;
  console.log(`Sending audio (${(audioBlob.size / 1024).toFixed(2)} KB) to ${url}`);

  // Use FormData for standard fetch with file uploads
  const formData = new FormData();
  // The backend likely expects the file under the 'audio' field name
  formData.append('audio', audioBlob, `recording-${Date.now()}.wav`);

  // AbortController for timeout (optional but good practice)
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120 * 1000); // 120 seconds timeout

  try {
    // Use standard window.fetch
    const response = await fetch(url, {
      method: 'POST',
      body: formData, // Send FormData directly
      signal: controller.signal, // Attach AbortController signal
      // No need to set Content-Type for FormData, fetch does it automatically
    });

    clearTimeout(timeoutId); // Clear the timeout if fetch completes

    console.log('Backend Response Status:', response.status);
    console.log('Backend Response OK:', response.ok);

    // Get response body for error details or success data
    let responseData: any;
    try {
        // Attempt to parse as JSON, fall back to text if needed
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
            responseData = await response.json();
        } else {
            responseData = await response.text(); // Get text for non-JSON errors
        }
    } catch (parseError) {
        console.error("Could not parse response body:", parseError);
        responseData = await response.text(); // Attempt to get text on JSON parse error
        if (!response.ok) {
             throw new Error(`Backend returned status ${response.status}. Could not parse response body.`);
        }
        // If response was OK but couldn't parse, maybe it was empty? Log and continue if appropriate.
        console.warn("Response OK, but failed to parse body. Assuming empty response.");
        responseData = {}; // Or null/undefined depending on expected success response
    }


    if (!response.ok) {
      let errorDetail = `Backend returned status ${response.status}`;
      if (typeof responseData === 'object' && responseData?.detail) {
          errorDetail = `[${response.status}] ${responseData.detail}`;
      } else if (typeof responseData === 'string' && responseData.length > 0) {
          errorDetail += `: ${responseData.substring(0, 100)}`;
      }
      throw new Error(errorDetail);
    }

    if (typeof responseData !== 'object' || responseData === null) {
      console.error("Backend response data is not a valid object:", responseData);
      throw new Error("Received invalid response format from backend.");
    }

    console.log('Received data from backend.');
    return responseData; // Return the parsed data

  } catch (error: any) { // Catch as 'any' or 'unknown' and type check
     clearTimeout(timeoutId); // Ensure timeout is cleared on error too
    console.error('Error sending audio to backend:', error);
    let friendlyMessage = "Failed to communicate with the assistant.";

    if (error.name === 'AbortError') {
        friendlyMessage = "The request timed out. The assistant might be busy.";
    } else if (error?.message) {
       // Keep existing error message checks, but they now apply to standard Fetch errors
       if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
         friendlyMessage = "Network error. Is the backend running?";
       } else if (error?.response?.status === 503 || error?.message?.includes("[503]")) { // Might need adjustment based on standard fetch error structure
         friendlyMessage = "Assistant service is unavailable. It might be starting up or encountered an error.";
       } else if (error.message.includes("Backend returned status") || error.message.includes("[")) { // Catch formatted errors too
         friendlyMessage = `Assistant error: ${error.message}`;
       } else if (error.message.length < 100 && !error.message.includes("signal is aborted")) { // Avoid overriding timeout message
           friendlyMessage = error.message;
       }
    }
    setError(friendlyMessage); // Use Zustand action
    return null;
  }
}

/**
 * Decodes Base64 audio data and returns an Audio object.
 * (This function doesn't use fetch, no changes needed here)
 * @param audioB64 Base64 encoded WAV audio string.
 * @returns Audio object or null on error.
 */
export function createAudioPlayer(audioB64: string): HTMLAudioElement | null {
    // ... (keep existing implementation)
    if (!audioB64) {
        console.warn('No audio data received to play.');
        return null;
    }
    try {
        const audioBytes = Buffer.from(audioB64, 'base64');
        const audioBlob = new Blob([audioBytes], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        // Cleanup handled in the component using the player
        return audio;
    } catch (error) {
        console.error('Error decoding or creating audio player:', error);
        setError("Failed to prepare audio for playback."); // Use Zustand action
        return null;
    }
}

/**
 * Calls the orchestrator endpoint to reset conversation history using standard fetch.
 */
export async function resetBackendHistory(): Promise<boolean> {
    const url = `${ORCHESTRATOR_BASE_URL}/reset_history`;
    console.log(`Sending request to reset history at ${url}`);

    // AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10 * 1000); // 10 seconds timeout

    try {
        // Use standard window.fetch
        const response = await fetch(url, {
            method: 'POST',
            signal: controller.signal,
        });
        clearTimeout(timeoutId); // Clear timeout

        if (!response.ok) {
            // Attempt to get error details from body
            let errorDetail = `Backend returned status ${response.status}`;
            try {
                 const errorText = await response.text();
                 if (errorText) {
                    errorDetail += `: ${errorText.substring(0,100)}`;
                 }
            } catch (_) { /* Ignore parse error */ }
            throw new Error(errorDetail);
        }
        console.log("Backend history reset successfully.");
        return true;
    } catch (error: any) {
        clearTimeout(timeoutId); // Clear timeout on error
        console.error("Error resetting backend history:", error);
        let friendlyMessage = "Failed to reset conversation history on the backend.";
        if (error.name === 'AbortError') {
           friendlyMessage = "Request to reset history timed out.";
        } else if (error?.message) {
           friendlyMessage = error.message;
        }
        setError(friendlyMessage); // Use Zustand action
        return false;
    }
}