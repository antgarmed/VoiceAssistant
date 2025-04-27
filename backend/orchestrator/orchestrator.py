# backend/orchestrator/orchestrator.py

import os
import logging
import time
import yaml
import base64
import asyncio
import json # For websocket messages
import uuid # For unique utterance IDs
from contextlib import asynccontextmanager
from collections import deque
from urllib.parse import urlparse, urlunparse

import httpx
import websockets # Keep import for call_tts_service_ws
from fastapi import FastAPI, HTTPException, UploadFile, File, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Tuple

# --- Constants & Environment Loading ---
load_dotenv()

CONFIG_PATH = os.getenv('CONFIG_PATH', '/app/config.yaml')
LOG_FILE_BASE = os.getenv('LOG_FILE_BASE', '/app/logs/service')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

_ASR_BASE_URL_ENV = os.getenv('ASR_SERVICE_URL', f"http://asr:{os.getenv('ASR_PORT', 5001)}")
_LLM_BASE_URL_ENV = os.getenv('LLM_SERVICE_URL', f"http://llm:{os.getenv('LLM_PORT', 5002)}")
_TTS_BASE_URL_ENV = os.getenv('TTS_SERVICE_URL', f"http://tts:{os.getenv('TTS_PORT', 5003)}")

SERVICE_NAME = "orchestrator"
LOG_PATH = f"{LOG_FILE_BASE}_{SERVICE_NAME}.log"

# --- Logging Setup ---
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)
logger = logging.getLogger(SERVICE_NAME)

# --- Global Variables & State ---
orchestrator_config: Dict[str, Any] = {}
api_endpoints: Dict[str, str] = {}
conversation_history: deque = deque(maxlen=10)
http_client: Optional[httpx.AsyncClient] = None

# --- Configuration Loading ---
def load_configuration():
    """Loads Orchestrator and API endpoint settings from YAML."""
    global orchestrator_config, api_endpoints, conversation_history
    default_api_endpoints = {
        'asr': f"{_ASR_BASE_URL_ENV}/transcribe",
        'llm': f"{_LLM_BASE_URL_ENV}/generate",
        'tts_ws': f"ws://{urlparse(_TTS_BASE_URL_ENV).netloc}/synthesize_stream" # Keep for call_tts_service_ws
    }
    try:
        logger.info(f"Loading configuration from: {CONFIG_PATH}")
        # ... (rest of config loading is fine) ...
        if not os.path.exists(CONFIG_PATH):
            logger.warning(f"Config file not found at {CONFIG_PATH}. Using defaults.")
            config = {}
        else:
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
            if not config:
                 logger.warning("Config file is empty or invalid YAML. Using defaults.")
                 config = {}

        orchestrator_config = config.get('orchestrator', {})
        api_endpoints_cfg = config.get('api_endpoints', {})
        api_endpoints['asr'] = api_endpoints_cfg.get('asr', default_api_endpoints['asr'])
        api_endpoints['llm'] = api_endpoints_cfg.get('llm', default_api_endpoints['llm'])
        api_endpoints['tts_ws'] = api_endpoints_cfg.get('tts_ws', default_api_endpoints['tts_ws']) # Resolve TTS WS URL
        
        logger.info(f"ASR Endpoint: {api_endpoints['asr']}")
        logger.info(f"LLM Endpoint: {api_endpoints['llm']}")
        logger.info(f"TTS WebSocket Endpoint Configured (for no-speech handling): {api_endpoints['tts_ws']}") 
        max_hist = orchestrator_config.get('max_history_turns', 5) * 2
        if max_hist <= 0: max_hist = 2
        conversation_history = deque(maxlen=max_hist)
        logger.info(f"Conversation history max length set to {max_hist} messages ({max_hist // 2} turns).")
        if not orchestrator_config.get('system_prompt'):
            logger.warning("System prompt not found in config, using default.")
            orchestrator_config['system_prompt'] = "You are a helpful voice assistant."
        else:
            logger.info("Loaded system prompt from config.")
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}. Using defaults.", exc_info=True)
        api_endpoints = default_api_endpoints
        orchestrator_config = {'max_history_turns': 5, 'system_prompt': 'You are a helpful voice assistant.'}
        conversation_history = deque(maxlen=10)

# --- Backend Service Interaction Helpers ---
async def call_asr_service(audio_file: UploadFile) -> str:
    # ... (Keep existing implementation) ...
    if not http_client: raise RuntimeError("HTTP client not initialized")
    files = {'audio': (audio_file.filename, await audio_file.read(), audio_file.content_type)}
    request_url = api_endpoints['asr']
    logger.info(f"Sending audio to ASR service at {request_url}")
    try:
        response = await http_client.post(request_url, files=files, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        transcript = data.get('text', '').strip()
        if not transcript: logger.warning("ASR returned an empty transcript."); return ""
        logger.info(f"ASR Response: '{transcript[:100]}...'")
        return transcript
    except httpx.TimeoutException: logger.error(f"ASR request timed out to {request_url}"); raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="ASR service request timed out.")
    except httpx.RequestError as e: logger.error(f"ASR request error to {request_url}: {e}"); raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"ASR service request failed: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"ASR service error: Status {e.response.status_code}, Response: {e.response.text[:500]}")
        detail = f"ASR service error ({e.response.status_code})"; backend_detail=None
        try: backend_detail = e.response.json().get("detail")
        except Exception: pass
        if backend_detail: detail += f": {backend_detail}"
        status_code = e.response.status_code if e.response.status_code >= 500 else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e: logger.error(f"Unexpected error calling ASR: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error communicating with ASR: {e}")


async def call_llm_service(history: List[Dict[str, str]]) -> str:
    # ... (Keep existing implementation) ...
    if not http_client: raise RuntimeError("HTTP client not initialized")
    request_url = api_endpoints['llm']
    payload = {"messages": history}
    logger.info(f"Sending {len(history)} messages to LLM service at {request_url}")
    logger.debug(f"LLM Payload: {payload}")
    try:
        response = await http_client.post(request_url, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        assistant_response = data.get('content', '').strip()
        if not assistant_response: logger.warning("LLM returned an empty response."); return "Sorry, I seem to be speechless right now."
        logger.info(f"LLM Response: '{assistant_response[:100]}...'")
        return assistant_response
    except httpx.TimeoutException: logger.error(f"LLM request timed out to {request_url}"); raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="LLM service request timed out.")
    except httpx.RequestError as e: logger.error(f"LLM request error to {request_url}: {e}"); raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM service request failed: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM service error: Status {e.response.status_code}, Response: {e.response.text[:500]}")
        detail = f"LLM service error ({e.response.status_code})"; backend_detail=None
        try: backend_detail = e.response.json().get("detail")
        except Exception: pass
        if backend_detail: detail += f": {backend_detail}"
        status_code = e.response.status_code if e.response.status_code >= 500 else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e: logger.error(f"Unexpected error calling LLM: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error communicating with LLM: {e}")


# --- This function is now ONLY used for the no-speech case ---
async def call_tts_service_ws(text: str, max_audio_length_ms: Optional[float] = None) -> bytes:
    """Calls the TTS service WebSocket endpoint to synthesize audio stream.
       ONLY intended for short, fixed responses like the no-speech case."""
    # --- Add specific warning ---
    logger.warning("Orchestrator making direct TTS call (expected only for no-speech handling).")
    # --- End warning ---
    ws_url = api_endpoints['tts_ws']
    utterance_id = str(uuid.uuid4()) # Still generate unique ID
    logger.info(f"[No-Speech TTS] Connecting to TTS WebSocket at {ws_url} for utterance_id: {utterance_id}")

    message_payload = {
        "type": "generate_chunk",
        "text_chunk": text,
        "utterance_id": utterance_id
    }
    if max_audio_length_ms is not None:
        message_payload["max_audio_length_ms"] = max_audio_length_ms
        logger.info(f"[No-Speech TTS] Requesting max audio length: {max_audio_length_ms}ms for {utterance_id}")
    else:
         logger.info(f"[No-Speech TTS] Sending text: '{text[:50]}...' for {utterance_id}")

    all_audio_bytes = bytearray()
    receive_timeout_seconds = 30.0 # Shorter timeout for no-speech
    websocket_connection = None

    try:
        # Use slightly shorter timeouts for this specific, short call
        async with websockets.connect(
            ws_url, open_timeout=15, close_timeout=10, ping_interval=None # No ping needed
        ) as websocket:
            websocket_connection = websocket
            logger.info(f"[No-Speech TTS] WebSocket connection established for {utterance_id}.")
            await websocket.send(json.dumps(message_payload))
            logger.info(f"[No-Speech TTS] Sent 'generate_chunk' request for {utterance_id}.")

            last_message_time = time.monotonic()
            while True:
                try:
                    wait_time = max(0, receive_timeout_seconds - (time.monotonic() - last_message_time))
                    if wait_time <= 0:
                        if all_audio_bytes: logger.warning(f"[No-Speech TTS] WS receive timed out (inactivity). Assuming stream ended."); break
                        else: logger.error(f"[No-Speech TTS] WS receive timed out (inactivity) with NO audio."); raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="TTS service timed out (no-speech).")

                    message_json = await asyncio.wait_for(websocket.recv(), timeout=wait_time)
                    last_message_time = time.monotonic()
                    message = json.loads(message_json)
                    msg_type = message.get("type")

                    if msg_type == "audio_chunk":
                        audio_b64 = message.get("audio_b64", "")
                        if audio_b64:
                            try: all_audio_bytes.extend(base64.b64decode(audio_b64))
                            except Exception as decode_err: logger.warning(f"[No-Speech TTS] Failed to decode chunk: {decode_err}")
                    elif msg_type == "error":
                        error_msg = message.get("message", "Unknown TTS error")
                        logger.error(f"[No-Speech TTS] Received error from TTS WS: {error_msg}"); raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"TTS service error: {error_msg}")
                    elif msg_type == "stream_end": logger.info(f"[No-Speech TTS] Received explicit 'stream_end'."); break
                    # Ignore context_cleared for this specific call
                    elif msg_type != "context_cleared": logger.warning(f"[No-Speech TTS] Received unknown message type '{msg_type}'.")

                except asyncio.TimeoutError:
                    if all_audio_bytes: logger.warning(f"[No-Speech TTS] WS receive timed out (asyncio). Assuming stream ended."); break
                    else: logger.error(f"[No-Speech TTS] WS receive timed out (asyncio) with NO audio."); raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="TTS service timed out (no-speech).")
                except websockets.exceptions.ConnectionClosedOK: logger.info(f"[No-Speech TTS] WS closed normally by server."); break
                except websockets.exceptions.ConnectionClosedError as e: logger.error(f"[No-Speech TTS] WS closed with error: {e}"); break # Break if we got some audio

        logger.info(f"[No-Speech TTS] Processing complete. Bytes: {len(all_audio_bytes)}.")
        # No context clear needed here as it's a one-off call
        return bytes(all_audio_bytes)

    # Keep specific exception handling for this call
    except websockets.exceptions.InvalidURI as e: logger.error(f"Invalid TTS WS URI: {ws_url}. Error: {e}"); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Invalid TTS WS URI configured: {ws_url}")
    except websockets.exceptions.WebSocketException as e: logger.error(f"TTS WS connection failed: {e}"); raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Failed to connect to TTS service WS: {e}")
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.error(f"Unexpected error in call_tts_service_ws: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error during TTS call: {e}")


# --- Conversation History Management ---
def update_history(user_text: str, assistant_text: str):
    if conversation_history.maxlen is None or conversation_history.maxlen <= 0: logger.warning("Conv history maxlen invalid."); return
    if user_text: conversation_history.append({"role": "user", "content": user_text})
    if assistant_text: conversation_history.append({"role": "assistant", "content": assistant_text})
    logger.debug(f"History updated. Len: {len(conversation_history)}/{conversation_history.maxlen}")

def get_formatted_history() -> List[Dict[str, str]]:
    system_prompt = orchestrator_config.get('system_prompt', '')
    history = []
    if system_prompt: history.append({"role": "system", "content": system_prompt})
    history.extend(list(conversation_history))
    return history

# --- FastAPI Lifespan & App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{SERVICE_NAME.upper()} Service starting up...")
    load_configuration()
    global http_client
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    http_client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True)
    logger.info("HTTP client initialized.")
    # Removed sleep, relying on docker-compose depends_on: condition: service_healthy
    # await asyncio.sleep(5)
    await check_backend_services() # Run checks after client init
    yield
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down...")
    if http_client: await http_client.aclose(); logger.info("HTTP client closed.")
    logger.info("Orchestrator Service shutdown complete.")

# --- check_backend_services includes retries ---
async def check_backend_services():
    if not http_client: return
    # Check ASR, LLM, and TTS HTTP health endpoints
    services_to_check = {
        "ASR": api_endpoints['asr'].replace('/transcribe', '/health'),
        "LLM": api_endpoints['llm'].replace('/generate', '/health'),
        "TTS_HTTP": urlunparse(('http', urlparse(api_endpoints['tts_ws']).netloc, '/health', '', '', '')),
    }
    logger.info("Checking backend service connectivity (with retries)...")
    all_services_ok = True # Track overall status
    max_retries = 3
    delay = 2.0

    for name, url in services_to_check.items():
        service_ok = False # Track status for this specific service
        for attempt in range(max_retries):
            logger.info(f"Checking {name} at {url} (Attempt {attempt+1}/{max_retries})...")
            try:
                response = await http_client.get(url, timeout=5.0)
                if response.status_code < 400:
                    logger.info(f"Backend service {name} health check successful (Status {response.status_code}).")
                    service_ok = True
                    break # Success for this service, move to next service
                else:
                    logger.warning(f"Backend service {name} health check attempt {attempt+1}/{max_retries} failed: Status {response.status_code}. URL: {url}. Response: {response.text[:200]}")
                    if response.status_code == 503: logger.warning(f" -> {name} service might still be loading/initializing.")
                    # Continue retrying
            except httpx.RequestError as e:
                logger.error(f"Failed to connect to backend service {name} (Attempt {attempt+1}/{max_retries}) at {url}: {e}")
                # Continue retrying
            except Exception as e:
                logger.error(f"Unexpected error during {name} health check (Attempt {attempt+1}/{max_retries}) at {url}: {e}", exc_info=True)
                service_ok = False # Mark as failed on unexpected error
                break # Stop retrying for this service on unexpected error

            if not service_ok and attempt + 1 < max_retries:
                 logger.info(f"Waiting {delay}s before retrying {name}...")
                 await asyncio.sleep(delay)
        # After retries for a specific service
        if not service_ok:
            logger.error(f"Backend service {name} failed health check after {max_retries} attempts.")
            all_services_ok = False # Mark overall failure if any service fails

    # Final overall status log
    if not all_services_ok:
        logger.error("One or more critical backend services could not be reached or failed health check during startup.")
    else:
        logger.info("Initial backend service connectivity and health checks passed.")


# --- FastAPI App Creation and CORS ---
app = FastAPI(lifespan=lifespan, title="Voice Assistant Orchestrator", version="1.1.0")
origins = ["*"] # Allow all origins for simplicity in local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API ---
class AssistResponse(BaseModel):
    user_transcript: str
    assistant_response: str
    assistant_audio_b64: str # Kept for API schema consistency, will be empty
    response_time_ms: float

# --- API Endpoints ---
@app.post("/assist", response_model=AssistResponse)
async def handle_assist_request(audio: UploadFile = File(..., description="User audio input (WAV, MP3, etc.)")):
    """
    Handles voice input, gets transcription and LLM response.
    Audio synthesis is now handled by the frontend connecting directly to TTS.
    """
    overall_start_time = time.monotonic()
    logger.info(f"Received /assist request for file: {audio.filename}, size: {audio.size}")

    try:
        # 1. Call ASR Service
        asr_start_time = time.monotonic()
        user_transcript = await call_asr_service(audio)
        asr_time = (time.monotonic() - asr_start_time) * 1000

        # Handle no speech (Calls deprecated local TTS function)
        if not user_transcript:
            logger.info("ASR returned no transcript. Generating no-speech response.")
            no_speech_response = "Sorry, I didn't hear anything."
            tts_no_speech_start_time = time.monotonic()
            try:
                 # Call the specific function for this case
                 no_speech_audio_bytes = await call_tts_service_ws(no_speech_response)
                 tts_no_speech_time = (time.monotonic() - tts_no_speech_start_time) * 1000
                 logger.info(f"No-speech TTS generation took {tts_no_speech_time:.0f}ms")
                 no_speech_audio_b64 = base64.b64encode(no_speech_audio_bytes).decode('utf-8') if no_speech_audio_bytes else ""
            except Exception as no_speech_err:
                 logger.error(f"Failed to generate no-speech audio via TTS: {no_speech_err}", exc_info=True)
                 no_speech_audio_b64 = "" # Fallback to empty audio on error

            return AssistResponse(
                user_transcript="",
                assistant_response=no_speech_response,
                assistant_audio_b64=no_speech_audio_b64, # May contain short audio or be empty
                response_time_ms=(time.monotonic() - overall_start_time) * 1000
            )

        # 2. Prepare history for LLM
        current_llm_input = get_formatted_history()
        current_llm_input.append({"role": "user", "content": user_transcript})

        # 3. Call LLM Service
        llm_start_time = time.monotonic()
        assistant_response = await call_llm_service(current_llm_input)
        llm_time = (time.monotonic() - llm_start_time) * 1000

        # --- Step 4 REMOVED: No direct TTS call from orchestrator for normal responses ---

        # 5. Update History (Still relevant)
        update_history(user_text=user_transcript, assistant_text=assistant_response)

        # --- Step 6 REMOVED: No audio encoding needed here ---
        assistant_audio_b64 = "" # Explicitly set to empty string

        # 7. Log and return (Adjust log message)
        overall_time = (time.monotonic() - overall_start_time) * 1000
        logger.info(f"Assist request (ASR+LLM only) processed in {overall_time:.2f}ms (ASR: {asr_time:.0f}ms, LLM: {llm_time:.0f}ms)")

        # Return response without audio data
        return AssistResponse(
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            assistant_audio_b64=assistant_audio_b64, # Will be empty
            response_time_ms=overall_time
        )

    except HTTPException as http_exc:
         logger.error(f"Pipeline failed due to HTTPException: {http_exc.status_code} - {http_exc.detail}")
         # Log the detail coming from downstream services if available
         if http_exc.detail: logger.error(f" -> Detail: {http_exc.detail}")
         raise http_exc # Re-raise the exception to return proper HTTP error
    except Exception as e:
        logger.error(f"Unexpected error during /assist pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal orchestration error occurred: {e}"
        )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    # Basic health check only confirms the orchestrator itself is running
    return {"service": SERVICE_NAME, "status": "ok", "details": "Orchestrator is running."}

# --- Reset History Endpoint ---
@app.post("/reset_history", status_code=status.HTTP_200_OK)
async def reset_conversation_history():
    global conversation_history
    try:
        max_hist = orchestrator_config.get('max_history_turns', 5) * 2
        if max_hist <= 0: max_hist = 2
        conversation_history = deque(maxlen=max_hist)
        logger.info(f"Conversation history reset via API (maxlen={max_hist}).")
        return {"message": "Conversation history cleared."}
    except Exception as e:
         logger.error(f"Error during history reset: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reset conversation history.")


# --- Main Execution Guard ---
# ... (Keep existing __main__ block as is) ...
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {SERVICE_NAME.upper()} service directly via __main__...")
    load_configuration()
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    http_client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True)
    port = int(os.getenv('ORCHESTRATOR_PORT', 5000))
    log_level_param = LOG_LEVEL.lower()
    logger.info(f"Launching Uvicorn on host 0.0.0.0, port {port} with log level {log_level_param}...")
    try:
        # Manual checks are less critical with depends_on, but can be run for direct execution testing
        # async def run_checks():
        #      await asyncio.sleep(15) 
        #      await check_backend_services()
        # # asyncio.run(run_checks())

        uvicorn.run("orchestrator:app", host="0.0.0.0", port=port, log_level=log_level_param, reload=False)
    except Exception as main_err:
         logger.critical(f"Failed to start Uvicorn directly: {main_err}", exc_info=True)
         exit(1)
    finally:
        if http_client and not http_client.is_closed:
            asyncio.run(http_client.aclose())
            logger.info(f"{SERVICE_NAME.upper()} HTTP client closed (direct run shutdown).")
        logger.info(f"{SERVICE_NAME.upper()} Service shutting down (direct run)...")

# --- END OF FILE ---
