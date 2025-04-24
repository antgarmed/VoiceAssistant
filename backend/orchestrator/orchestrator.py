# backend/orchestrator/orchestrator.py
# --- Added temporary max_audio_length_ms limit for testing ---
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
from urllib.parse import urlparse, urlunparse # Import URL parsing tools

import httpx
import websockets # Import websockets library
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
        'tts_ws': f"ws://{urlparse(_TTS_BASE_URL_ENV).netloc}/synthesize_stream"
    }
    try:
        logger.info(f"Loading configuration from: {CONFIG_PATH}")
        if not os.path.exists(CONFIG_PATH):
            # If config doesn't exist, log warning and use defaults
            logger.warning(f"Config file not found at {CONFIG_PATH}. Using defaults.")
            config = {} # Set empty config to proceed with defaults
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
        tts_ws_from_config = api_endpoints_cfg.get('tts_ws')
        if tts_ws_from_config:
            parsed_uri = urlparse(tts_ws_from_config)
            if parsed_uri.scheme not in ('ws', 'wss'):
                logger.warning(f"TTS WebSocket URI in config '{tts_ws_from_config}' has invalid scheme '{parsed_uri.scheme}'. Forcing 'ws://'.")
                api_endpoints['tts_ws'] = urlunparse(('ws', parsed_uri.netloc, parsed_uri.path, parsed_uri.params, parsed_uri.query, parsed_uri.fragment))
            else:
                api_endpoints['tts_ws'] = tts_ws_from_config
        else:
            api_endpoints['tts_ws'] = default_api_endpoints['tts_ws']
        logger.info(f"ASR Endpoint: {api_endpoints['asr']}")
        logger.info(f"LLM Endpoint: {api_endpoints['llm']}")
        logger.info(f"TTS WebSocket Endpoint: {api_endpoints['tts_ws']}")
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
    """Calls the ASR service to transcribe audio."""
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
    """Calls the LLM service to generate a response."""
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

# --- Corrected call_tts_service_ws ---
async def call_tts_service_ws(text: str, max_audio_length_ms: Optional[float] = None) -> bytes:
    """Calls the TTS service WebSocket endpoint to synthesize audio stream."""
    ws_url = api_endpoints['tts_ws']
    utterance_id = str(uuid.uuid4())
    logger.info(f"Connecting to TTS WebSocket at {ws_url} for utterance_id: {utterance_id}")
    # Construct the message payload including the optional max_audio_length_ms
    message_payload = {
        "type": "generate_chunk",
        "text_chunk": text,
        "utterance_id": utterance_id
    }
    # Only add max_audio_length_ms if provided (for temporary testing)
    # The TTS service itself should have its own default if not provided here
    if max_audio_length_ms is not None:
        message_payload["max_audio_length_ms"] = max_audio_length_ms
        logger.info(f"Requesting max audio length: {max_audio_length_ms}ms for {utterance_id}")
    else:
         logger.info(f"Sending text to TTS service via WS (no explicit max length): '{text[:50]}...' for {utterance_id}")

    all_audio_bytes = bytearray()
    receive_timeout_seconds = 45.0 # Default timeout - ADJUST IF NEEDED LATER

    try:
        async with websockets.connect(ws_url, open_timeout=20, close_timeout=10, ping_interval=20, ping_timeout=20) as websocket:
            logger.info(f"TTS WebSocket connection established for {utterance_id}.")
            await websocket.send(json.dumps(message_payload))
            logger.info(f"Sent 'generate_chunk' request for {utterance_id}.")

            last_message_time = time.monotonic()
            while True:
                try:
                    # Wait for the next message with timeout relative to the last received message
                    wait_time = max(0, receive_timeout_seconds - (time.monotonic() - last_message_time))
                    if wait_time == 0: # Check before waiting if timeout already exceeded
                        if all_audio_bytes:
                             logger.warning(f"TTS WebSocket receive timed out after {receive_timeout_seconds}s of inactivity for {utterance_id}. Received {len(all_audio_bytes)} bytes. Assuming stream ended.")
                             break
                        else:
                             logger.error(f"TTS WebSocket receive timed out after {receive_timeout_seconds}s for {utterance_id} with NO audio received.")
                             raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="TTS service timed out waiting for audio.")

                    message_json = await asyncio.wait_for(websocket.recv(), timeout=wait_time)
                    last_message_time = time.monotonic() # Reset timer on successful receive
                    message = json.loads(message_json)
                    msg_type = message.get("type")

                    if msg_type == "audio_chunk":
                        audio_b64 = message.get("audio_b64", "")
                        if audio_b64:
                            try:
                                chunk_bytes = base64.b64decode(audio_b64)
                                all_audio_bytes.extend(chunk_bytes)
                                logger.debug(f"Received audio chunk ({len(chunk_bytes)} bytes) for {utterance_id}.")
                            except Exception as decode_err:
                                logger.warning(f"Failed to decode base64 audio chunk for {utterance_id}: {decode_err}")
                    elif msg_type == "error":
                        error_msg = message.get("message", "Unknown TTS error")
                        logger.error(f"Received error from TTS WebSocket for {utterance_id}: {error_msg}")
                        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"TTS service error: {error_msg}")
                    elif msg_type == "stream_end":
                        logger.info(f"Received explicit 'stream_end' from TTS for {utterance_id}.")
                        break
                    elif msg_type == "context_cleared":
                         logger.info(f"Received 'context_cleared' confirmation from TTS for {utterance_id}.")
                    else:
                        logger.warning(f"Received unknown message type '{msg_type}' from TTS WS for {utterance_id}.")

                except asyncio.TimeoutError:
                    # This block is now reached only if wait_for times out
                    if all_audio_bytes:
                         logger.warning(f"TTS WebSocket receive timed out after {receive_timeout_seconds}s of inactivity for {utterance_id}. Received {len(all_audio_bytes)} bytes. Assuming stream ended.")
                         break
                    else:
                         logger.error(f"TTS WebSocket receive timed out after {receive_timeout_seconds}s for {utterance_id} with NO audio received.")
                         raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="TTS service timed out waiting for audio.")
                except websockets.exceptions.ConnectionClosedOK:
                    logger.info(f"TTS WebSocket connection closed normally by server for {utterance_id} after receiving {len(all_audio_bytes)} bytes.")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                     logger.error(f"TTS WebSocket connection closed with error for {utterance_id}: {e}. Received {len(all_audio_bytes)} bytes.")
                     if all_audio_bytes: break
                     raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"TTS WebSocket connection error: {e}")

            if websocket.open:
                try:
                     await websocket.send(json.dumps({"type": "clear_context", "utterance_id": utterance_id}))
                     logger.info(f"Sent 'clear_context' request for {utterance_id}.")
                except websockets.exceptions.ConnectionClosed:
                     logger.warning(f"Could not send 'clear_context' for {utterance_id}, connection already closed.")
                except Exception as clear_err:
                     logger.warning(f"Error sending 'clear_context' for {utterance_id}: {clear_err}")
            else:
                logger.warning(f"WebSocket connection for {utterance_id} already closed before sending 'clear_context'.")

        logger.info(f"TTS WebSocket processing for {utterance_id} complete. Total audio bytes: {len(all_audio_bytes)}.")
        if not all_audio_bytes:
            logger.warning(f"TTS service returned empty audio data via WebSocket for {utterance_id}.")
            return b""
        return bytes(all_audio_bytes)

    except websockets.exceptions.InvalidURI as e:
        logger.error(f"Invalid TTS WebSocket URI: {ws_url}. Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Invalid TTS WebSocket URI configured: {ws_url}")
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"TTS WebSocket connection failed to {ws_url}: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Failed to connect to TTS service WebSocket: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling TTS via WebSocket: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error communicating with TTS: {e}")

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
    await asyncio.sleep(5)
    await check_backend_services()
    yield
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down...")
    if http_client: await http_client.aclose(); logger.info("HTTP client closed.")
    logger.info("Orchestrator Service shutdown complete.")

async def check_backend_services():
    if not http_client: return
    services_to_check = {
        "ASR": api_endpoints['asr'].replace('/transcribe', '/health'),
        "LLM": api_endpoints['llm'].replace('/generate', '/health'),
        "TTS": urlunparse(('http', urlparse(api_endpoints['tts_ws']).netloc, '/health', '', '', '')),
    }
    logger.info("Checking backend service connectivity...")
    all_ok = True
    for name, url in services_to_check.items():
        logger.info(f"Checking {name} at {url}...")
        try:
            response = await http_client.get(url, timeout=5.0)
            if response.status_code >= 400 :
                 logger.warning(f"Backend service {name} health check failed: Status {response.status_code}. URL: {url}. Response: {response.text[:200]}")
                 if response.status_code == 503: logger.warning(f" -> {name} service is up but model might still be loading/failed.")
                 all_ok = False
            else: logger.info(f"Backend service {name} health check successful (Status {response.status_code}). URL: {url}")
        except httpx.RequestError as e: logger.error(f"Failed to connect to backend service {name} at {url}: {e}"); all_ok = False
        except Exception as e: logger.error(f"Unexpected error during {name} health check at {url}: {e}"); all_ok = False
    if not all_ok: logger.error("One or more critical backend services could not be reached or failed health check during startup.")
    else: logger.info("Initial backend service connectivity and health checks passed.")

# --- FastAPI App Creation and CORS ---
app = FastAPI(lifespan=lifespan, title="Voice Assistant Orchestrator", version="1.1.0")
origins = ["*"]
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
    assistant_audio_b64: str
    response_time_ms: float

# --- API Endpoints ---
@app.post("/assist", response_model=AssistResponse)
async def handle_assist_request(audio: UploadFile = File(..., description="User audio input (WAV, MP3, etc.)")):
    """
    Main endpoint to handle voice assistant interaction. (Uses WS for TTS)
    """
    overall_start_time = time.monotonic()
    logger.info(f"Received /assist request for file: {audio.filename}, size: {audio.size}")

    try:
        # 1. Call ASR Service
        asr_start_time = time.monotonic()
        user_transcript = await call_asr_service(audio)
        asr_time = (time.monotonic() - asr_start_time) * 1000

        # --- Handle no speech ---
        if not user_transcript:
            logger.info("ASR returned no transcript. Generating no-speech response.")
            no_speech_response = "Sorry, I didn't hear anything."
            tts_no_speech_start_time = time.monotonic()
            # Limit max length for no-speech response for speed
            no_speech_audio_bytes = await call_tts_service_ws(no_speech_response, max_audio_length_ms=5000)
            tts_no_speech_time = (time.monotonic() - tts_no_speech_start_time) * 1000
            logger.info(f"No-speech TTS took {tts_no_speech_time:.0f}ms")
            no_speech_audio_b64 = base64.b64encode(no_speech_audio_bytes).decode('utf-8') if no_speech_audio_bytes else ""
            return AssistResponse(
                user_transcript="",
                assistant_response=no_speech_response,
                assistant_audio_b64=no_speech_audio_b64,
                response_time_ms=(time.monotonic() - overall_start_time) * 1000
            )

        # 2. Prepare history for LLM
        current_llm_input = get_formatted_history()
        current_llm_input.append({"role": "user", "content": user_transcript})

        # 3. Call LLM Service
        llm_start_time = time.monotonic()
        assistant_response = await call_llm_service(current_llm_input)
        llm_time = (time.monotonic() - llm_start_time) * 1000

        # 4. Call TTS Service via WebSocket (WITH TEMPORARY LIMIT)
        tts_start_time = time.monotonic()
        # --- Apply TEMPORARY max_audio_length_ms limit for testing ---
        assistant_audio_bytes = await call_tts_service_ws(
            assistant_response,
            max_audio_length_ms=5000 # Limit to 5 seconds (~375 frames)
        )
        # ----------------------------------------------------------
        tts_time = (time.monotonic() - tts_start_time) * 1000

        # 5. Update History
        update_history(user_text=user_transcript, assistant_text=assistant_response)

        # 6. Encode audio
        assistant_audio_b64 = base64.b64encode(assistant_audio_bytes).decode('utf-8') if assistant_audio_bytes else ""

        # 7. Log and return
        overall_time = (time.monotonic() - overall_start_time) * 1000
        logger.info(f"Full assist request processed in {overall_time:.2f}ms (ASR: {asr_time:.0f}ms, LLM: {llm_time:.0f}ms, TTS: {tts_time:.0f}ms)")

        return AssistResponse(
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            assistant_audio_b64=assistant_audio_b64,
            response_time_ms=overall_time
        )

    except HTTPException as http_exc:
         logger.error(f"Pipeline failed due to HTTPException: {http_exc.status_code} - {http_exc.detail}")
         raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during /assist pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal orchestration error occurred: {e}"
        )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
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


# --- Main Execution Guard (for local debugging) ---
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
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=port, log_level=log_level_param, reload=False)
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down (direct run)...")