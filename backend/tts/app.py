# backend/tts/app.py
# --- Propagating max_audio_length_ms and cleanup ---

import os
import logging
import io
import time
import base64
import asyncio
import json
import numpy as np
import soundfile as sf
import torch
import yaml
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status as fastapi_status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from huggingface_hub import login, logout
from dotenv import load_dotenv
from starlette.websockets import WebSocketState
# --- CSM Library Imports ---
CSM_LOADED = False
try:
    import generator
    import models
    load_csm_1b = generator.load_csm_1b
    Segment = generator.Segment
    Generator = generator.Generator
    # logger_csm = logging.getLogger("generator") # Comment out if not needed
    # logger_csm.setLevel(os.getenv('LOG_LEVEL', 'info').upper())
    CSM_LOADED = True
    logging.info("Successfully prepared direct imports for copied generator/models.")
except ImportError as e:
    logging.error(f"FATAL: Failed direct import of copied 'generator'/'models'. Error: {e}", exc_info=True)
    def load_csm_1b(**kwargs): raise NotImplementedError("CSM library import failed")
    class Segment: pass
    class Generator: pass

# --- Constants & Environment Loading ---
load_dotenv()
CONFIG_PATH = os.getenv('CONFIG_PATH', '/app/config.yaml')
CACHE_DIR = Path(os.getenv('CACHE_DIR', '/cache'))
HF_CACHE_DIR = Path(os.getenv('HF_HOME', CACHE_DIR / "huggingface"))
LOG_FILE_BASE = os.getenv('LOG_FILE_BASE', '/app/logs/service')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'info').upper()
USE_GPU_ENV = os.getenv('USE_GPU', 'auto').lower()
HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
TTS_SPEAKER_ID = int(os.getenv('TTS_SPEAKER_ID', '4'))
TTS_MODEL_REPO_ID_DEFAULT = 'senstella/csm-expressiva-1b'
TTS_MODEL_REPO_ID_ENV = os.getenv('TTS_MODEL_REPO_ID', TTS_MODEL_REPO_ID_DEFAULT)
SERVICE_NAME = "tts_streaming"
LOG_PATH = f"{LOG_FILE_BASE}_{SERVICE_NAME}.log"

# --- Logging Setup ---
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)]
)
logger = logging.getLogger(SERVICE_NAME)

# --- Global Variables ---
tts_generator: Optional[Generator] = None
tts_config: Dict[str, Any] = {}
effective_device: str = "cpu"
model_load_info: Dict[str, Any] = {"status": "pending"}
# Remove expected_sample_rate global, get from tts_config or generator later
# expected_sample_rate: int = 24000
utterance_context_cache: Dict[str, List[Segment]] = {}

# --- Configuration Loading ---
def load_configuration():
    global tts_config, effective_device, TTS_SPEAKER_ID, model_load_info
    try:
        logger.info(f"Loading configuration from: {CONFIG_PATH}")
        if not os.path.exists(CONFIG_PATH):
             logger.warning(f"Config file not found at {CONFIG_PATH}. Using defaults/env vars.")
             config = {'tts': {}}
        else:
            with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)
            if not config or 'tts' not in config:
                 logger.warning(f"Config file {CONFIG_PATH} is missing 'tts' section. Using defaults.")
                 config['tts'] = {}
        tts_config = config.get('tts', {})
        config_model_id = tts_config.get('model_repo_id')
        if TTS_MODEL_REPO_ID_ENV != TTS_MODEL_REPO_ID_DEFAULT:
            tts_config['model_repo_id'] = TTS_MODEL_REPO_ID_ENV
            logger.info(f"Using TTS_MODEL_REPO_ID from environment: {TTS_MODEL_REPO_ID_ENV}")
        elif config_model_id:
             tts_config['model_repo_id'] = config_model_id
             logger.info(f"Using model_repo_id from config.yaml: {config_model_id}")
        else:
             tts_config['model_repo_id'] = TTS_MODEL_REPO_ID_DEFAULT
             logger.info(f"Using default TTS_MODEL_REPO_ID: {TTS_MODEL_REPO_ID_DEFAULT}")
        config_speaker_id_str = str(tts_config.get('speaker_id', '4'))
        env_speaker_id_str = os.getenv('TTS_SPEAKER_ID', config_speaker_id_str)
        try: TTS_SPEAKER_ID = int(env_speaker_id_str)
        except ValueError: logger.warning(f"Invalid Speaker ID '{env_speaker_id_str}'. Using default 4."); TTS_SPEAKER_ID = 4
        tts_config['effective_speaker_id'] = TTS_SPEAKER_ID
        logger.info(f"Final Effective TTS Model Repo ID: {tts_config['model_repo_id']}")
        logger.info(f"Final Effective TTS Speaker ID: {TTS_SPEAKER_ID}")
        # Don't need expected_sample_rate from config anymore, get from generator
        # expected_sample_rate = int(tts_config.get('expected_sample_rate', 24000))
        # logger.info(f"Expected sample rate from config/default: {expected_sample_rate} Hz")
        config_device = tts_config.get('device', 'auto')
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available via torch check: {cuda_available}, Torch: {torch.__version__}")
        if cuda_available and (USE_GPU_ENV == 'auto' or USE_GPU_ENV == 'true'):
            if config_device.startswith("cuda"): effective_device = config_device
            else: effective_device = "cuda"
            logger.info(f"CUDA is available and requested/auto. Using CUDA device: '{effective_device}'.")
        else:
             effective_device = "cpu"
             logger.critical(f"FATAL: CUDA is required for CSM, but it's not available or disabled (USE_GPU={USE_GPU_ENV}). Cannot proceed with model loading.")
             model_load_info.update({"status": "error", "error": "CUDA unavailable/disabled, required by CSM."})
        tts_config['effective_device'] = effective_device
        logger.info(f"TTS effective device target: {effective_device}")
    except Exception as e:
        logger.critical(f"Unexpected error loading configuration: {e}", exc_info=True)
        tts_config = {}; model_load_info.update({"status": "error", "error": f"Config error: {e}"})

# --- Model Loading ---
def load_tts_model():
    global tts_generator, model_load_info, tts_config
    if not CSM_LOADED: # ... (error handling) ...
         error_msg = model_load_info.get("error", "CSM library components failed import.")
         logger.critical(f"Cannot load model: {error_msg}")
         raise RuntimeError(error_msg)
    if not tts_config: # ... (error handling) ...
        error_msg = model_load_info.get("error", "TTS configuration missing/invalid.")
        logger.critical(f"Cannot load model: {error_msg}")
        model_load_info.setdefault("status", "error"); model_load_info.setdefault("error", error_msg)
        raise RuntimeError(error_msg)
    device_to_load = tts_config.get('effective_device')
    if not device_to_load or not device_to_load.startswith("cuda"): # ... (error handling) ...
         error_msg = f"CSM requires CUDA, but effective device is '{device_to_load}'."
         logger.critical(f"FATAL: {error_msg}")
         model_load_info.update({"status": "error", "error": error_msg})
         raise RuntimeError(error_msg)
    model_repo_id = tts_config.get('model_repo_id')
    if not model_repo_id: # ... (error handling) ...
         error_msg = "TTS model_repo_id missing from config."
         logger.critical(f"FATAL: {error_msg}")
         model_load_info.update({"status": "error", "error": error_msg})
         raise RuntimeError(error_msg)
    logger.info(f"Attempting to load TTS model: {model_repo_id} on device: {device_to_load}")
    logger.info(f"Using cache directory (HF_HOME): {HF_CACHE_DIR}")
    model_load_info = {"status": "loading", "model_repo_id": model_repo_id, "device": device_to_load}
    start_time = time.monotonic()
    try:
        if HF_TOKEN: # ... (login logic) ...
             logger.info("Logging into Hugging Face Hub.")
             login(token=HF_TOKEN, add_to_git_credential=False)
             logger.info("HF Login successful.")
        else: logger.warning("HUGGING_FACE_TOKEN env var not set.")
        logger.info(f"Calling generator.load_csm_1b(model_id='{model_repo_id}', device='{device_to_load}')")
        tts_generator = load_csm_1b( model_id=model_repo_id, device=device_to_load )
        if tts_generator is None: raise RuntimeError(f"load_csm_1b returned None.")
        if not isinstance(tts_generator, Generator): raise TypeError(f"load_csm_1b did not return Generator.")
        load_time = time.monotonic() - start_time
        logger.info(f"Model load call completed in {load_time:.2f}s.")
        # --- Get sample rate from generator and store in config ---
        actual_sample_rate = getattr(tts_generator, 'sample_rate', None)
        if actual_sample_rate:
             logger.info(f"Detected generator sample rate: {actual_sample_rate} Hz")
             tts_config['actual_sample_rate'] = actual_sample_rate
        else:
             logger.warning(f"Could not get sample rate from generator. Using default 24000 Hz for encoding.")
             tts_config['actual_sample_rate'] = 24000 # Fallback default
        # --- End sample rate handling ---
        model_load_info.update({"status": "loaded", "load_time_s": round(load_time, 2), "sample_rate": tts_config['actual_sample_rate']})
        logger.info(f"TTS Model '{model_repo_id}' loaded successfully on {device_to_load}.")
        try: # ... (device confirmation logic) ...
             actual_device = next(tts_generator._model.parameters()).device
             logger.info(f"Model confirmed on device: {actual_device}")
        except Exception as dev_check_err: logger.warning(f"Could not confirm model device post-load: {dev_check_err}")
    except Exception as e:
        logger.critical(f"FATAL: Model loading failed for '{model_repo_id}': {e}", exc_info=True)
        tts_generator = None; load_time = time.monotonic() - start_time
        model_load_info.update({"status": "error", "error": f"Model loading failed: {e}", "load_time_s": round(load_time, 2)})
        raise RuntimeError(f"TTS model loading failed: {e}") from e
    finally:
        if HF_TOKEN: # ... (logout logic) ...
             try: logout(); logger.info("HF logout successful.")
             except Exception as logout_err: logger.warning(f"HF logout error: {logout_err}")

# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{SERVICE_NAME.upper()} Service starting up...")
    global model_load_info
    model_load_info = {"status": "initializing"}
    startup_error = None
    try:
        if not CSM_LOADED: # ... (error handling) ...
             startup_error = model_load_info.get("error", "CSM library import failed at startup.")
             raise RuntimeError(startup_error)
        load_configuration()
        if model_load_info.get("status") == "error": # ... (error handling) ...
             startup_error = model_load_info.get("error", "Config loading failed.")
             raise RuntimeError(startup_error)
        if tts_config.get('effective_device','cpu').startswith('cuda'):
             load_tts_model()
             if model_load_info.get("status") != "loaded": # ... (error handling) ...
                   startup_error = model_load_info.get("error", "Model loading status not 'loaded' after attempt.")
                   raise RuntimeError(startup_error)
        else: # ... (error handling) ...
             startup_error = "CUDA device not configured or available, cannot load CSM model."
             raise RuntimeError(startup_error)
        logger.info("Lifespan startup sequence completed successfully.")
    except Exception as e:
        startup_error = str(e)
        logger.critical(f"Lifespan startup failed: {startup_error}", exc_info=True)
        model_load_info.setdefault("status", "error"); model_load_info.setdefault("error", startup_error)
    yield
    # --- Shutdown Logic --- (Remains the same)
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down...")
    global tts_generator, utterance_context_cache, effective_device
    if tts_generator: # ... (cleanup) ...
        logger.info("Releasing TTS generator instance...")
        try: del tts_generator; tts_generator = None; logger.info("TTS generator instance deleted.")
        except Exception as del_err: logger.warning(f"Error deleting generator instance: {del_err}")
    utterance_context_cache.clear(); logger.info("Utterance context cache cleared.")
    if effective_device and effective_device.startswith("cuda"): # ... (cleanup) ...
        try: torch.cuda.empty_cache(); logger.info("Cleared CUDA cache.")
        except Exception as e: logger.warning(f"CUDA cache clear failed during shutdown: {e}")
    logger.info("TTS Service shutdown complete.")

# --- FastAPI App Initialization ---
# Bump version number reflecting changes
app = FastAPI(lifespan=lifespan, title="TTS Streaming Service (CSM - senstella)", version="1.3.1")

# --- Helper Functions ---
def numpy_to_base64_wav(audio_np: np.ndarray, samplerate: int) -> str:
    # ... (remains the same) ...
    if not isinstance(audio_np, np.ndarray): logger.warning(f"Input not NumPy array (type: {type(audio_np)}). Returning empty string."); return ""
    if audio_np.size == 0: logger.warning("Attempted encode empty NumPy array. Returning empty string."); return ""
    try:
        if audio_np.dtype != np.float32: audio_np = audio_np.astype(np.float32)
        max_val = np.max(np.abs(audio_np))
        if max_val > 1.0: logger.warning(f"Audio data max abs val > 1.0 ({max_val:.3f}). Normalizing."); audio_np = audio_np / max_val
        buffer = io.BytesIO(); sf.write(buffer, audio_np, samplerate, format='WAV', subtype='FLOAT'); buffer.seek(0)
        wav_bytes = buffer.getvalue(); b64_string = base64.b64encode(wav_bytes).decode('utf-8')
        logger.debug(f"Encoded {len(wav_bytes)} bytes WAV ({len(audio_np)/samplerate:.2f}s) to base64 string.")
        return b64_string
    except Exception as e: logger.error(f"Error encoding numpy audio to WAV base64: {e}", exc_info=True); return ""

# --- WebSocket Endpoint (Passing max_audio_length_ms) ---
@app.websocket("/synthesize_stream")
async def synthesize_stream_endpoint(websocket: WebSocket):
    client_host = websocket.client.host if websocket.client else "unknown_host"
    client_port = websocket.client.port if websocket.client else "unknown_port"
    connection_id = f"{client_host}:{client_port}"
    await websocket.accept()
    logger.info(f"WebSocket connection accepted from: {connection_id}")

    if not tts_generator or model_load_info.get("status") != "loaded":
        # ... (error handling remains the same) ...
        error_detail = model_load_info.get("error", "TTS service not ready or model failed to load.")
        logger.error(f"Rejecting WebSocket connection {connection_id}: Service Status='{model_load_info.get('status')}'. Reason: {error_detail}")
        error_payload = {"type": "error", "message": f"TTS service unavailable: {error_detail}", "model_status": model_load_info.get("status", "unknown")}
        try: await websocket.send_json(error_payload)
        except Exception as send_err: logger.warning(f"Failed to send error status to connecting client {connection_id}: {send_err}")
        try: await websocket.close(code=fastapi_status.WS_1011_INTERNAL_ERROR)
        except Exception as close_err: logger.warning(f"Failed to close WebSocket connection cleanly for {connection_id} after error: {close_err}")
        return

    current_utterance_id: Optional[str] = None
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                msg_type = message.get("type")
                utt_id_log = message.get('utterance_id', 'N/A')
                logger.debug(f"WS Recv ({connection_id}): Type='{msg_type}' UttId='{utt_id_log}'")

                if msg_type == "generate_chunk":
                    text_chunk = message.get("text_chunk", "").strip()
                    utterance_id = message.get("utterance_id")
                    # --- Extract max_audio_length_ms ---
                    max_len_ms_from_msg = message.get('max_audio_length_ms') # Can be None
                    # --- End Extraction ---

                    if not text_chunk: logger.debug(f"Received empty text_chunk for UttId {utterance_id} from {connection_id}. Skipping."); continue
                    if not utterance_id:
                        logger.warning(f"Missing utterance_id from {connection_id}. Sending error response.")
                        await websocket.send_json({"type": "error", "message": "Missing required 'utterance_id'."}); continue

                    if utterance_id != current_utterance_id:
                        logger.info(f"Switching to utterance_id ({connection_id}): '{utterance_id}' (Previous: '{current_utterance_id}')")
                        current_utterance_id = utterance_id
                        if utterance_id not in utterance_context_cache:
                            logger.debug(f"Initializing empty context cache for utterance_id: {utterance_id}")
                            utterance_context_cache[utterance_id] = []

                    gen_start_time = time.monotonic()
                    try:
                        context_for_csm: List[Segment] = utterance_context_cache.get(utterance_id, [])
                        # --- Get defaults from config ---
                        # Use generator's sample rate for logging/encoding
                        effective_sr = getattr(tts_generator, 'sample_rate', 24000)
                        default_max_len_ms = float(tts_config.get('max_audio_length_ms', 90000)) # Default from config
                        temp = float(tts_config.get('temperature', 0.9))
                        top_k = int(tts_config.get('top_k', 50))

                        # --- Determine effective max_len_ms to pass to generator ---
                        if max_len_ms_from_msg is not None:
                            try:
                                max_len_ms = float(max_len_ms_from_msg)
                                if max_len_ms <= 0:
                                    logger.warning(f"Received invalid max_audio_length_ms ({max_len_ms_from_msg}). Using default.")
                                    max_len_ms = default_max_len_ms
                                else:
                                    logger.info(f"Using max_audio_length_ms from request: {max_len_ms}ms")
                            except (ValueError, TypeError):
                                logger.warning(f"Received non-numeric max_audio_length_ms ('{max_len_ms_from_msg}'). Using default.")
                                max_len_ms = default_max_len_ms
                        else:
                             max_len_ms = default_max_len_ms # Use default if not provided in message
                        # -------------------------------------------------------------

                        logger.info(f"Generating chunk ({connection_id}): UttId='{utterance_id}', CtxLen={len(context_for_csm)}, SpkID={TTS_SPEAKER_ID}, MaxLenMs={max_len_ms}, Temp={temp}, TopK={top_k}, Text='{text_chunk[:60]}...'")

                        # --- Pass the determined max_len_ms to generate call ---
                        audio_tensor = tts_generator.generate(
                            text=text_chunk,
                            speaker=TTS_SPEAKER_ID,
                            context=context_for_csm,
                            max_audio_length_ms=max_len_ms, # Pass the determined max length
                            temperature=temp,
                            topk=top_k
                        )
                        # --------------------------------------------------
                        gen_time = time.monotonic() - gen_start_time

                        if not isinstance(audio_tensor, torch.Tensor) or audio_tensor.ndim == 0 or audio_tensor.numel() == 0:
                            logger.warning(f"CSM generation returned empty tensor ({connection_id}, UttId: {utterance_id}).")
                            audio_b64 = ""
                            new_segment = Segment(text=text_chunk, speaker=TTS_SPEAKER_ID, audio=None)
                        else:
                            # audio_tensor is already on CPU from generator.generate()
                            audio_np = audio_tensor.numpy()
                            audio_duration_s = len(audio_np) / effective_sr
                            logger.info(f"CSM generation ({connection_id}, UttId: {utterance_id}) took {gen_time:.3f}s. Audio Duration: {audio_duration_s:.2f}s")
                            # Store CPU tensor in segment
                            new_segment = Segment(text=text_chunk, speaker=TTS_SPEAKER_ID, audio=audio_tensor)
                            audio_b64 = numpy_to_base64_wav(audio_np, effective_sr)

                        if utterance_id in utterance_context_cache: utterance_context_cache[utterance_id].append(new_segment)
                        else: logger.warning(f"Utterance context cache for '{utterance_id}' ({connection_id}) disappeared. Re-initializing."); utterance_context_cache[utterance_id] = [new_segment]

                        await websocket.send_json({
                            "type": "audio_chunk", "utterance_id": utterance_id, "audio_b64": audio_b64,
                            "text_chunk": text_chunk, "generation_time_ms": round(gen_time * 1000)
                        })
                        logger.debug(f"Sent audio chunk for UttId {utterance_id} to {connection_id}")

                    # --- Keep specific error handling ---
                    except RuntimeError as rt_err: logger.error(f"RuntimeError during TTS generation ({connection_id}, UttId: {utterance_id}): {rt_err}", exc_info=False); await websocket.send_json({"type": "error", "message": f"TTS runtime error: {rt_err}", "utterance_id": utterance_id}) # Less verbose traceback for runtime
                    except ValueError as val_err: logger.error(f"ValueError during TTS generation ({connection_id}, UttId: {utterance_id}): {val_err}", exc_info=False); await websocket.send_json({"type": "error", "message": f"TTS input error: {val_err}", "utterance_id": utterance_id})
                    except Exception as gen_err: logger.error(f"Unexpected error during TTS generation ({connection_id}, UttId: {utterance_id}): {gen_err}", exc_info=True); await websocket.send_json({"type": "error", "message": f"Unexpected TTS generation error: {gen_err}", "utterance_id": utterance_id})

                elif msg_type == "clear_context": # ... (remains the same) ...
                    utterance_id = message.get("utterance_id")
                    if utterance_id:
                        if utterance_id in utterance_context_cache: logger.info(f"Clearing context ({connection_id}) for utterance_id: {utterance_id}"); del utterance_context_cache[utterance_id]
                        else: logger.debug(f"Received clear_context for non-cached utterance_id: {utterance_id} ({connection_id})")
                        if utterance_id == current_utterance_id: current_utterance_id = None
                        await websocket.send_json({"type": "context_cleared", "utterance_id": utterance_id})
                    else: logger.warning(f"Received clear_context without utterance_id from {connection_id}"); await websocket.send_json({"type": "error", "message": "Missing 'utterance_id' for clear_context."})

                else: # ... (remains the same) ...
                    logger.warning(f"Unknown WebSocket message type '{msg_type}' received from {connection_id}")
                    await websocket.send_json({"type": "error", "message": f"Unknown command type: {msg_type}"})

            except WebSocketDisconnect:
                 logger.info(f"WebSocket disconnected by client: {connection_id}")
                 break
            except json.JSONDecodeError: # ... (remains the same) ...
                logger.error(f"Failed to decode JSON message from {connection_id}. Message ignored.")
                try: await websocket.send_json({"type": "error", "message": "Invalid JSON received."})
                except Exception: pass
            except Exception as e: # ... (remains the same) ...
                logger.error(f"Unexpected error in WebSocket loop ({connection_id}): {e}", exc_info=True)
                try: await websocket.send_json({"type": "error", "message": "Internal server error occurred."})
                except Exception: pass
                break
        logger.info(f"WebSocket message loop finished for {connection_id}")
    finally: # ... (remains the same) ...
        logger.info(f"Cleaning up WebSocket connection resources for {connection_id}")
        if current_utterance_id and current_utterance_id in utterance_context_cache:
            logger.info(f"Cleaning up final context cache for utterance '{current_utterance_id}' from connection {connection_id}.")
            try: del utterance_context_cache[current_utterance_id]
            except KeyError: logger.debug(f"Context cache for '{current_utterance_id}' ({connection_id}) was already removed.")
        if websocket.client_state == WebSocketState.CONNECTED:
            logger.debug(f"Attempting to close connected WebSocket for {connection_id} in finally block.")
            try: await websocket.close(code=fastapi_status.WS_1000_NORMAL_CLOSURE)
            except Exception as close_err: logger.warning(f"Error closing WebSocket for {connection_id} in finally block: {close_err}")
        else: logger.info(f"WebSocket for {connection_id} already closed (State: {websocket.client_state}). No close action needed.")

# --- Health Check Endpoint ---
@app.get("/health", status_code=fastapi_status.HTTP_200_OK)
async def health_check():
    # ... (remains the same) ...
    current_status = model_load_info.get("status", "unknown")
    is_healthy = (current_status == "loaded" and tts_generator is not None)
    response_content = { # ... (content definition) ...
        "service": SERVICE_NAME, "status": "ok" if is_healthy else "error",
        "details": { "model_status": current_status, "model_repo_id": tts_config.get('model_repo_id', 'N/A'),
                     "effective_device": tts_config.get('effective_device', 'N/A'), "target_speaker_id": tts_config.get('effective_speaker_id', 'N/A'),
                     "actual_sample_rate": tts_config.get('actual_sample_rate', 'N/A'), "generator_object_present": (tts_generator is not None),
                     "load_info": model_load_info } }
    status_code = fastapi_status.HTTP_200_OK if is_healthy else fastapi_status.HTTP_503_SERVICE_UNAVAILABLE
    if not is_healthy: # ... (error reporting) ...
        logger.warning(f"Health check reporting unhealthy: Status='{current_status}', Generator Present={tts_generator is not None}")
        if model_load_info.get("error"): response_content["details"]["error_message"] = model_load_info.get("error")
        return JSONResponse(status_code=status_code, content=response_content)
    try: # ... (generator check) ...
        _ = tts_generator.device; _ = tts_generator.sample_rate; logger.debug(f"Health check: Generator instance accessed successfully (device: {tts_generator.device}).")
    except Exception as gen_check_err: # ... (error handling) ...
        logger.error(f"Health check failed accessing generator properties: {gen_check_err}", exc_info=True)
        response_content["status"] = "error"; response_content["details"]["error_details"] = f"Generator access error: {gen_check_err}"
        model_load_info["status"] = "error_runtime_access"; model_load_info["error"] = f"Generator access error: {gen_check_err}"
        status_code = fastapi_status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=response_content)

# --- Main Execution Guard ---
if __name__ == "__main__":
    # ... (remains the same) ...
    logger.info(f"Starting {SERVICE_NAME.upper()} service directly via __main__...")
    main_startup_error = None
    try: # ... (startup logic) ...
        logger.info("Running startup sequence (direct run)...")
        if not CSM_LOADED: main_startup_error = model_load_info.get("error", "CSM import failed."); raise RuntimeError(main_startup_error)
        load_configuration()
        if model_load_info.get("status") == "error": main_startup_error = model_load_info.get("error", "Config loading failed."); raise RuntimeError(main_startup_error)
        if tts_config.get('effective_device','cpu').startswith('cuda'):
            load_tts_model()
            if model_load_info.get("status") != "loaded": main_startup_error = model_load_info.get("error", "Model loading status not 'loaded'."); raise RuntimeError(main_startup_error)
        else: main_startup_error = "CUDA device not configured or available, cannot load CSM model."; model_load_info.update({"status": "error", "error": main_startup_error}); raise RuntimeError(main_startup_error)
        port = int(os.getenv('TTS_PORT', 5003)); host = os.getenv('TTS_HOST', "0.0.0.0"); log_level_param = os.getenv('LOG_LEVEL', 'info').lower()
        logger.info(f"Direct run startup successful. Launching Uvicorn on {host}:{port}...")
        import uvicorn
        uvicorn.run("app:app", host=host, port=port, log_level=log_level_param, reload=False)
    except RuntimeError as e: logger.critical(f"Direct run failed during startup: {e}"); exit(1)
    except ImportError as e: logger.critical(f"Direct run failed: Missing dependency? {e}", exc_info=True); exit(1)
    except Exception as e: logger.critical(f"Unexpected error during direct run: {e}", exc_info=True); exit(1)
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down (direct run)...")