# backend/tts/app.py
# --- PASTE THIS ENTIRE BLOCK INTO YOUR FILE ---

import os
import logging
import io
import time
import base64
import asyncio # Ensure asyncio is imported
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
    CSM_LOADED = True
    logging.info("Successfully prepared direct imports for copied generator/models.")
except ImportError as e:
    logging.error(f"FATAL: Failed direct import of copied 'generator'/'models'. Error: {e}", exc_info=True)
    # Define dummy versions so the rest of the code doesn't raise NameError immediately
    def load_csm_1b(**kwargs): raise NotImplementedError("CSM library import failed")
    class Segment: pass
    class Generator: pass

# --- Constants & Environment Loading ---
load_dotenv()
CONFIG_PATH = os.getenv('CONFIG_PATH', '/app/config.yaml')
CACHE_DIR = Path(os.getenv('CACHE_DIR', '/cache'))
HF_HOME_ENV = os.getenv('HF_HOME')
HF_CACHE_DIR = Path(HF_HOME_ENV) if HF_HOME_ENV else CACHE_DIR / "huggingface"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
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
utterance_context_cache: Dict[str, List[Segment]] = {}

# --- Configuration Loading ---
# ... (Keep existing load_configuration function as is) ...
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

        # Determine Model Repo ID (Env Var > Config > Default)
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

        # Determine Speaker ID (Env Var > Config > Default)
        config_speaker_id_str = str(tts_config.get('speaker_id', '4')) # Default to 4
        env_speaker_id_str = os.getenv('TTS_SPEAKER_ID', config_speaker_id_str)
        try:
            TTS_SPEAKER_ID = int(env_speaker_id_str)
        except ValueError:
            logger.warning(f"Invalid Speaker ID '{env_speaker_id_str}'. Using default 4.")
            TTS_SPEAKER_ID = 4
        tts_config['effective_speaker_id'] = TTS_SPEAKER_ID

        logger.info(f"Final Effective TTS Model Repo ID: {tts_config['model_repo_id']}")
        logger.info(f"Final Effective TTS Speaker ID: {TTS_SPEAKER_ID}")

        # Determine Device (Env Var USE_GPU > Config 'device' > Auto)
        config_device = tts_config.get('device', 'auto').lower()
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available via torch check: {cuda_available}, Torch: {torch.__version__}")

        if cuda_available and (USE_GPU_ENV == 'true' or (USE_GPU_ENV == 'auto' and config_device != 'cpu')):
            if config_device.startswith("cuda"):
                effective_device = config_device # Use specific cuda device from config (e.g., "cuda:0")
            else:
                effective_device = "cuda" # Default to "cuda"
            logger.info(f"CUDA is available and requested/auto. Using CUDA device: '{effective_device}'.")
        else:
             effective_device = "cpu"
             # CSM requires CUDA, so treat this as a fatal error if CUDA was expected
             if USE_GPU_ENV == 'true' or (USE_GPU_ENV == 'auto' and config_device != 'cpu'):
                 logger.critical(f"FATAL: CUDA requested/required but unavailable/disabled (USE_GPU={USE_GPU_ENV}, cuda_available={cuda_available}). Cannot load CSM model.")
                 model_load_info.update({"status": "error", "error": "CUDA unavailable/disabled, required by CSM."})
             else:
                  logger.warning("CUDA not available or disabled. TTS service cannot load CSM model.")
                  model_load_info.update({"status": "error", "error": "CUDA unavailable/disabled, required by CSM."})

        tts_config['effective_device'] = effective_device
        logger.info(f"TTS effective device target: {effective_device}")

        # Store other config values with defaults
        tts_config['max_audio_length_ms'] = float(tts_config.get('max_audio_length_ms', 90000))
        tts_config['temperature'] = float(tts_config.get('temperature', 0.9))
        tts_config['top_k'] = int(tts_config.get('top_k', 50))
        logger.info(f"Generation params: max_len={tts_config['max_audio_length_ms']}ms, temp={tts_config['temperature']}, top_k={tts_config['top_k']}")

    except Exception as e:
        logger.critical(f"Unexpected error loading configuration: {e}", exc_info=True)
        tts_config = {} # Reset config on error
        model_load_info.update({"status": "error", "error": f"Config error: {e}"})

# --- Model Loading ---
# ... (Keep existing load_tts_model function as is) ...
def load_tts_model():
    global tts_generator, model_load_info, tts_config
    if not CSM_LOADED:
         error_msg = model_load_info.get("error", "CSM library components failed import.")
         logger.critical(f"Cannot load model: {error_msg}")
         raise RuntimeError(error_msg)

    if not tts_config or model_load_info.get("status") == "error":
        error_msg = model_load_info.get("error", "TTS configuration missing or invalid.")
        logger.critical(f"Cannot load model: {error_msg}")
        model_load_info.setdefault("status", "error")
        model_load_info.setdefault("error", error_msg)
        raise RuntimeError(error_msg)

    device_to_load = tts_config.get('effective_device')
    if not device_to_load or not device_to_load.startswith("cuda"):
         error_msg = model_load_info.get("error", f"CSM requires CUDA, but effective device is '{device_to_load}'.")
         logger.critical(f"FATAL: {error_msg}")
         model_load_info.update({"status": "error", "error": error_msg})
         raise RuntimeError(error_msg)

    model_repo_id = tts_config.get('model_repo_id')
    if not model_repo_id:
         error_msg = "TTS model_repo_id missing from config."
         logger.critical(f"FATAL: {error_msg}")
         model_load_info.update({"status": "error", "error": error_msg})
         raise RuntimeError(error_msg)

    logger.info(f"Attempting to load TTS model: {model_repo_id} on device: {device_to_load}")
    logger.info(f"Using cache directory (HF_HOME): {HF_CACHE_DIR}")
    model_load_info = {"status": "loading", "model_repo_id": model_repo_id, "device": device_to_load}
    start_time = time.monotonic()

    try:
        logger.info(f"Calling generator.load_csm_1b(model_id='{model_repo_id}', device='{device_to_load}')")
        tts_generator = load_csm_1b( model_id=model_repo_id, device=device_to_load )

        if tts_generator is None:
            raise RuntimeError(f"generator.load_csm_1b returned None for model '{model_repo_id}'.")
        if not isinstance(tts_generator, Generator):
            raise TypeError(f"load_csm_1b did not return a Generator instance (got {type(tts_generator)}).")

        load_time = time.monotonic() - start_time
        logger.info(f"Model load call completed in {load_time:.2f}s.")

        actual_sample_rate = getattr(tts_generator, 'sample_rate', None)
        if actual_sample_rate:
             logger.info(f"Detected generator sample rate: {actual_sample_rate} Hz")
             tts_config['actual_sample_rate'] = actual_sample_rate
        else:
             logger.warning(f"Could not get sample rate from generator. Using default 24000 Hz for encoding.")
             tts_config['actual_sample_rate'] = 24000 # Fallback default

        model_load_info.update({"status": "loaded", "load_time_s": round(load_time, 2), "sample_rate": tts_config['actual_sample_rate']})
        logger.info(f"TTS Model '{model_repo_id}' loaded successfully on {device_to_load}.")

        try:
             actual_device = next(tts_generator._model.parameters()).device
             logger.info(f"Model confirmed on device: {actual_device}")
             if str(actual_device) != device_to_load:
                  logger.warning(f"Model loaded on {actual_device} but target was {device_to_load}.")
        except Exception as dev_check_err:
             logger.warning(f"Could not confirm model device post-load: {dev_check_err}")

    except Exception as e:
        logger.critical(f"FATAL: Model loading failed for '{model_repo_id}': {e}", exc_info=True)
        tts_generator = None
        load_time = time.monotonic() - start_time
        model_load_info.update({"status": "error", "error": f"Model loading failed: {e}", "load_time_s": round(load_time, 2)})
        raise RuntimeError(f"TTS model loading failed: {e}") from e

# --- FastAPI Lifespan ---
# ... (Keep existing lifespan function as is) ...
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown, including model loading and optional warm-up."""
    logger.info(f"{SERVICE_NAME.upper()} Service starting up...")
    global model_load_info, tts_config, tts_generator
    model_load_info = {"status": "initializing"}
    startup_error = None

    try:
        # --- Startup sequence logic ---
        if not CSM_LOADED:
             startup_error = model_load_info.get("error", "CSM library import failed at startup.")
             model_load_info.update({"status": "error", "error": startup_error})
             raise RuntimeError(startup_error)

        load_configuration()
        if model_load_info.get("status") == "error":
             startup_error = model_load_info.get("error", "Config loading failed.")
             raise RuntimeError(startup_error)

        if tts_config.get('effective_device','cpu').startswith('cuda'):
             load_tts_model()
             if model_load_info.get("status") != "loaded":
                   startup_error = model_load_info.get("error", "Model loading status not 'loaded' after successful call.")
                   logger.error(f"Inconsistent state: load_tts_model completed but status is {model_load_info.get('status')}")
                   model_load_info["status"] = "error"
                   model_load_info.setdefault("error", startup_error)
                   raise RuntimeError(startup_error)
        else:
             startup_error = model_load_info.get("error", "CUDA device not configured or available.")
             logger.critical(f"Lifespan: {startup_error}")
             raise RuntimeError(startup_error)

        if tts_generator and model_load_info.get("status") == "loaded":
            logger.info("Attempting TTS model warm-up with dummy inference...")
            warmup_speaker_id = tts_config.get('effective_speaker_id', 4)
            try:
                await asyncio.to_thread(
                     tts_generator.generate,
                     text="Ready.",
                     speaker=warmup_speaker_id,
                     context=[],
                     max_audio_length_ms=1000
                )
                logger.info("TTS model warm-up inference completed successfully.")
            except AttributeError:
                logger.warning("tts_generator does not have a 'generate' method, skipping lifespan warmup.")
            except Exception as warmup_err:
                logger.warning(f"TTS model warm-up failed: {warmup_err}", exc_info=True)

        logger.info("Lifespan startup sequence completed successfully.")

    except Exception as e:
        startup_error = str(e)
        logger.critical(f"Lifespan startup failed: {startup_error}", exc_info=True)
        model_load_info["status"] = "error"
        model_load_info.setdefault("error", startup_error if startup_error else "Unknown startup error")
        raise RuntimeError(f"Critical startup error: {startup_error}") from e

    logger.info("Yielding control to FastAPI application...")
    yield
    logger.info("FastAPI application finished.")

    # --- Shutdown Logic ---
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down...")
    if 'utterance_context_cache' in globals():
        logger.info(f"Clearing utterance context cache ({len(globals()['utterance_context_cache'])} items)...")
        globals()['utterance_context_cache'].clear()
        logger.info("Utterance context cache cleared.")

    if 'tts_generator' in globals() and globals()['tts_generator']:
        logger.info("Releasing TTS generator instance...")
        try:
            del globals()['tts_generator']
            globals()['tts_generator'] = None
            logger.info("TTS generator instance deleted.")
        except Exception as del_err:
            logger.warning(f"Error deleting generator instance: {del_err}")

    if 'effective_device' in globals() and globals()['effective_device'] and globals()['effective_device'].startswith("cuda"):
        try:
            logger.info("Attempting to clear CUDA cache...")
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache.")
        except Exception as e:
            logger.warning(f"CUDA cache clear failed during shutdown: {e}")

    logger.info("TTS Service shutdown complete.")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan, title="TTS Streaming Service (CSM - senstella)", version="1.3.1")


# --- Helper Functions ---
# Ensure PCM_16 fix is present
def numpy_to_base64_wav(audio_np: np.ndarray, samplerate: int) -> str:
    """Converts a NumPy audio array to a base64 encoded WAV string using PCM_16."""
    if not isinstance(audio_np, np.ndarray):
        logger.warning(f"Input not NumPy array (type: {type(audio_np)}). Returning empty string.")
        return ""
    if audio_np.size == 0:
        logger.warning("Attempted encode empty NumPy array. Returning empty string.")
        return ""
    try:
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        max_val = np.max(np.abs(audio_np))
        if max_val > 1.0:
            logger.debug(f"Normalizing audio before writing WAV (max abs val: {max_val:.3f}).")
            audio_np = audio_np / max_val
        elif max_val < 1e-6:
            logger.warning(f"Audio data seems silent (max abs val: {max_val:.3e}).")

        buffer = io.BytesIO()
        # Use PCM_16 for better browser compatibility
        logger.debug(f"Writing audio to WAV buffer (PCM_16, {samplerate}Hz)...")
        sf.write(buffer, audio_np, samplerate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        wav_bytes = buffer.getvalue()
        b64_string = base64.b64encode(wav_bytes).decode('utf-8')
        logger.debug(f"Encoded {len(wav_bytes)} bytes WAV ({len(audio_np)/samplerate:.2f}s) to base64 string.")
        return b64_string
    except Exception as e:
        logger.error(f"Error encoding numpy audio to WAV base64: {e}", exc_info=True)
        return ""

# --- Main WebSocket Endpoint ---
@app.websocket("/synthesize_stream")
async def synthesize_stream_endpoint(websocket: WebSocket):
    # ... (Keep existing synthesize_stream_endpoint function as is) ...
    """Handles WebSocket for streaming TTS synthesis."""
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"
    logger.info(f"WebSocket connection request from {client_host}:{client_port} to /synthesize_stream")

    await websocket.accept()
    logger.info(f"WebSocket connection accepted for {client_host}:{client_port} on /synthesize_stream")

    if not tts_generator or model_load_info.get("status") != "loaded":
        err_msg = model_load_info.get("error", "TTS model is not ready or failed to load.")
        logger.error(f"Rejecting WebSocket connection (post-accept): {err_msg}")
        try:
            await websocket.send_json({"type": "error", "message": err_msg})
        except Exception as send_err:
             logger.warning(f"Could not send error message before closing WS: {send_err}")
        await websocket.close(code=fastapi_status.WS_1011_INTERNAL_ERROR, reason="TTS Service Not Ready")
        return

    current_utt = None
    loop = asyncio.get_running_loop()

    try:
        while True:
            try:
                raw_data = await websocket.receive_text()
                msg = json.loads(raw_data)
                logger.debug(f"Received WS message: {msg}")
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {client_host}:{client_port} disconnected.")
                break
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON over WebSocket.")
                try: await websocket.send_json({"type":"error", "message":"Invalid JSON format"})
                except: pass
                continue
            except Exception as e:
                 logger.error(f"Unexpected error receiving WebSocket message: {e}", exc_info=True)
                 try: await websocket.send_json({"type":"error", "message":f"Server error receiving message: {e}"})
                 except: pass
                 continue

            msg_type = msg.get("type")

            if msg_type == "clear_context":
                 utterance_id_to_clear = msg.get("utterance_id")
                 if utterance_id_to_clear and utterance_id_to_clear in utterance_context_cache:
                      del utterance_context_cache[utterance_id_to_clear]
                      logger.info(f"Cleared context cache for utterance_id: {utterance_id_to_clear}")
                      try: await websocket.send_json({"type":"context_cleared", "utterance_id": utterance_id_to_clear})
                      except: pass
                 else:
                      logger.warning(f"Received clear_context for unknown/missing utterance_id: {utterance_id_to_clear}")
                 continue

            if msg_type != "generate_chunk":
                logger.warning(f"Received unknown message type: {msg_type}")
                try: await websocket.send_json({"type":"error", "message":f"Unknown message type: {msg_type}"})
                except: pass
                continue

            text_chunk   = msg.get("text_chunk","").strip()
            utterance_id = msg.get("utterance_id")
            max_len_ms   = float(msg.get("max_audio_length_ms", tts_config.get("max_audio_length_ms", 90000)))

            if not text_chunk or not utterance_id:
                logger.warning(f"Missing text_chunk or utterance_id in generate_chunk message.")
                try: await websocket.send_json({"type":"error","message":"Missing text_chunk or utterance_id"})
                except: pass
                continue

            if utterance_id != current_utt:
                 if current_utt and current_utt in utterance_context_cache:
                     logger.warning(f"Switching utterance context from {current_utt} to {utterance_id} without explicit clear.")
                 current_utt = utterance_id
                 if current_utt not in utterance_context_cache:
                      utterance_context_cache[current_utt] = []
                      logger.info(f"Initialized new context cache for utterance_id: {current_utt}")
                 else:
                      logger.info(f"Reusing existing context cache for utterance_id: {current_utt}")

            context_for_csm = utterance_context_cache.get(current_utt, [])
            effective_sr    = tts_config.get("actual_sample_rate", 24000)
            temp            = float(tts_config.get("temperature", 0.9))
            topk            = int(tts_config.get("top_k", 50))
            logger.info(f"Generating chunk for utt_id={current_utt}, speaker={TTS_SPEAKER_ID}, temp={temp}, topk={topk}, context_len={len(context_for_csm)}")
            logger.debug(f"Text chunk: '{text_chunk[:50]}...'")

            def on_chunk_generated(chunk: torch.Tensor, utt_id=current_utt, txt_chunk=text_chunk):
                if not loop.is_running():
                    logger.warning(f"Event loop closed before sending chunk for {utt_id}. Skipping.")
                    return
                try:
                    if chunk is None or chunk.numel() == 0:
                         logger.warning(f"Generator produced an empty chunk for {utt_id}. Skipping send.")
                         return

                    b64 = numpy_to_base64_wav(chunk.numpy(), effective_sr)
                    if not b64:
                        logger.error(f"Failed to encode audio chunk to base64 for {utt_id}")
                        return

                    coro = websocket.send_json({
                        "type":         "audio_chunk",
                        "utterance_id": utt_id,
                        "audio_b64":    b64,
                        "text_chunk":   txt_chunk,
                    })
                    future = asyncio.run_coroutine_threadsafe(coro, loop)

                    def log_send_exception(fut):
                        try: fut.result(timeout=0)
                        except Exception as send_exc: logger.error(f"Error sending audio chunk for {utt_id} via WebSocket: {send_exc}", exc_info=False)
                    future.add_done_callback(log_send_exception)

                except Exception as e_inner:
                     logger.error(f"Error in on_chunk_generated callback for {utt_id}: {e_inner}", exc_info=True)

            generation_exception = None
            try:
                logger.debug(f"Starting generate_stream in thread for {current_utt}...")
                await asyncio.to_thread(
                    lambda: list(
                        tts_generator.generate_stream(
                            text=text_chunk, speaker=TTS_SPEAKER_ID, context=context_for_csm,
                            max_audio_length_ms=max_len_ms, temperature=temp, topk=topk,
                            on_chunk_generated=on_chunk_generated
                        )
                    )
                )
                logger.debug(f"generate_stream thread finished for {current_utt}.")

                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"type": "stream_end", "utterance_id": current_utt})
                        logger.info(f"Sent explicit stream_end for {current_utt}.")
                    else:
                        logger.warning(f"WebSocket closed before explicit stream_end could be sent for {current_utt}.")
                except Exception as send_exc:
                    logger.warning(f"Could not send stream_end message for {current_utt}: {send_exc}")

            except Exception as gen_exc:
                logger.error(f"Error during TTS generation thread for {current_utt}: {gen_exc}", exc_info=True)
                generation_exception = gen_exc

            if generation_exception:
                 try:
                     if websocket.client_state == WebSocketState.CONNECTED:
                          await websocket.send_json({"type":"error", "message": f"TTS generation failed: {generation_exception}", "utterance_id": current_utt})
                 except Exception as send_err_exc:
                      logger.error(f"Failed to send generation error message to client for {current_utt}: {send_err_exc}")

            if not generation_exception:
                try:
                    if current_utt in utterance_context_cache:
                        utterance_context_cache[current_utt].append(
                            Segment(text=text_chunk, speaker=TTS_SPEAKER_ID, audio=None)
                        )
                        logger.debug(f"Appended text segment to context for {current_utt}. New context length: {len(utterance_context_cache[current_utt])}")
                    else:
                         logger.warning(f"Context {current_utt} was cleared before text segment could be appended.")
                except Exception as append_err:
                     logger.error(f"Error appending segment to context cache for {current_utt}: {append_err}")

        logger.info(f"Exited main WebSocket loop for {client_host}:{client_port}.")

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_host}:{client_port} disconnected (caught in outer block).")
    except Exception as e:
        logger.error(f"Unhandled error in WebSocket handler for {client_host}:{client_port}: {e}", exc_info=True)
    finally:
        logger.info(f"Performing final WebSocket cleanup for {client_host}:{client_port}, last utterance: {current_utt}")
        if current_utt and current_utt in utterance_context_cache:
            try:
                del utterance_context_cache[current_utt]
                logger.info(f"Cleared context cache for final utterance: {current_utt}")
            except KeyError:
                 logger.info(f"Context cache already cleared for final utterance: {current_utt}")
            except Exception as cache_del_err:
                 logger.error(f"Error deleting context cache for {current_utt}: {cache_del_err}")

        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=fastapi_status.WS_1000_NORMAL_CLOSURE)
                logger.info(f"Closed WebSocket connection in finally block for {client_host}:{client_port}")
            except Exception as close_err:
                 logger.error(f"Error closing WebSocket in finally block: {close_err}")
        else:
            logger.info(f"WebSocket connection already closed for {client_host}:{client_port}")

# --- WebSocket Health Check Endpoint ---
@app.websocket("/ws_health")
async def websocket_health_check(websocket: WebSocket):
    """Accepts a WebSocket connection and immediately closes it."""
    await websocket.accept()
    # Optional: Log the health check connection attempt
    # logger.debug(f"WebSocket health check connection accepted from {websocket.client.host}:{websocket.client.port}")
    await websocket.close(code=fastapi_status.WS_1000_NORMAL_CLOSURE)
    # logger.debug("WebSocket health check connection closed.")

# --- HTTP Health Check Endpoint ---
@app.get("/health", status_code=fastapi_status.HTTP_200_OK)
async def health_check():
    # ... (Keep existing health_check implementation as is) ...
    current_status = model_load_info.get("status", "unknown")
    is_healthy = (current_status == "loaded" and tts_generator is not None)
    response_content = {
        "service": SERVICE_NAME, "status": "ok" if is_healthy else "error",
        "details": { "model_status": current_status,
                     "model_repo_id": tts_config.get('model_repo_id', 'N/A'),
                     "effective_device": tts_config.get('effective_device', 'N/A'),
                     "target_speaker_id": tts_config.get('effective_speaker_id', 'N/A'),
                     "actual_sample_rate": tts_config.get('actual_sample_rate', 'N/A'),
                     "generator_object_present": (tts_generator is not None),
                     "load_info": model_load_info }
    }
    status_code = fastapi_status.HTTP_200_OK if is_healthy else fastapi_status.HTTP_503_SERVICE_UNAVAILABLE

    if not is_healthy:
        logger.warning(f"Health check reporting unhealthy: Status='{current_status}', Generator Present={tts_generator is not None}")
        if model_load_info.get("error"):
             response_content["details"]["error_message"] = model_load_info.get("error")
    else:
        try:
            _ = tts_generator.device
            _ = tts_generator.sample_rate
            logger.debug(f"Health check: Generator instance accessed successfully (device: {tts_generator.device}).")
        except Exception as gen_check_err:
            logger.error(f"Health check failed accessing generator properties: {gen_check_err}", exc_info=True)
            response_content["status"] = "error"
            response_content["details"]["error_details"] = f"Generator access error: {gen_check_err}"
            model_load_info["status"] = "error_runtime_access"
            model_load_info["error"] = f"Generator access error: {gen_check_err}"
            status_code = fastapi_status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(status_code=status_code, content=response_content)

# --- Main Execution Guard ---
# ... (Keep existing __main__ block as is) ...
if __name__ == "__main__":
    logger.info(f"Starting {SERVICE_NAME.upper()} service directly via __main__...")
    main_startup_error = None
    try:
        logger.info("Running startup sequence (direct run)...")
        if not CSM_LOADED:
             main_startup_error = model_load_info.get("error", "CSM import failed.")
             raise RuntimeError(main_startup_error)
        load_configuration()
        if model_load_info.get("status") == "error":
             main_startup_error = model_load_info.get("error", "Config loading failed.")
             raise RuntimeError(main_startup_error)
        if tts_config.get('effective_device','cpu').startswith('cuda'):
            load_tts_model() # This will raise if it fails
            if model_load_info.get("status") != "loaded":
                 main_startup_error = model_load_info.get("error", "Model loading status not 'loaded'.")
                 raise RuntimeError(main_startup_error)
        else:
             main_startup_error = model_load_info.get("error", "CUDA device not configured or available.")
             model_load_info.update({"status": "error", "error": main_startup_error}) # Ensure status reflects this
             raise RuntimeError(main_startup_error)

        if tts_generator and model_load_info.get("status") == "loaded":
            logger.info("Attempting direct run warm-up...")
            try:
                tts_generator.generate(text="Ready.", speaker=tts_config.get('effective_speaker_id', 4), context=[], max_audio_length_ms=1000)
                logger.info("Direct run warm-up successful.")
            except Exception as warmup_err:
                 logger.warning(f"Direct run warm-up failed: {warmup_err}")

        port = int(os.getenv('TTS_PORT', 5003))
        host = os.getenv('TTS_HOST', "0.0.0.0")
        log_level_param = LOG_LEVEL.lower()
        logger.info(f"Direct run startup successful. Launching Uvicorn on {host}:{port}...")
        import uvicorn
        uvicorn.run("app:app", host=host, port=port, log_level=log_level_param, reload=False)

    except RuntimeError as e:
        logger.critical(f"Direct run failed during startup: {e}", exc_info=False)
        exit(1)
    except ImportError as e:
        logger.critical(f"Direct run failed: Missing dependency? {e}", exc_info=True)
        exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error during direct run: {e}", exc_info=True)
        exit(1)
    finally:
        logger.info(f"{SERVICE_NAME.upper()} Service shutting down (direct run)...")
        if 'tts_generator' in globals() and globals()['tts_generator']:
            del globals()['tts_generator']
            globals()['tts_generator'] = None
        if 'utterance_context_cache' in globals():
            globals()['utterance_context_cache'].clear()
        if 'effective_device' in globals() and globals()['effective_device'] and globals()['effective_device'].startswith("cuda"):
             try: torch.cuda.empty_cache()
             except: pass
        logger.info(f"{SERVICE_NAME.upper()} Service shutdown complete (direct run).")

# --- END OF FILE ---
