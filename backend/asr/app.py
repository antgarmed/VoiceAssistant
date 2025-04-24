# backend/asr/app.py
import os
import logging
import io
import time
import numpy as np
import soundfile as sf
import torch
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

# --- ADD Pydub IMPORT ---
try:
    from pydub import AudioSegment
    pydub_available = True
except ImportError:
    logging.warning("pydub library not found. Audio conversion will not be available.")
    pydub_available = False
# --- END ADD Pydub IMPORT ---

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from huggingface_hub import login, logout
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# --- Constants & Environment Loading ---
load_dotenv()
CONFIG_PATH = os.getenv('CONFIG_PATH', '/app/config.yaml')
CACHE_DIR = Path(os.getenv('CACHE_DIR', '/cache'))
HF_CACHE_DIR = Path(os.getenv('HF_HOME', CACHE_DIR / "huggingface"))
LOG_FILE_BASE = os.getenv('LOG_FILE_BASE', '/app/logs/service')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
USE_GPU_ENV = os.getenv('USE_GPU', 'auto').lower()
HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
SERVICE_NAME = "asr"
LOG_PATH = f"{LOG_FILE_BASE}_{SERVICE_NAME}.log"

# --- Logging Setup ---
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig( level=LOG_LEVEL, format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s", handlers=[ logging.StreamHandler(), logging.FileHandler(LOG_PATH) ])
logger = logging.getLogger(SERVICE_NAME)

# --- Global Variables ---
asr_model: Optional[WhisperModel] = None
asr_config: Dict[str, Any] = {}
effective_device: str = "cpu"
model_load_info: Dict[str, Any] = {"status": "pending"}

# --- Configuration Loading ---
def load_configuration():
    global asr_config, effective_device
    try:
        logger.info(f"Loading configuration from: {CONFIG_PATH}")
        if not os.path.exists(CONFIG_PATH): raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)
        if not config or 'asr' not in config: raise ValueError("Config file is empty or missing 'asr' section.")
        asr_config = config['asr']
        if not asr_config.get('model_name'): raise ValueError("Missing 'model_name' in asr configuration.")
        config_device = asr_config.get('device', 'auto')
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}, Torch version: {torch.__version__}")
        logger.info(f"USE_GPU environment variable: '{USE_GPU_ENV}'")
        logger.info(f"Configured ASR device: '{config_device}'")
        if USE_GPU_ENV == 'false': effective_device = "cpu"; logger.info("GPU usage explicitly disabled via environment variable.")
        elif config_device == "cpu": effective_device = "cpu"; logger.info("ASR device configured to CPU.")
        elif cuda_available and (config_device == "auto" or config_device.startswith("cuda")): effective_device = config_device if config_device.startswith("cuda") else "cuda"; logger.info(f"Attempting to use CUDA device '{effective_device}' for ASR.")
        else: effective_device = "cpu"; logger.warning(f"CUDA device '{config_device}' requested but not available or USE_GPU=false. Falling back to CPU.") if (config_device == "auto" or config_device.startswith("cuda")) and USE_GPU_ENV != 'false' else logger.info("Using CPU for ASR.")
        asr_config['effective_device'] = effective_device
        logger.info(f"ASR effective device set to: {effective_device}")
    except (FileNotFoundError, ValueError) as e: logger.critical(f"Configuration error: {e}. ASR service cannot start correctly.", exc_info=True); asr_config = {}; model_load_info.update({"status": "error", "error": f"Configuration error: {e}"})
    except Exception as e: logger.critical(f"Unexpected error loading configuration: {e}. ASR service cannot start correctly.", exc_info=True); asr_config = {}; model_load_info.update({"status": "error", "error": f"Unexpected config error: {e}"})

# --- Model Loading / Downloading ---
def load_asr_model():
    global asr_model, model_load_info
    if not asr_config: logger.error("Skipping model load due to configuration errors."); return
    model_name = asr_config.get('model_name'); compute_type = asr_config.get('compute_type', 'int8'); device_to_load = asr_config.get('effective_device', 'cpu'); cache_path = HF_CACHE_DIR
    logger.info(f"Attempting to load/download ASR model: {model_name}"); logger.info(f"Target device: {device_to_load}, Compute type: {compute_type}"); logger.info(f"Using cache directory (HF_HOME): {cache_path}")
    model_load_info = {"status": "loading", "model_name": model_name, "device": device_to_load, "compute_type": compute_type}; start_time = time.monotonic()
    try:
        if HF_TOKEN: logger.info("Logging into Hugging Face Hub using provided token."); login(token=HF_TOKEN)
        asr_model = WhisperModel( model_name, device=device_to_load, compute_type=compute_type, download_root=str(cache_path))
        load_time = time.monotonic() - start_time; model_load_info.update({"status": "loaded", "load_time_s": round(load_time, 2)}); logger.info(f"ASR Model '{model_name}' loaded successfully in {load_time:.2f} seconds.")
    except Exception as e: logger.critical(f"FATAL: Failed to load or download ASR model '{model_name}': {e}", exc_info=True); asr_model = None; load_time = time.monotonic() - start_time; model_load_info.update({"status": "error", "error": str(e), "load_time_s": round(load_time, 2)}); raise RuntimeError(f"ASR model loading failed: {e}") from e
    finally:
        if HF_TOKEN:
            try: logout(); logger.info("Logged out from Hugging Face Hub.")
            except Exception as logout_err: logger.warning(f"Could not log out from Hugging Face Hub: {logout_err}")

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{SERVICE_NAME.upper()} Service starting up..."); model_load_info = {"status": "initializing"}
    load_configuration()
    if asr_config:
        try: load_asr_model()
        except RuntimeError as e: logger.critical(f"Lifespan startup failed due to model load error: {e}")
    else: logger.error("Skipping model load during startup due to config errors."); model_load_info = {"status": "error", "error": "Configuration failed"}
    yield
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down..."); global asr_model
    if asr_model: logger.info("Releasing ASR model resources..."); del asr_model; asr_model = None
    if effective_device.startswith("cuda"):
        try: torch.cuda.empty_cache(); logger.info("Cleared PyTorch CUDA cache.")
        except Exception as e: logger.warning(f"Could not clear CUDA cache during shutdown: {e}")
    logger.info("ASR Service shutdown complete.")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan, title="ASR Service", version="1.1.0")

# --- Audio Preprocessing Helper ---
async def preprocess_audio(audio_bytes: bytes, filename: str) -> np.ndarray:
    """Reads audio bytes, converts to WAV using pydub if necessary, then mono float32 numpy array at 16kHz."""
    logger.debug(f"Preprocessing audio from '{filename}' ({len(audio_bytes)} bytes)")
    start_time = time.monotonic()
    input_stream = io.BytesIO(audio_bytes)
    processed_stream = input_stream # Start with the original stream

    # --- CONVERSION STEP using pydub ---
    if pydub_available:
        try:
            logger.debug("Attempting to load audio with pydub...")
            # Load audio segment using pydub, format='*' tells it to detect
            audio_segment = AudioSegment.from_file(input_stream)

            # Ensure minimum frame rate and export to WAV format in memory
            # Whisper expects 16kHz, so set frame rate here if conversion is happening
            if audio_segment.frame_rate < 16000:
                 logger.warning(f"Input audio frame rate {audio_segment.frame_rate}Hz is low, setting to 16000Hz during conversion.")
                 audio_segment = audio_segment.set_frame_rate(16000)

            wav_stream = io.BytesIO()
            audio_segment.export(wav_stream, format="wav")
            wav_stream.seek(0) # Rewind stream to the beginning
            processed_stream = wav_stream # Use the converted WAV stream
            logger.info(f"Successfully converted audio '{filename}' to WAV format using pydub.")

        except Exception as pydub_err:
             logger.warning(f"Pydub failed to load/convert '{filename}': {pydub_err}. Falling back to soundfile with original data.", exc_info=True)
             # Reset stream to original bytes if pydub fails
             processed_stream = io.BytesIO(audio_bytes)
             processed_stream.seek(0) # Ensure stream is at the beginning
    else:
         logger.warning("Pydub not available, attempting direct load with soundfile.")
    # --- END CONVERSION STEP ---

    try:
        # Now read using soundfile (should be WAV data if conversion succeeded)
        audio_data, samplerate = sf.read(processed_stream, dtype='float32', always_2d=True)
        logger.debug(f"Read audio via soundfile: SR={samplerate}Hz, Shape={audio_data.shape}, Duration={audio_data.shape[0]/samplerate:.2f}s")

        # Convert to mono by averaging channels if stereo
        if audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
            logger.debug(f"Converted stereo to mono. New shape: {audio_data.shape}")
        else:
            audio_data = audio_data[:, 0] # Squeeze mono channel dim

        # Resample to 16kHz if necessary (e.g., if pydub wasn't used or original WAV wasn't 16k)
        if samplerate != 16000:
            logger.info(f"Resampling audio from {samplerate}Hz to 16000Hz using torchaudio...")
            try:
                import torchaudio
                import torchaudio.transforms as T
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                resampler = T.Resample(orig_freq=samplerate, new_freq=16000).to('cpu')
                resampled_tensor = resampler(audio_tensor.cpu())
                audio_data = resampled_tensor.squeeze(0).numpy()
                logger.info(f"Resampled audio to 16kHz. New shape: {audio_data.shape}")
                samplerate = 16000
            except ImportError:
                logger.error("Torchaudio is required for resampling but not found/installed.")
                raise HTTPException(status_code=501, detail="Audio resampling required but torchaudio not available.")
            except Exception as resample_err:
                logger.error(f"Error during resampling: {resample_err}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to resample audio: {resample_err}")

        if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)

        preprocess_time = time.monotonic() - start_time
        logger.debug(f"Audio preprocessing completed in {preprocess_time:.3f} seconds.")
        return audio_data

    except sf.SoundFileError as sf_err:
        # This error likely means the original format was bad OR pydub failed AND the original was also bad
        logger.error(f"Soundfile error processing audio '{filename}' after potential conversion attempt: {sf_err}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not read or decode audio file: {sf_err}")
    except Exception as e:
        logger.error(f"Unexpected error preprocessing audio '{filename}' after potential conversion: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error processing audio: {e}")


# --- API Endpoints ---
@app.post("/transcribe")
async def transcribe_audio_endpoint(audio: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, WebM, Ogg, etc.)")):
    if not asr_model or model_load_info.get("status") != "loaded":
        error_detail = model_load_info.get("error", "Model not available or failed to load.")
        logger.error(f"Transcription request failed: {error_detail}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"ASR model unavailable: {error_detail}")

    req_start_time = time.monotonic()
    logger.info(f"Received transcription request for file: {audio.filename} ({audio.content_type})")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Received empty audio file.")

        # Preprocess audio: Handles conversion, mono, 16kHz, float32 numpy array
        audio_np = await preprocess_audio(audio_bytes, audio.filename or "uploaded_audio")
        audio_duration_sec = len(audio_np) / 16000.0

        beam_size = asr_config.get('beam_size', 5)
        language_code = asr_config.get('language', None) # Let Whisper detect by default unless specified
        vad_filter = asr_config.get('vad_filter', True)
        vad_parameters = asr_config.get('vad_parameters', {"threshold": 0.5})

        logger.info(f"Starting transcription (beam={beam_size}, lang={language_code or 'auto'}, vad={vad_filter})...")
        transcribe_start_time = time.monotonic()

        segments_generator, info = asr_model.transcribe( audio_np, beam_size=beam_size, language=language_code, vad_filter=vad_filter, vad_parameters=vad_parameters)

        transcribed_text_parts = []; segment_count = 0
        try:
            for segment in segments_generator: transcribed_text_parts.append(segment.text); segment_count += 1
        except Exception as seg_err: logger.error(f"Error processing transcription segment: {seg_err}", exc_info=True)

        transcribed_text = " ".join(transcribed_text_parts).strip()
        transcribe_time = time.monotonic() - transcribe_start_time
        total_req_time = time.monotonic() - req_start_time

        logger.info(f"Transcription completed in {transcribe_time:.3f}s ({segment_count} segments). Total request time: {total_req_time:.3f}s.")
        logger.info(f"Detected lang: {info.language} (Prob: {info.language_probability:.2f}), Audio duration: {info.duration:.2f}s (processed: {audio_duration_sec:.2f}s)")
        if len(transcribed_text) < 200: logger.debug(f"Transcription result: '{transcribed_text}'")
        else: logger.debug(f"Transcription result (truncated): '{transcribed_text[:100]}...{transcribed_text[-100:]}'")

        return JSONResponse( status_code=status.HTTP_200_OK, content={ "text": transcribed_text, "language": info.language, "language_probability": info.language_probability, "audio_duration_ms": round(info.duration * 1000), "processing_time_ms": round(total_req_time * 1000), "transcription_time_ms": round(transcribe_time * 1000), })

    except HTTPException as http_exc: logger.warning(f"HTTP exception during transcription: {http_exc.status_code} - {http_exc.detail}"); raise http_exc
    except Exception as e: logger.error(f"Unexpected error during transcription request for {audio.filename}: {e}", exc_info=True); raise HTTPException( status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    current_status = model_load_info.get("status", "unknown")
    response_content = { "service": SERVICE_NAME, "status": "ok" if current_status == "loaded" else "error", "model_status": current_status, "model_name": model_load_info.get("model_name", asr_config.get('model_name', 'N/A')), "device": model_load_info.get("device", effective_device), "compute_type": model_load_info.get("compute_type", asr_config.get('compute_type', 'N/A')), "load_info": model_load_info }
    if current_status != "loaded": return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=response_content)
    return response_content

# --- Main Execution Guard (for local debugging) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {SERVICE_NAME.upper()} service directly via __main__...")
    logger.info("Running startup sequence...")
    model_load_info = {"status": "initializing"}
    load_configuration()
    if asr_config:
        try: load_asr_model()
        except RuntimeError as e: logger.critical(f"Direct run failed: Model load error: {e}"); exit(1)
    else: logger.critical("Direct run failed: Configuration error."); exit(1)
    port = int(os.getenv('ASR_PORT', 5001)); log_level_param = LOG_LEVEL.lower()
    logger.info(f"Launching Uvicorn on port {port} with log level {log_level_param}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level=log_level_param, reload=False)
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down (direct run)...")