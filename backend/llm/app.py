# backend/llm/app.py
import os
import logging
import time
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from llama_cpp import Llama, LlamaGrammar # type: ignore
from huggingface_hub import hf_hub_download, login, logout
from dotenv import load_dotenv

# --- Constants & Environment Loading ---
load_dotenv()

CONFIG_PATH = os.getenv('CONFIG_PATH', '/app/config.yaml')
CACHE_DIR = Path(os.getenv('CACHE_DIR', '/cache'))
# Specific subdirectory within the cache for downloaded GGUF models
LLM_CACHE_DIR = CACHE_DIR / "llm"

LOG_FILE_BASE = os.getenv('LOG_FILE_BASE', '/app/logs/service')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
USE_GPU_ENV = os.getenv('USE_GPU', 'auto').lower()
HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

SERVICE_NAME = "llm"
LOG_PATH = f"{LOG_FILE_BASE}_{SERVICE_NAME}.log"

# --- Logging Setup ---
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
# Ensure LLM cache directory exists
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)
logger = logging.getLogger(SERVICE_NAME)

# --- Global Variables ---
llm_model: Optional[Llama] = None
llm_config: Dict[str, Any] = {}
effective_n_gpu_layers: int = 0
model_load_info: Dict[str, Any] = {"status": "pending"} # Track loading status
gguf_model_path: Optional[Path] = None # Store the resolved path to the GGUF file

# --- Configuration Loading ---
def load_configuration():
    """Loads LLM settings from the YAML config file."""
    global llm_config, effective_n_gpu_layers
    try:
        logger.info(f"Loading configuration from: {CONFIG_PATH}")
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'llm' not in config:
            raise ValueError("Config file is empty or missing 'llm' section.")

        llm_config = config['llm']
        # Validate essential LLM config keys for downloading/loading
        if not llm_config.get('model_repo_id') or not llm_config.get('model_filename'):
            raise ValueError("Missing 'model_repo_id' or 'model_filename' in llm configuration.")

        # Determine effective GPU layers based on config and environment
        config_n_gpu_layers = llm_config.get('n_gpu_layers', 0)
        logger.info(f"Configured n_gpu_layers: {config_n_gpu_layers}")
        logger.info(f"USE_GPU environment variable: '{USE_GPU_ENV}'")

        if USE_GPU_ENV == 'false':
            effective_n_gpu_layers = 0
            logger.info("GPU usage explicitly disabled via environment variable (n_gpu_layers=0).")
        elif USE_GPU_ENV == 'auto' or USE_GPU_ENV == 'true':
            effective_n_gpu_layers = config_n_gpu_layers
            if effective_n_gpu_layers != 0:
                logger.info(f"GPU usage enabled/auto. Using configured n_gpu_layers: {effective_n_gpu_layers}. Availability checked at load time.")
            else:
                logger.info("GPU usage enabled/auto, but n_gpu_layers=0. Using CPU.")
        else: # Unrecognized USE_GPU value
            effective_n_gpu_layers = 0
            logger.warning(f"Unrecognized USE_GPU value '{USE_GPU_ENV}'. Assuming CPU (n_gpu_layers=0).")

        llm_config['effective_n_gpu_layers'] = effective_n_gpu_layers # Store effective value
        logger.info(f"LLM effective n_gpu_layers set to: {effective_n_gpu_layers}")

    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Configuration error: {e}. LLM service cannot start correctly.", exc_info=True)
        llm_config = {} # Prevent partial config use
        model_load_info.update({"status": "error", "error": f"Configuration error: {e}"})
    except Exception as e:
        logger.critical(f"Unexpected error loading configuration: {e}. LLM service cannot start correctly.", exc_info=True)
        llm_config = {}
        model_load_info.update({"status": "error", "error": f"Unexpected config error: {e}"})


# --- Model Downloading ---
def download_gguf_model_if_needed() -> Optional[Path]:
    """Checks for the GGUF model file and downloads it if missing."""
    global gguf_model_path, model_load_info
    if not llm_config: # Config failed
        return None

    repo_id = llm_config['model_repo_id']
    filename = llm_config['model_filename']
    # Define the target path within our dedicated LLM cache subdir
    target_path = LLM_CACHE_DIR / filename
    gguf_model_path = target_path # Store globally for loading later

    if target_path.exists():
        logger.info(f"GGUF model '{filename}' found locally at {target_path}.")
        model_load_info["download_status"] = "cached"
        return target_path

    logger.info(f"GGUF model '{filename}' not found locally. Attempting download from repo '{repo_id}'.")
    model_load_info.update({"status": "downloading", "repo_id": repo_id, "filename": filename})
    start_time = time.monotonic()

    try:
        # Login to Hugging Face Hub if token is provided
        if HF_TOKEN:
            logger.info("Logging into Hugging Face Hub using provided token for download.")
            login(token=HF_TOKEN)

        # Download the specific file to our designated cache directory
        logger.info(f"Downloading {filename} from {repo_id} to {LLM_CACHE_DIR}...")
        downloaded_path_str = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=LLM_CACHE_DIR, # Use specific cache subdir
            local_dir=LLM_CACHE_DIR, # Force download into this dir
            local_dir_use_symlinks=False, # Avoid symlinks if problematic
            resume_download=True,
            # token=HF_TOKEN, # Handled by login()
        )
        download_time = time.monotonic() - start_time
        downloaded_path = Path(downloaded_path_str)

        # Verify download path matches expected target path after potential internal caching by hf_hub
        if downloaded_path.resolve() != target_path.resolve():
             # This case might occur if hf_hub places it in a 'snapshots' subdir within cache_dir
             logger.warning(f"Downloaded file path '{downloaded_path}' differs from target '{target_path}'. Ensuring file exists at target.")
             if not target_path.exists() and downloaded_path.exists():
                 # If target doesn't exist but download does, move it
                 target_path.parent.mkdir(parents=True, exist_ok=True)
                 downloaded_path.rename(target_path)
             elif not target_path.exists() and not downloaded_path.exists():
                  raise FileNotFoundError("Download reported success but target file not found.")
             # If target *does* exist, assume hf_hub handled it correctly (e.g., hard link or direct placement)

        logger.info(f"Successfully downloaded GGUF model '{filename}' to {target_path} in {download_time:.2f} seconds.")
        model_load_info["download_status"] = "downloaded"
        return target_path

    except Exception as e:
        logger.critical(f"FATAL: Failed to download GGUF model '{filename}' from '{repo_id}': {e}", exc_info=True)
        gguf_model_path = None # Ensure path is None on failure
        model_load_info.update({"status": "error", "download_status": "failed", "error": f"Download failed: {e}"})
        # Re-raise critical error for Fail Fast strategy
        raise RuntimeError(f"GGUF model download failed: {e}") from e
    finally:
        if HF_TOKEN:
             try: logout()
             except Exception: pass # Ignore logout errors

# --- Model Loading ---
def load_llm_model(model_path: Path):
    """Loads the GGUF model using llama-cpp-python."""
    global llm_model, model_load_info
    if not llm_config or not model_path or not model_path.exists():
        error_msg = f"Cannot load LLM model - configuration missing, model path invalid, or file not found at '{model_path}'."
        logger.error(error_msg)
        model_load_info.update({"status": "error", "error": error_msg})
        raise ValueError(error_msg) # Raise error to signal failure

    n_ctx = llm_config.get('n_ctx', 2048)
    chat_format = llm_config.get('chat_format', 'llama-3') # Default to llama-3 format
    n_gpu_layers_to_load = llm_config.get('effective_n_gpu_layers', 0)

    logger.info(f"Attempting to load LLM model from: {model_path}")
    logger.info(f"Parameters: n_gpu_layers={n_gpu_layers_to_load}, n_ctx={n_ctx}, chat_format={chat_format}")
    model_load_info["status"] = "loading_model"
    start_time = time.monotonic()

    try:
        llm_model = Llama(
            model_path=str(model_path), # llama-cpp expects string path
            n_gpu_layers=n_gpu_layers_to_load,
            n_ctx=n_ctx,
            chat_format=chat_format,
            verbose=LOG_LEVEL == 'DEBUG',
            # seed=1337, # Optional for reproducibility
            # n_batch=512, # Adjust based on VRAM/performance
        )
        load_time = time.monotonic() - start_time
        # Check actual GPU layers used after load
        actual_gpu_layers = -999 # Placeholder
        try:
            # Accessing internal context details might change between versions
            if llm_model and llm_model.ctx and hasattr(llm_model.ctx, "n_gpu_layers"):
                 actual_gpu_layers = llm_model.ctx.n_gpu_layers
            elif llm_model and hasattr(llm_model, 'model') and hasattr(llm_model.model, 'n_gpu_layers'): # Older attribute access
                 actual_gpu_layers = llm_model.model.n_gpu_layers()
        except Exception as e:
            logger.warning(f"Could not determine actual GPU layers used: {e}")


        offload_status = f"requested={n_gpu_layers_to_load}, actual={actual_gpu_layers if actual_gpu_layers != -999 else 'unknown'}"
        logger.info(f"LLM Model '{model_path.name}' loaded successfully in {load_time:.2f}s. GPU Layer Status: {offload_status}")
        model_load_info.update({"status": "loaded", "load_time_s": round(load_time, 2), "actual_gpu_layers": actual_gpu_layers if actual_gpu_layers != -999 else None})

    except Exception as e:
        logger.critical(f"FATAL: Failed to load LLM model from '{model_path}': {e}", exc_info=True)
        llm_model = None
        model_load_info.update({"status": "error", "error": f"Model load failed: {e}"})
        # Re-raise critical error for Fail Fast strategy
        raise RuntimeError(f"LLM model loading failed: {e}") from e

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Sequence
    logger.info(f"{SERVICE_NAME.upper()} Service starting up...")
    model_load_info = {"status": "initializing"}
    load_configuration() # Step 1: Load config

    if llm_config: # Proceed only if config loaded okay
        downloaded_path = None
        try:
            downloaded_path = download_gguf_model_if_needed() # Step 2: Download if needed
            if downloaded_path:
                load_llm_model(downloaded_path) # Step 3: Load model
            else:
                 # This case should be caught by exceptions in download function
                 logger.critical("Model path not available after download check, cannot load model.")
                 model_load_info.update({"status": "error", "error": "Model file unavailable after download check"})
                 raise RuntimeError("GGUF model path unavailable after download check")

        except RuntimeError as e:
            # Critical error during download or load, logged already.
            logger.critical(f"Lifespan startup failed due to critical error: {e}")
            # Let FastAPI start, healthcheck will fail
    else:
        logger.error("Skipping model download/load during startup due to config errors.")
        model_load_info = {"status": "error", "error": "Configuration failed"}

    yield # Application runs here

    # Shutdown Sequence
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down...")
    global llm_model
    if llm_model:
        logger.info("Releasing LLM model resources...")
        # Explicitly delete the object to trigger llama.cpp cleanup if implemented (__del__)
        del llm_model
        llm_model = None
        import gc
        gc.collect() # Encourage garbage collection
    logger.info("LLM Service shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan, title="LLM Service", version="1.1.0")

# --- Pydantic Models for API ---
class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class GenerateRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = Field(None, gt=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, gt=0.0, lt=1.0)
    # stream: Optional[bool] = False # Future use

class GenerateResponse(BaseModel):
    role: str = "assistant"
    content: str
    model: str # Model repo/filename used
    usage: Dict[str, int] # Token usage stats

# --- API Endpoints ---
@app.post("/generate", response_model=GenerateResponse)
async def generate_completion(request: GenerateRequest):
    """Generates chat completion using the loaded Llama GGUF model."""
    if not llm_model or model_load_info.get("status") != "loaded":
        error_detail = model_load_info.get("error", "Model not available or failed to load.")
        logger.error(f"Generation request failed: {error_detail}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"LLM model unavailable: {error_detail}")

    logger.info(f"Received generation request with {len(request.messages)} messages.")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Request messages: {[msg.model_dump() for msg in request.messages]}")

    req_start_time = time.monotonic()

    # Get generation parameters from request or config defaults
    temperature = request.temperature if request.temperature is not None else llm_config.get('temperature', 0.7)
    max_tokens = request.max_tokens if request.max_tokens is not None else llm_config.get('max_tokens', 512)
    top_p = request.top_p if request.top_p is not None else llm_config.get('top_p', 0.9)
    stream = False # For non-streaming response

    messages_dict_list = [msg.model_dump() for msg in request.messages]

    try:
        logger.info(f"Generating chat completion (temp={temperature}, max_tokens={max_tokens}, top_p={top_p})...")
        generation_start_time = time.monotonic()

        completion = llm_model.create_chat_completion(
            messages=messages_dict_list,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            # stop=["<|eot_id|>"] # Usually handled by chat_format="llama-3"
        )

        generation_time = time.monotonic() - generation_start_time
        total_req_time = time.monotonic() - req_start_time

        if not completion or 'choices' not in completion or not completion['choices']:
             logger.error("LLM generation returned empty/invalid completion object.")
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM returned empty response")

        response_message = completion['choices'][0]['message']
        response_content = response_message.get('content', '').strip()
        response_role = response_message.get('role', 'assistant')
        model_identifier = f"{llm_config.get('model_repo_id', '?')}/{llm_config.get('model_filename', '?')}"
        usage_stats = completion.get('usage', {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        logger.info(f"Generation successful in {generation_time:.3f}s. Total request time: {total_req_time:.3f}s.")
        logger.info(f"Usage - Prompt: {usage_stats.get('prompt_tokens', 0)}, Completion: {usage_stats.get('completion_tokens', 0)}, Total: {usage_stats.get('total_tokens', 0)}")
        # Limit logging long responses unless DEBUG
        logger.debug(f"Generated response content (first 100 chars): '{response_content[:100]}...'")

        return GenerateResponse(
            role=response_role,
            content=response_content,
            model=model_identifier,
            usage=usage_stats
        )

    except Exception as e:
        logger.error(f"LLM generation failed unexpectedly: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during LLM generation: {e}"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Provides LLM service health status."""
    current_status = model_load_info.get("status", "unknown")
    response_content = {
        "service": SERVICE_NAME,
        "status": "ok" if current_status == "loaded" else "error",
        "model_status": current_status,
        "model_repo_id": llm_config.get('model_repo_id', 'N/A'),
        "model_filename": llm_config.get('model_filename', 'N/A'),
        "model_file_path": str(gguf_model_path) if gguf_model_path else 'N/A',
        "gpu_layers_effective": llm_config.get('effective_n_gpu_layers', 'N/A'),
        "load_info": model_load_info # Detailed status/error/timing
    }

    if current_status != "loaded":
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=response_content)

    return response_content


# --- Main Execution Guard (for local debugging) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {SERVICE_NAME.upper()} service directly via __main__...")
    # Manually run lifespan startup steps
    logger.info("Running startup sequence...")
    model_load_info = {"status": "initializing"}
    load_configuration()
    if llm_config:
        downloaded_path = None
        try:
            downloaded_path = download_gguf_model_if_needed()
            if downloaded_path:
                load_llm_model(downloaded_path)
            else:
                logger.critical("Direct run failed: Model file unavailable after download check.")
                exit(1)
        except RuntimeError as e:
             logger.critical(f"Direct run failed: Critical error during startup: {e}")
             exit(1) # Exit if model fails in direct run
    else:
        logger.critical("Direct run failed: Configuration error.")
        exit(1)

    # Launch server
    port = int(os.getenv('LLM_PORT', 5002))
    log_level_param = LOG_LEVEL.lower()
    logger.info(f"Launching Uvicorn on port {port} with log level {log_level_param}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level=log_level_param, reload=False)
    logger.info(f"{SERVICE_NAME.upper()} Service shutting down (direct run)...")