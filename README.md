# Local Voice Assistant

This project provides a high-performance, local voice assistant featuring real-time transcription, LLM reasoning, text-to-speech, and a desktop GUI built with Tauri and React. It runs entirely on your machine (after model downloads) for privacy and speed, with GPU acceleration. It features the new Sesame CSM model for expressive, state-of-the-art local speech synthesis.

## Features

*   **Real-time Speech-to-Text (ASR):** Fast transcription using `faster-whisper`.
*   **Local Language Model (LLM):** On-device response generation using a quantized Llama 3.2 1B model via `llama-cpp-python`.
*   **Natural Text-to-Speech (TTS):** Expressive voice output using the Sesame CSM model (`senstella/csm-expressiva-1b` fine-tune).
*   **Desktop GUI:** Cross-platform user interface built with Tauri and React (TypeScript).
*   **Speaking Animations:** Visual feedback when the assistant is speaking.
*   **Conversation History:** Remembers recent turns for context.
*   **GPU Acceleration (CUDA):** Utilizes NVIDIA GPUs for significant performance improvements.
*   **Dockerized Backend:** Backend services (ASR, LLM, TTS, Orchestrator) run in Docker containers.
*   **Automatic Model Downloads:** Required AI models are downloaded on first launch.

## Tech Stack

*   **Frontend GUI:** Tauri (v1), React (v18+), TypeScript, Zustand, Vite, npm - need to update these
*   **Backend Framework:** Python (3.10), FastAPI, Uvicorn
*   **ASR Model:** `faster-whisper` (Default: `large-v3`, configurable in `shared/config.yaml`)
*   **LLM Model:** `llama-cpp-python`, Llama 3.2 1B GGUF (Default: `bartowski/Llama-3.2-1B-Instruct-GGUF`, `Q4_K_M` quantization, configurable in `shared/config.yaml`)
*   **TTS Model:** `senstella/csm-expressiva-1b` (Sesame CSM fine-tune), Mimi/Encodec audio tokenizer.
*   **Containerization:** Docker, Docker Compose
*   **Model Management:** Hugging Face Hub (`huggingface_hub`)

## Project Structure

```plaintext
C:\Users\maste\Desktop\voice-assistant\
├── .env                      # REQUIRED: Environment variables (API keys, ports)
├── .env.example              # Example environment file
├── docker-compose.yml        # Docker service orchestration
├── README.md                 # This file
│
├── backend/
│   ├── asr/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── llm/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── orchestrator/
│   │   ├── orchestrator.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── tts/
│       ├── app.py
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── models.py
│       ├── generator.py
│       └── csm_utils/
│           ├── __init__.py
│           └── loader.py
│
├── frontend/
│   ├── public/
│   │   ├── tauri.svg
│   │   └── vite.svg
│   ├── src/
│   │   ├── assets/
│   │   │   └── react.svg
│   │   ├── components/
│   │   │   ├── icons/
│   │   │   ├── MicrophoneButton.css
│   │   │   ├── MicrophoneButton.tsx
│   │   │   ├── SpeakingAnimation.css
│   │   │   ├── SpeakingAnimation.tsx
│   │   │   ├── StatusDisplay.css
│   │   │   ├── StatusDisplay.tsx
│   │   │   ├── TranscriptDisplay.css
│   │   │   └── TranscriptDisplay.tsx
│   │   ├── lib/
│   │   │   ├── api.ts
│   │   │   ├── audioUtils.ts
│   │   │   └── store.ts
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── index.css
│   │   ├── main.tsx
│   │   └── vite-env.d.ts
│   │
│   ├── src-tauri/
│   │   ├── capabilities/
│   │   │   └── default.json
│   │   ├── gen/
│   │   │   └── schemas/
│   │   │       └── (...)
│   │   ├── icons/
│   │   │   └── (...)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   └── main.rs
│   │   ├── target/          # Build artifacts (ignored by git)
│   │   ├── .gitignore
│   │   ├── build.rs
│   │   ├── Cargo.lock
│   │   ├── Cargo.toml
│   │   └── tauri.conf.json
│   │
│   ├── .gitignore
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── README.md
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
│
└── shared/
    ├── config.yaml           # Service configurations (models, params)
    ├── logs/                 # Log file output directory (mounted)
    │   ├── .keep
    │   └── service_*.log
    └── models/               # (Initially empty, models stored in cache volume)
        └── .keep




        
        
        
Prerequisites:
Ensure the following are installed and configured:

Git: (https://git-scm.com/downloads)
Docker & Docker Compose: (https://docs.docker.com/get-docker/)
NVIDIA GPU Drivers (CUDA 12.1+): (https://www.nvidia.com/Download/index.aspx)
NVIDIA Container Toolkit: (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Verify with docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi.
Node.js & npm (v18+): (https://nodejs.org/)
Rust & Cargo: (https://www.rust-lang.org/tools/install)
Tauri Build Dependencies: Follow OS-specific instructions: Tauri Prerequisites Guide.

Setup

Clone Repository:
bashgit clone https://github.com/ReisCook/VoiceAssistant.git voice-assistant
cd voice-assistant

Configure Environment:
bash# Run from the project root directory (voice-assistant)
cp .env.example .env

# --> EDIT the .env file <--
# Add your HUGGING_FACE_TOKEN. Review ports if needed.
# Required line: HUGGING_FACE_TOKEN=hf_YourActualTokenHere

Install Frontend Dependencies:
bash# Run from the project root directory (voice-assistant)
cd frontend
npm install
cd ..


Building the Backend
Build the Docker images for all backend services. Required initially and after changes.
bash# Run from the project root directory (voice-assistant)
docker compose build
(Optional) To rebuild only the TTS service without cache:
bashdocker compose build --no-cache tts
Running the Application

Start Backend Services:
Run this from the project root directory. Ensure Docker is running.
bashdocker compose up -d

Wait: Allow several minutes for services to initialize, especially TTS downloading/loading models. Check status with docker compose ps. Ensure services show 'healthy' or 'running' before proceeding.


Start Frontend Application:
Open a separate terminal, navigate to the frontend directory, and run:
bashcd frontend
npm run tauri dev

The Tauri application window should appear.


Use: Interact with the Tauri application window (microphone/input).

Stopping the Application

Stop Frontend: Close the Tauri window and press Ctrl+C in the terminal where npm run tauri dev is running.
Stop Backend: In the terminal at the project root directory, run:
bashdocker compose down
(Keeps the downloaded model cache volume).

Debugging & Logs

Follow All Backend Logs: (Press Ctrl+C to stop)
bash# Run from the project root directory
docker compose logs -f

Follow Specific Service Logs (e.g., TTS):
bash# Run from the project root directory
docker compose logs -f tts
(Use asr, llm, or orchestrator for other services)

Full Cleanup (Optional)

Stop backend and remove cached models: (Caution: Models will need re-downloading)
bash# Run from the project root directory
docker compose down -v

Remove only the model cache volume: (Volume name is likely voice-assistant_model_cache)
bash# Run from the project root directory
# Verify name first with: docker volume ls
docker volume rm voice-assistant_model_cache


TTS Model Details (Internal)
The TTS service (senstella/csm-expressiva-1b) uses the following configuration internally:

Backbone: Llama-1B (2048 dim, 16 layers, 32 query heads, 8 KV heads)
Decoder: Llama-100M (1024 dim, 4 layers, 8 query heads, 2 KV heads)
Audio Vocab Size: 2051 per codebook
Num Codebooks: 32
Audio Tokenizer: Mimi / Encodec (24kHz, 75Hz frame rate)
Speaker ID: Requires speaker_id=4 for the target voice.
