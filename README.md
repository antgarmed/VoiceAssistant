# Local Voice Assistant (Astra) - Need to change then name

# Need to update this document, this program is not capable of ASR yet, files have been added/removed, and the terminal commands need to be listed here.

A high-performance, local-first voice assistant featuring real-time transcription, LLM reasoning, text-to-speech, and a polished desktop GUI built with Tauri and React. This project runs entirely on your local machine (after initial model downloads) for enhanced privacy and speed, leveraging GPU acceleration where possible.


<!-- (Add a screenshot or GIF showing the application interface!!!) -->

## Key Features

*   **Real-time Speech-to-Text (ASR):** Fast and accurate transcription using `faster-whisper`.
*   **Local Language Model (LLM):** On-device response generation using a quantized Llama 3.2 1B model via `llama-cpp-python`.
*   **Natural Text-to-Speech (TTS):** Expressive voice output using a fine-tuned Sesame CSM model.
*   **Desktop GUI:** Cross-platform user interface built with Tauri and React (TypeScript).
*   **Speaking Animations:** Visual feedback when the assistant is speaking.
*   **Conversation History:** Remembers the last few turns of the conversation for context.
*   **GPU Acceleration (CUDA):** Utilizes NVIDIA GPUs for significant performance improvements in ASR, LLM, and especially TTS.
*   **Dockerized Backend:** Backend services (ASR, LLM, TTS, Orchestrator) are isolated in Docker containers for easy setup and dependency management.
*   **Automatic Model Downloads:** Required AI models are automatically downloaded on first launch if not found locally.

## Tech Stack

*   **Frontend GUI:** Tauri (v1), React (v18+), TypeScript, Zustand (State Management), Vite
*   **Backend Framework:** Python (3.10+), FastAPI, Uvicorn
*   **ASR Model:** `faster-whisper` (Default: `large-v3`)
*   **LLM Model:** `llama-cpp-python`, Llama 3.2 1B GGUF (Default: `bartowski/Llama-3.2-1B-Instruct-GGUF`, `Q4_K_M` quantization)
*   **TTS Model:** Sesame CSM (Default Fine-tune: `senstella/csm-expressiva-1b`)
*   **Containerization:** Docker, Docker Compose
*   **Model Management:** Hugging Face Hub (`huggingface_hub`)

## Project Structure

```plaintext
voice-assistant/
├── .env                      # Environment variables (API keys, ports, config)
├── docker-compose.yml        # Docker service orchestration
│
├── backend/                  # Python backend services
│   ├── asr/                  # Speech-to-Text Service
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── llm/                  # Language Model Service
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── orchestrator/         # Central control service
│   │   ├── orchestrator.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── tts/                  # Text-to-Speech Service
│       ├── app.py
│       ├── Dockerfile
│       ├── requirements.txt
│       └── csm_utils/        # (Optional CSM helpers)
│           └── ...
│
├── frontend/                 # Tauri + React frontend application
│   ├── public/               # Static assets
│   ├── src/                  # React source code
│   │   ├── components/       # React UI components
│   │   ├── lib/              # Stores, API client, utils
│   │   ├── App.tsx           # Main app component
│   │   ├── main.tsx          # React entry point
│   │   └── index.css         # Global styles
│   │
│   ├── src-tauri/            # Tauri Rust backend source
│   │   ├── icons/            # Application icons
│   │   ├── src/main.rs       # Tauri Rust entry point (usually default)
│   │   ├── Cargo.toml        # Rust dependencies
│   │   └── tauri.conf.json   # Tauri configuration (permissions, window)
│   │
│   ├── package.json          # Node.js dependencies
│   ├── tsconfig.json         # TypeScript config
│   └── vite.config.ts        # Vite config
│
└── shared/                   # Shared resources (config, logs)
    ├── config.yaml           # Service configurations (models, params)
    ├── logs/                 # Log file output directory (mounted)
    └── models/               # (Initially empty, models stored in cache volume)


    Voice Assistant Terminal Commands:

Tauri:
Run server:		npm run tauri dev
Docker:
Build:			docker compose build --no-cache tts
Start/restart:		docker compose up -d --force-recreate tts
Logging:		docker compose logs -f tts

Sesame CSM Model Info:
Backbone: llama-1B, Dim: 2048
Decoder: llama-100M, Dim: 1024
Audio tokenizer: Mimi/Encodec

Text Embeddings: 128256 -> 2048
Audio Embeddings: 65632 (32x2051) -> 2048
Codebook0 Head Out Dim: 2051	
Audio Head Shape: torch.Size([31, 1024, 2051])
Backbone max_seq_len: 2048
Decoder max_seq_len: 2048
Num Codebooks: 32, Audio Vocab Size: 2051
Pre-computed backbone mask: torch.Size([2048, 2048]), device=cuda:0
Pre-computed decoder mask: torch.Size([2048, 2048]), device=cuda:0


