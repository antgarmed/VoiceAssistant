```
# Local Voice Assistant - A functioning Sesame CSM project - Real-time factor: 0.647x with 4070 Ti Super

This project provides a high-performance, local voice assistant featuring real-time transcription, LLM reasoning, text-to-speech, and a cross-platform GUI built with **Tauri 2.x** and **React**. It runs fully offline after setup, ensuring privacy, speed, and GPU acceleration. It features the new **Sesame CSM** model for expressive, local speech synthesis.

Features

- Real-time Speech-to-Text (ASR): Fast transcription using `faster-whisper`
- Local Language Model (LLM): On-device response generation using Llama 3.2 1B via `llama-cpp-python`
- Natural Text-to-Speech (TTS): output via `senstella/csm-expressiva-1b` (Sesame CSM)
- Desktop GUI: Built with **Tauri 2.5+** and React (TypeScript)
- Speaking Animations: Visual feedback when speaking
- Conversation History: Remembers context from recent turns
- GPU Acceleration: Uses CUDA for speed and performance
- Modular Backend: ASR, LLM, TTS, and orchestrator in Docker containers
- Auto Model Downloads: Models downloaded at first launch from Hugging Face
- Real-time factor: 0.647x with 4070 Ti Super

Tech Stack

- Frontend: Tauri 2.5.1, React 18+, TypeScript, Zustand, Vite
- Backend: Python 3.10, FastAPI, Uvicorn
- ASR: `faster-whisper` (large-v3)
- LLM: Llama 3.2 1B (Q4_K_M via GGUF)
- TTS: Sesame CSM fine-tune (`senstella/csm-expressiva-1b`)
- Containerization: Docker, Docker Compose
- Model Hub: Hugging Face (`huggingface_hub`)

Project Structure

C:\Users\maste\Desktop\voice-assistant\
├── .env
├── .env.example
├── docker-compose.yml
├── README.md
│
├── backend\
│   ├── asr\
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── llm\
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── orchestrator\
│   │   ├── orchestrator.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── tts\
│       ├── app.py
│       ├── Dockerfile
│       ├── generator.py
│       ├── models.py
│       ├── requirements.txt
│       └── csm_utils\
│           ├── __init__.py
│           └── loader.py
│
├── frontend\
│   ├── .gitignore
│   ├── .vscode\
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── README.md
│   ├── node_modules\
│   ├── public\
│   │   ├── tauri.svg
│   │   └── vite.svg
│   ├── src\
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── index.css
│   │   ├── main.tsx
│   │   ├── vite-env.d.ts
│   │   ├── assets\
│   │   ├── components\
│   │   └── lib\
│   └── src-tauri\
│       ├── .gitignore
│       ├── build.rs
│       ├── Cargo.lock
│       ├── Cargo.toml
│       ├── tauri.conf.json
│       ├── capabilities\
│       │   └── default.json
│       ├── gen\
│       │   └── schemas\
│       ├── icons\
│       │   ├── (all required .ico/.png files)
│       ├── src\
│       │   ├── lib.rs
│       │   └── main.rs
│       └── target\
│           ├── debug\
│           ├── .rustc_info.json
│           └── CACHEDIR.TAG
│
├── shared\
│   ├── config.yaml
│   ├── logs\
│   └── models\

Prerequisites

- Docker & Docker Compose: https://docs.docker.com/get-docker/
- NVIDIA GPU Drivers (CUDA 12.1+): https://www.nvidia.com/Download/index.aspx
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html  
  Verify:
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
- Node.js & npm (v18+): https://nodejs.org/
- Rust & Cargo: https://www.rust-lang.org/tools/install
- Tauri CLI & dependencies: https://tauri.app/v1/guides/getting-started/prerequisites or cargo install tauri-cli

Min specs:
GPU: Nvidia with 8GB+ VRAM CUDA Version 12.1.1+
RAM: 32 GB RAM
OS: Windows or Linux

Docker Desktop must be installed and running
Llama 3.2 1b is a gated repo, you must request and be given access to acquire it from huggingface

You will need these if you don't have them already:
npm install
cargo install tauri-cli


TLDR Setup:
Install and run docker
Request access to Llama 3.2 1b and acquire huggingface key (then replace in .env file)
Install dependencies (rust, Tauri, Nvidia Container Toolkit)
Build backend then run, start frontend 

Setup

Configure: 
Edit `.env` and change:
HUGGING_FACE_TOKEN=hf_yourTokenHere

Install Frontend:
cd frontend  
npm install  
cd ..

Build Backend:
docker compose build  
# Optional: rebuild TTS only
docker compose build --no-cache tts

Run the App

Start Backend:
docker compose up -d  
Check with:
docker compose ps

Start Frontend:
cd frontend  
npm run tauri dev

Stop the App

Close the frontend window or press Ctrl+C  
Stop backend:
docker compose down

Logs

All:
docker compose logs -f

One service:
docker compose logs -f tts

Cleanup (Optional)

Remove all:
docker compose down -v

Just model cache:
docker volume ls  
docker volume rm voice-assistant_model_cache

Notes:

TTS Model Internals

- Backbone: Llama-1B (2048d, 16 layers, 32Q, 8KV)
- Decoder: Llama-100M (1024d, 4 layers, 8Q, 2KV)
- Audio Vocab: 2051 per codebook × 32 codebooks
- Tokenizer: Mimi / Encodec (24kHz, 75Hz)


Quick Commands:

docker compose down -v
docker compose build --no-cache tts
docker compose up -d
npm run tauri dev
docker compose logs -f


Issue 1 (fixed):

TTS generation failure: Tensor dimension mismatch during backbone forward pass (expanded size 2048 != existing size 16 at dim 3);

Proposed Fix:

projected_c0_embed → .squeeze(1) to drop the extra sequence dim.

The same squeeze added to next_decoder_input_embed inside the loop.

These tweaks make the tensors [1, 1024] and [1, 1024] before stacking, eliminating the stack expects each tensor to be equal size runtime error.

Issue 2 (fixed):

Problem: The KV cache tensors (k_cache, v_cache) allocated during model setup were too small.

Cause: Torchtune's setup_caches function ignored the requested max_seq_len parameter, likely defaulting to the model's original (shorter) pre-training sequence length.

Effect: During generation, when the current position (cache_pos) plus the sequence length being added (seq_len, even if just 1) exceeded the small allocated capacity (k_cache.shape[2]), an AssertionError occurred within the cache update logic.

Issue 3 (fixed):

The system requests only ~5000ms of audio max for this short prompt (max_audio_length_ms=5000 set by orchestrator).

The CSM model's backbone expects a certain minimum prompt+generation length to work effectively.

The WebSocket expects at least one audio_chunk or stream_end to be sent back to the client.
But your TTS server never sends either.

Thus, the orchestrator and client hang waiting for either audio or stream end, until timeout.

The system is correctly architected. The error lies in an unguarded edge-case when CSM fails to generate meaningful audio quickly for very short prompts.
