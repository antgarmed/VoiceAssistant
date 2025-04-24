```
# Local Voice Assistant

This project provides a high-performance, local voice assistant featuring real-time transcription, LLM reasoning, text-to-speech, and a cross-platform GUI built with **Tauri 2.x** and **React**. It runs fully offline after setup, ensuring privacy, speed, and GPU acceleration. It features the new **Sesame CSM** model for expressive, local speech synthesis.

Features

- Real-time Speech-to-Text (ASR): Fast transcription using `faster-whisper`
- Local Language Model (LLM): On-device response generation using Llama 3.2 1B via `llama-cpp-python`
- Natural Text-to-Speech (TTS): Expressive output via `senstella/csm-expressiva-1b` (Sesame CSM)
- Desktop GUI: Built with **Tauri 2.5+** and React (TypeScript)
- Speaking Animations: Visual feedback when speaking
- Conversation History: Remembers context from recent turns
- GPU Acceleration: Uses CUDA for speed and performance
- Modular Backend: ASR, LLM, TTS, and orchestrator in Docker containers
- Auto Model Downloads: Models downloaded at first launch from Hugging Face

Tech Stack

- Frontend: Tauri 2.5.1, React 18+, TypeScript, Zustand, Vite
- Backend: Python 3.10, FastAPI, Uvicorn
- ASR: `faster-whisper` (large-v3)
- LLM: Llama 3.2 1B (Q4_K_M via GGUF)
- TTS: Sesame CSM fine-tune (`senstella/csm-expressiva-1b`)
- Audio Tokenizer: Mimi / Encodec
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

- Git: https://git-scm.com/downloads
- Docker & Docker Compose: https://docs.docker.com/get-docker/
- NVIDIA GPU Drivers (CUDA 12.1+): https://www.nvidia.com/Download/index.aspx
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html  
  Verify:
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
- Node.js & npm (v18+): https://nodejs.org/
- Rust & Cargo: https://www.rust-lang.org/tools/install
- Tauri CLI & dependencies: https://tauri.app/v1/guides/getting-started/prerequisites

Setup

Clone:
git clone https://github.com/ReisCook/VoiceAssistant.git voice-assistant  
cd voice-assistant

Configure:
cp .env.example .env  
Edit `.env` and add:
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

TTS Model Internals

- Backbone: Llama-1B (2048d, 16 layers, 32Q, 8KV)
- Decoder: Llama-100M (1024d, 4 layers, 8Q, 2KV)
- Audio Vocab: 2051 per codebook × 32 codebooks
- Tokenizer: Mimi / Encodec (24kHz, 75Hz)
- Voice: speaker_id=4
```
