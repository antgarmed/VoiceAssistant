# Sesame CSM Voice Assistant

## Overview
A high-performance, local voice assistant with real-time transcription, LLM reasoning, and text-to-speech. Runs fully offline after setup and features Sesame CSM for expressive speech synthesis. Real-time factor: 0.6x with NVIDIA 4070 Ti Super.

## Features
- Real-time Speech-to-Text using `distil-whisper`
- On-device LLM using Llama 3.2 1B 
- Natural TTS via Sesame CSM (`senstella/csm-expressiva-1b`)
- Desktop GUI with Tauri/React
- Conversation history and speaking animations
- GPU acceleration with CUDA
- Modular Docker-based backend

## Tech Stack
- **Frontend**: Tauri 2.5.1, React 18+, TypeScript
- **Backend**: Python 3.10, FastAPI, Uvicorn
- **Models**: `faster-whisper` (large-v3), Llama 3.2 1B (GGUF), Sesame CSM

## Requirements
- NVIDIA GPU: 8GB+ VRAM
- 32GB RAM
- Docker Desktop
- NVIDIA GPU Drivers (CUDA 12.1+)
- NVIDIA Container Toolkit
- Node.js & npm (v18+)
- Rust & Cargo
- Hugging Face access to Llama 3.2 1B

## Setup
1. **Prerequisites**:
   - Install Docker Desktop and ensure it's running
   - Install Rust, Tauri, and NVIDIA Container Toolkit
   - Request access to Llama 3.2 1B on Hugging Face

2. **Configuration**:
   - Edit `.env` file and set `HUGGING_FACE_TOKEN=hf_yourTokenHere`

3. **Backend**:
   - Build: `docker compose build`
   - Run: `docker compose up -d`

4. **Frontend**:
   - Install dependencies: `cd frontend && npm install && npm install uuid`
   - Start: `npm run tauri dev`

## Usage
- Start backend: `docker compose up -d`
- Start frontend: `cd frontend && npm run tauri dev`
- View logs: `docker compose logs -f`
- Stop: `docker compose down`



## Project Structure

├── .env
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



## Quick Commands


docker compose down -v
docker compose build --no-cache tts
docker compose up -d
npm run tauri dev
docker compose logs -f


