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
- **Models**: `distil-whisper` (large-v3.5), Llama 3.2 1B (GGUF), Sesame CSM

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
- Add your huggingface token and request access to the models (need to add links)
- Build backend: `docker compose build`
- Start backend: `docker compose up -d`
- Build frontend: `npm install && npm install uuid`
- Start frontend: `cd frontend && npm run tauri dev`
- View logs: `docker compose logs -f`
- Stop: `docker compose down`

## Architecture

Overview:

The project is split into several independent services orchestrated with Docker Compose. Each service handles a layer of the pipeline: voice input -> transcription -> reasoning -> speech output.

- asr: Speech-to-Text service exposing HTTP endpoints `/transcribe` and `/health`.
- llm: Language model service exposing `/generate` and `/health`.
- tts: Text-to-Speech service with an HTTP `/health` endpoint and a WebSocket `/synthesize_stream` endpoint for streaming audio.
- orchestrator: Coordinator service that accepts audio from the frontend at `/assist`, calls ASR and LLM, manages conversation history, and returns assistant text (no audio). Audio playback is handled by the frontend connecting directly to TTS.

Ports and networking:

- Ports are configured via environment variables in `docker-compose.yml` (e.g. `ASR_PORT`, `LLM_PORT`, `TTS_PORT`, `ORCHESTRATOR_PORT`).
- All services connect to the Docker network `voice-assistant-net` defined in `docker-compose.yml`.
- The `model_cache` volume is shared for local model storage.

Data flow (simplified):

1. The frontend records audio and POSTs it to the Orchestrator `/assist` (multipart/form-data).
2. The Orchestrator forwards the audio to the ASR service `/transcribe` and receives the transcript.
3. If the transcript is empty, the Orchestrator may call the TTS WebSocket for a short "no-speech" audio response; in the normal flow the Orchestrator builds the history and calls the LLM `/generate`.
4. The Orchestrator updates conversation history and returns the assistant's text to the frontend.
5. The frontend connects to the TTS service (WebSocket `/synthesize_stream`) to synthesize and play the assistant's response.

Key files and locations:

- `docker-compose.yml`: service, network, volume and environment variable definitions.
- `backend/asr/`: ASR service code and Dockerfile.
- `backend/llm/`: LLM service code and Dockerfile.
- `backend/tts/`: TTS service code, `generator.py`, and model utilities (includes `csm_utils/`).
- `backend/orchestrator/orchestrator.py`: orchestration logic, `/assist` handling, calls to ASR/LLM/TTS, and history management.
- `frontend/`: Tauri + React application that records audio, displays UI, and connects to TTS for audio playback.
- `shared/config.yaml`: central configuration used by containers.

Operational notes:

- TTS uses a WebSocket for audio streaming, which is why the frontend connects directly for real-time playback.
- LLM and TTS services are designed to use GPU (configurable via `USE_GPU`, `NVIDIA_VISIBLE_DEVICES`, and docker runtime options in `docker-compose.yml`).
- Make sure to set `HUGGING_FACE_TOKEN` and have access to required models (e.g. Llama 3.2 1B and `senstella/csm-expressiva-1b`) before starting the services.

