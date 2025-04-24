# backend/asr/Dockerfile
# --- Use CUDA base image that includes cuDNN 8 ---
    FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

    ARG PYTHON_VERSION=3.10
    ARG USER=appuser
    ARG GROUP=appgroup
    ARG UID=1000
    ARG GID=1000
    
    WORKDIR /app
    
    ENV PYTHONUNBUFFERED=1
    ENV DEBIAN_FRONTEND=noninteractive
    ENV ASR_PORT=5001
    ENV CACHE_DIR=/cache
    ENV CONFIG_PATH=/app/config.yaml
    ENV LOG_FILE_BASE=/app/logs/service
    ENV LOG_LEVEL=info
    ENV USE_GPU=auto
    ENV HF_HOME=${CACHE_DIR}/huggingface
    ENV TRANSFORMERS_CACHE=${CACHE_DIR}/huggingface
    ENV PATH="/app/.venv/bin:$PATH"
    
    # Create non-root user/group
    RUN groupadd -g ${GID} ${GROUP} && \
        useradd -u ${UID} -g ${GID} -ms /bin/bash ${USER}
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        build-essential \
        libsndfile1 \
        ffmpeg \
        curl \
     && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Set up Python virtual environment
    RUN python${PYTHON_VERSION} -m venv /app/.venv
    
    # Create cache directory and set permissions
    RUN mkdir -p ${CACHE_DIR}/huggingface && chown ${USER}:${GROUP} ${CACHE_DIR} ${CACHE_DIR}/huggingface
    
    # Copy requirements and install Python packages into venv
    COPY requirements.txt .
    RUN . /app/.venv/bin/activate && \
        pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt
    
    # Copy application code
    COPY app.py .
    
    # Create log directory and set permissions for app directory
    RUN mkdir -p /app/logs && chown ${USER}:${GROUP} /app/logs
    RUN chown -R ${USER}:${GROUP} /app
    
    # Switch to non-root user
    USER ${USER}
    
    # Expose the application port
    EXPOSE ${ASR_PORT}
    
    # Define health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
      CMD curl --fail http://localhost:${ASR_PORT}/health || exit 1
    
    # Default command to run the application
    CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${ASR_PORT} --log-level ${LOG_LEVEL}"]