# backend/tts/Dockerfile
ARG BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.10
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=appuser
ARG GROUPNAME=appgroup

WORKDIR /app

# Create non-root user and group
RUN groupadd -g ${GROUP_ID} ${GROUPNAME} && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -ms /bin/bash ${USERNAME}

# Install Python, venv, pip, essential build tools, git, AND curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/* # Added curl here

# Create and activate virtual environment
ENV VENV_PATH=/app/.venv
RUN python${PYTHON_VERSION} -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"

# Set cache directory for Hugging Face (and potentially pip)
ENV HF_HOME=/cache/huggingface
ENV PIP_CACHE_DIR=/cache/pip
RUN mkdir -p ${HF_HOME} ${PIP_CACHE_DIR} && \
    chown ${USERNAME}:${GROUPNAME} /cache ${HF_HOME} ${PIP_CACHE_DIR}

# Install FFmpeg libraries needed for PyAV (av package)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN . ${VENV_PATH}/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
# Ensure these paths match your project structure relative to backend/tts
COPY csm_utils ./csm_utils
COPY generator.py models.py ./
COPY app.py .
# COPY config.yaml . # Optional: copy default config if needed, volume mount usually preferred

# Create logs directory and set permissions
ENV LOG_DIR=/app/logs
RUN mkdir -p ${LOG_DIR} && chown ${USERNAME}:${GROUPNAME} ${LOG_DIR}

# Change ownership of the app directory
RUN chown -R ${USERNAME}:${GROUPNAME} /app

# Switch to non-root user
USER ${USERNAME}:${GROUPNAME}

# Expose port (will be mapped by docker-compose)
EXPOSE 5003

# Default command (can be overridden by docker-compose)
# Ensure HF_TOKEN is passed as env var by docker-compose
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]