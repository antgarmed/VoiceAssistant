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
 && rm -rf /var/lib/apt/lists/*

# --- START: Added websocat installation ---
ARG WEBSOCAT_VERSION=1.13.0
ARG TARGETARCH # Docker buildx automatically sets this (e.g., amd64, arm64)

# Install wget and ca-certificates if needed for download
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && rm -rf /var/lib/apt/lists/*

# Download and install websocat based on architecture
RUN case ${TARGETARCH} in \
        amd64)   WEBSOCAT_ARCH="x86_64-unknown-linux-musl" ;; \
        arm64)   WEBSOCAT_ARCH="aarch64-unknown-linux-musl" ;; \
        # Add other architectures if needed, e.g. arm/v7 \
        *) echo "Unsupported architecture: ${TARGETARCH}"; exit 1 ;; \
    esac && \
    wget -q "https://github.com/vi/websocat/releases/download/v${WEBSOCAT_VERSION}/websocat.${WEBSOCAT_ARCH}" -O /usr/local/bin/websocat && \
    chmod +x /usr/local/bin/websocat && \
    # Verify installation (optional)
    websocat --version
# --- END: Added websocat installation ---


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
COPY csm_utils ./csm_utils
COPY generator.py models.py ./
COPY app.py .

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
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]

