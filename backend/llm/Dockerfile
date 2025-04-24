FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG USER=appuser
ARG GROUP=appgroup
ARG UID=1000
ARG GID=1000

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV LLM_PORT=5002
ENV CACHE_DIR=/cache
ENV CONFIG_PATH=/app/config.yaml
ENV LOG_FILE_BASE=/app/logs/service
ENV LOG_LEVEL=info
ENV USE_GPU=auto
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1
ENV HF_HOME=${CACHE_DIR}/huggingface
ENV TRANSFORMERS_CACHE=${CACHE_DIR}/huggingface
ENV PATH="/app/.venv/bin:$PATH"

RUN groupadd -g ${GID} ${GROUP} && \
    useradd -u ${UID} -g ${GID} -ms /bin/bash ${USER}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    build-essential \
    cmake \
    pkg-config \
    curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /app/.venv

RUN mkdir -p ${CACHE_DIR}/huggingface && chown ${USER}:${GROUP} ${CACHE_DIR} ${CACHE_DIR}/huggingface

COPY requirements.txt .
RUN . /app/.venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    echo "Attempting pip install with CMAKE_ARGS='${CMAKE_ARGS}' FORCE_CMAKE='${FORCE_CMAKE}'" && \
    (CMAKE_ARGS="${CMAKE_ARGS}" FORCE_CMAKE="${FORCE_CMAKE}" pip install --no-cache-dir -r /app/requirements.txt) || \
    (echo "WARNING: llama-cpp-python GPU build failed. Falling back to CPU-only build..." && \
     CMAKE_ARGS="" FORCE_CMAKE=0 pip install --no-cache-dir -r /app/requirements.txt)

COPY app.py .

RUN mkdir -p /app/logs && chown ${USER}:${GROUP} /app/logs
RUN chown -R ${USER}:${GROUP} /app

USER ${USER}

EXPOSE ${LLM_PORT}

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
  CMD curl --fail http://localhost:${LLM_PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${LLM_PORT} --log-level ${LOG_LEVEL}"]