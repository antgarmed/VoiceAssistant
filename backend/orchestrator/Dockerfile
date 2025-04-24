FROM python:3.10-slim

ARG PYTHON_VERSION=3.10
ARG USER=appuser
ARG GROUP=appgroup
ARG UID=1000
ARG GID=1000

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV ORCHESTRATOR_PORT=5000
ENV ASR_SERVICE_URL="http://asr:5001"
ENV LLM_SERVICE_URL="http://llm:5002"
ENV TTS_SERVICE_URL="http://tts:5003"
ENV CONFIG_PATH=/app/config.yaml
ENV LOG_FILE_BASE=/app/logs/service
ENV LOG_LEVEL=info
ENV PATH="/app/.venv/bin:$PATH"

RUN groupadd -g ${GID} ${GROUP} && \
    useradd -u ${UID} -g ${GID} -ms /bin/bash ${USER}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/.venv

COPY requirements.txt .
RUN . /app/.venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY orchestrator.py .

RUN mkdir -p /app/logs && chown ${USER}:${GROUP} /app/logs
RUN chown -R ${USER}:${GROUP} /app

USER ${USER}

EXPOSE ${ORCHESTRATOR_PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:${ORCHESTRATOR_PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn orchestrator:app --host 0.0.0.0 --port ${ORCHESTRATOR_PORT} --log-level ${LOG_LEVEL}"]