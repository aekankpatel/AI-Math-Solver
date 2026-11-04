# Render free web service (512 MB RAM, 0.1 CPU). Build context: backend/ (see render.yaml).
# Local development keeps using backend/Dockerfile and docker-compose.yml.
FROM python:3.12-slim

# Tuned for 512 MB: one sandbox worker whose address-space limit (640 MB, about 215 MB above its
# idle footprint) stops a memory-hungry job inside the worker before the instance runs out of
# memory; a longer time limit because the instance has only a tenth of a CPU.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PORT=10000 \
    TRUSTED_PROXY_HOPS=1 \
    SANDBOX_WORKERS=1 \
    SANDBOX_MEMORY_MB=640 \
    SANDBOX_TIMEOUT_SECONDS=20 \
    SANDBOX_QUEUE_TIMEOUT_SECONDS=25 \
    ENABLE_DOCS=false \
    LLM_EXPLANATIONS=false

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 mathenza
USER mathenza

EXPOSE 10000
# One API process: the token budget, rate limits and sessions live in its memory.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --no-server-header"]
