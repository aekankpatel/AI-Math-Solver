# Hugging Face Spaces (Docker SDK, free "CPU basic": 2 vCPU, 16 GB RAM).
# Built from the bundle assembled by scripts/build_space_bundle.sh (backend/app + requirements.txt).
# Local development keeps using backend/Dockerfile and docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    TRUSTED_PROXY_HOPS=1 \
    SANDBOX_WORKERS=2 \
    SANDBOX_MEMORY_MB=1536 \
    ENABLE_DOCS=false \
    LLM_EXPLANATIONS=false

# Spaces run containers as UID 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user app ./app

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')" || exit 1
# One API process: the token budget, rate limits and sessions live in its memory;
# computations run in the sandbox worker pool (SANDBOX_WORKERS).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--no-server-header"]
