# =============================================================================
# ASTINA — multi-stage Dockerfile
# -----------------------------------------------------------------------------
# Stage 1 (builder): compile Python wheels and resolve all transitive deps
#                    in a throwaway image. Produces a clean /install prefix
#                    that is then COPY'ed into the runtime image.
# Stage 2 (runtime): minimal Python 3.11-slim with just the compiled deps.
#                    Runs as non-root (UID 1000) and listens on $PORT.
#
# Build args:
#   PIP_EXTRA_INDEX_URL — override to install CUDA builds of torch-scatter /
#                          torch-sparse (e.g. https://data.pyg.org/whl/torch-2.1.0+cu118).
#   TORCH_VERSION        — pinned torch version (default 2.1.0).
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for building wheels (numpy, scipy, lightgbm, hdbscan, ...).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install deps into a relocatable prefix so the runtime stage can COPY them.
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2 — runtime
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POLARS_SKIP_CPU_CHECK=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3072 \
    ASTINA_LOG_FORMAT=json \
    HOME=/home/appuser

# Default to port 8501 for Docker Desktop local runs.
# Cloud Run uses the same PORT value and can override it at runtime.
ENV PORT=8501
EXPOSE 8501

# System runtime deps only — no compilers, no headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bring the prebuilt site-packages from the builder.
COPY --from=builder /install /usr/local

# Non-root user for Cloud Run best practice.
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app
COPY --chown=appuser:appuser . .

# Required writable directories (cache, models, streamlit config).
RUN mkdir -p /app/cache /app/models /app/.streamlit /home/appuser \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

# Healthcheck against the Streamlit-internal health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD sh -c "curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1"

# Streamlit config supports Docker Desktop and Cloud Run through PORT.
CMD ["sh", "-c", "streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=true --server.enableXsrfProtection=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
