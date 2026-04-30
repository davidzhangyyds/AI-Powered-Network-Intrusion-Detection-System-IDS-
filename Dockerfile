# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1 — Builder: install python deps into a self-contained virtualenv
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps required by some wheels (numpy/scipy/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv under /opt/venv so we can copy it into the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements-docker.txt .

RUN pip install --upgrade pip \
 && pip install -r requirements-docker.txt


# =============================================================================
# Stage 2 — Runtime: tiny image with just python + venv + app code
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860 \
    PATH="/opt/venv/bin:$PATH"

# Copy the prebuilt venv from the builder
COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user (Azure best practice)
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Copy only what the runtime actually needs.
# .dockerignore filters out the rest (venv/, mlruns/, notebooks/, large CSVs, ...)
COPY --chown=appuser:appuser src/      ./src/
COPY --chown=appuser:appuser models/   ./models/
COPY --chown=appuser:appuser outputs/  ./outputs/
COPY --chown=appuser:appuser mlflow.db ./mlflow.db

USER appuser

EXPOSE 7860

# Quick liveness check — Gradio exposes /config when up
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request as u; \
u.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/config', timeout=3)" \
    || exit 1

# The app itself reads PORT and binds to 0.0.0.0 — no shell-form needed.
CMD ["python", "src/ui_frontend.py"]