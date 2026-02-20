# ══════════════════════════════════════════════════════════════════════════════
#  hyper-sense-backend  ·  Dockerfile
#  Multi-stage build → final image is python:3.12-slim (~200 MB with PyTorch CPU)
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────── Stage 1: Builder ─────────────────────────────
FROM python:3.12-slim AS builder

# Install build tools needed to compile some C-extension packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy & install dependencies into an isolated prefix so we can COPY them cleanly
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/pkg --no-cache-dir -r requirements.txt


# ─────────────────────────────── Stage 2: Runtime ─────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="hyper-sense-backend"
LABEL description="IoT people-counting API powered by YOLO + FastAPI"

# Runtime system libraries required by OpenCV / PIL / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /pkg /usr/local

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy application code
COPY main.py .

# Pre-download the YOLO model into the container so the first request is fast.
# Ultralytics stores models in ~/.cache/ultralytics/  (mapped to appuser's home)
USER appuser
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

# Expose REST API port
EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
# 2 workers is a safe default for a single-core / 1 GB RAM Raspberry Pi 5.
# On a stronger server increase --workers to match CPU cores.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--no-access-log"]
