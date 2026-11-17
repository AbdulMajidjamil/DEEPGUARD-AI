# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil - DEEPGUARD AI container image (derived from SENTINEL, see DERIVATION.md)
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_THREADS=1 \
    SPATIAL_BATCH=4 \
    DEEPGUARD_DATA_DIR=/app/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 deepguard

COPY requirements.txt .
# CPU-only PyTorch keeps the image small; for CUDA build with a CUDA base image instead.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=deepguard:deepguard . .
RUN mkdir -p /app/data /app/weights && chown -R deepguard:deepguard /app/data /app/weights
USER deepguard

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
