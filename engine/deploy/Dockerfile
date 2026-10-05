# Snappy v2.5 — CPU production image.
#
# For GPU, see deploy/Dockerfile.gpu (CUDA 12.1 base + matching torch wheel).
# Build:    docker build -f deploy/Dockerfile -t snappy:v2.5 .
# Run:      docker run --rm -p 8765:8765 -v $PWD/backend/data:/app/backend/data \
#                  -v $PWD/captures:/app/captures snappy:v2.5
#
# The image is intentionally fat (~3 GB) because it bakes in:
#   - opencv-python-headless (the Docker-friendly variant)
#   - mediapipe (~150 MB)
#   - torch CPU wheel + transformers (~1.5 GB)
#   - pyiqa, ultralytics, hsemotion-onnx, faster-whisper
#
# At runtime the model files (CLIP weights, NIMA weights, YOLO weights,
# face_landmarker.task) live under /app/backend/models — face_landmarker
# is checked into the repo; CLIP/NIMA/YOLO auto-download on first request.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SNAPPY_DEVICE=cpu

# System libs needed by OpenCV / mediapipe / onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use the headless OpenCV in containers (no X libs required).
RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN sed 's/^opencv-python>/opencv-python-headless>/' /tmp/requirements.txt > /tmp/req.docker.txt \
    && pip install -r /tmp/req.docker.txt

COPY . /app

EXPOSE 8765

# Healthcheck — Docker / Kubernetes can probe /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8765/health || exit 1

CMD ["python3", "run.py"]
