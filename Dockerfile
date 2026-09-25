# Reproducible container: the acceptance gate says the project must run from documented
# instructions on a clean machine. This is that clean machine, pinned in a file.
#
# Deliberate choices:
#   * CPU-only PyTorch. The image is for reproducibility and CI, not for speed; the GPU
#     path is the laptop's, and shared/device.py falls back to CPU by itself.
#   * opencv-python-headless, not opencv-python: there is no display inside a container,
#     and the GUI build drags in X11 libraries that would never be used.
#   * The model weights are NOT baked in. They are 5 MB of trained artefact, not source,
#     and mounting them keeps the image reproducible from the repository alone.
#   * Runs as a non-root user. A container that watches a camera should not be root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# libgl1 + libglib2.0-0 are what OpenCV needs even in its headless build.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-download PyTorch.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install opencv-python-headless

COPY . .

# Data written at runtime (logs, outbox, evidence) belongs to the runtime user.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/data \
 && chown -R app:app /app/data
USER app

EXPOSE 8000

# The image is healthy when the API answers, not merely when the process exists.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs http://127.0.0.1:8000/health || exit 1

# 0.0.0.0 inside the container is correct - the container boundary is the isolation.
# run.py binds 127.0.0.1 on a laptop, which is the right default there. See D-027.
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
