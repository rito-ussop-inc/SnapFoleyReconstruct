# SnapFoley cloud — HF Spaces free tier (2 vCPU / 16GB RAM / no GPU)
FROM python:3.11-slim

WORKDIR /app

ENV MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/app/.config/ultralytics \
    FONTCONFIG_PATH=/tmp/fontconfig \
    HOME=/app \
    ENABLE_VLM=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsm6 libxext6 libxrender1 libglib2.0-0 libgomp1 fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/matplotlib /tmp/fontconfig /app/.config/ultralytics /app/tasks

# CPU-only torch first (own layer; ~10x smaller than CUDA builds)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY . .

# Weights ship in the repo — this validates everything imports + loads on this image
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('yolo ok')"

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
