# SnapFoley — Fully Local Winning Build (RITO laptop)

Specs: i5-12450H | 8GB RAM (7.65 usable) | RTX 2050 4GB VRAM | 477GB SSD
Python: 3.11 (host 3.14 breaks mediapipe — use Docker)

## Feature list (shipped in app/)
- upload-video / upload-script (podcast TXT) / live capture-floor
- Footstep detector (YOLOv8n + MediaPipe c=1) + script detector
- Local VLM SmolVLM-500M CPU (no remote API)
- Retrieval SFX (CLAP + FAISS, fallback to bundled mp3)
- Generation disabled by default (needs >4GB VRAM)
- Multi-track timeline JSON + stems + CSV/JSON export + ffmpeg mix
- Health / ffmpeg-info, offline guarantee

## Run it
```powershell
cd ai-foley-studio-backend
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements-local.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
# docs http://127.0.0.1:7860/docs
```

## QA loop (must be GREEN before ship)
```powershell
python qa_loop.py
# checks: offline refs, config tuned, imports, health, synthetic video, retrieval fallback, RAM budget
# writes qa_report.json, loops max 3 rounds
```

## Hardware tuning (winning vs generic)
- max_workers 4 -> 2 (OOM on 8GB)
- ffmpeg threads 4 -> 2
- VLM device CPU (keep GPU for YOLO only)
- Generation OFF (4GB VRAM can't hold Stable Audio + YOLO)
- MediaPipe complexity 1
