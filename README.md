---
title: SnapFoley
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SnapFoley — offline video-to-foley

Upload any video → SnapFoley detects sound-producing events → generates a matched foley track → returns the final video with audio muxed in. **100% local**: no third-party APIs, no API keys, no internet at runtime.

**Pipeline:** YOLOv8n + MediaPipe pose (footsteps) → MediaPipe Hands (typing) → MOG2 motion fallback (activity/ambience) → type-aware procedural / retrieval audio → ffmpeg mux.

## Web UI
Served at `/` — upload a video, watch detection progress, then generate + preview/download the final video with foley.

## API
| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | liveness + ffmpeg info |
| `/api/upload-video` | POST | upload video file |
| `/api/process/{id}` | POST | detect events |
| `/api/status/{id}` | GET | processing progress |
| `/api/timeline/{id}` | GET | event timeline JSON |
| `/api/generate-audio-video/{id}` | POST | build final video with foley |
| `/api/audio-video-status/{id}` | GET | build progress |
| `/api/download-audio-video/{id}` | GET | final mp4 download |
| `/api/export-csv` / `/api/export-json` | GET | timeline exports |

Tuned for free-tier containers (2 vCPU / 16GB RAM): nano-class models, strided frame sampling, VLM disabled by default (`ENABLE_VLM=1` to enable).

## QA
`python qa_loop.py` must be green before shipping (writes `qa_report.json`).
