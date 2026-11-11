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
| `/api/upload-video` | POST | upload video file (single body — small files) |
| `/api/upload-chunk` | POST | one ~4MB part of a chunked upload (tunnel proxies cap body size) |
| `/api/upload-complete` | POST | finalize chunked upload → creates the task |
| `/api/process/{id}` | POST | detect events |
| `/api/status/{id}` | GET | processing progress |
| `/api/timeline/{id}` | GET | event timeline JSON |
| `/api/generate-audio-video/{id}` | POST | build final video with foley |
| `/api/audio-video-status/{id}` | GET | build progress |
| `/api/download-audio-video/{id}` | GET | final mp4 download |
| `/api/export-csv` / `/api/export-json` | GET | timeline exports |

Tuned for free-tier containers (2 vCPU / 16GB RAM): nano-class models, strided frame sampling, VLM disabled by default (`ENABLE_VLM=1` to enable).

## Run in GitHub Codespaces — stable public URL (free)

1. On the repo page: **Code → Codespaces → Create codespace on `main`** (a 4-core/8GB machine is auto-requested)
2. First build takes ~5–8 min (ffmpeg + python deps install automatically)
3. In the terminal run: `./start_snapfoley.sh`
4. The web UI opens at `https://<codespace-name>-7860.app.github.dev` — **this URL is stable across restarts**
5. If the page asks you to log in to GitHub, set port 7860 to Public: Ports panel → right-click 7860 → Port Visibility → Public (the start script attempts this automatically)

Free tier includes 120 core-hours/month (≈30 h on a 4-core machine). Codespaces auto-stop after 30 min idle; restart = open the codespace → `./start_snapfoley.sh` → same URL.

## QA
`python qa_loop.py` must be green before shipping (writes `qa_report.json`).
