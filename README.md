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

## For teammates — working on this repo

Everything a teammate needs is committed; nothing lives only on the original laptop:

- **Zero setup on Codespaces:** opening a Codespace runs `.devcontainer/setup.sh` (installs ffmpeg + all Python deps), and `postStartCommand` auto-launches the server — the UI is simply there when the editor opens.
- **Models are in the repo:** `yolov8n.pt` is committed; `app/` detectors degrade gracefully to the motion-energy fallback if MediaPipe/YOLO weights are missing, so detection always returns events.
- **All sound synthesis is in `app/mixer.py`** (physical synthesis — no external sample downloads, no licensing traps). The `audio_bank/` retrieval samples are committed too.
- **Run anywhere with Python 3.10–3.11:** `pip install -r requirements-cloud.txt` + `ffmpeg` on PATH → `uvicorn app.main:app --port 7860`. (MediaPipe needs ≤3.11; Python 3.12+ works but falls back to motion-only detection.)
- **Verify a change:** `python qa_loop.py` must be 7/7 green; then upload the demo videos through the UI (boxing → 16 punches, basketball → 10 bounces, typing → keystrikes, POV walk → steps) and check the timeline.

### Why everything is local?

Three deliberate reasons, not limitations:

1. **Hackathon-critical reliability:** cloud AI ( paid APIs) fail at demo time — rate limits, keys expiring, Wi-Fi dead zones. The demo must run off a laptop battery or a free Codespace with no network beyond the page itself.
2. **Zero cost:** every "good" foley/sound-effect API charges per call and needs an API key; the team has none. Physical synthesis in `mixer.py` + committed CC0 samples produce the sounds offline for free, forever.
3. **Privacy + simplicity for users:** raw, unedited video never leaves the machine — no uploads to someone else's storage, no accounts, no quota.

### Sharing your exact setup with teammates

Cloning the GitHub repo **is** the full setup — the detectors, mixer, models, samples, devcontainer and scripts all live in the repo. What is *not* in the repo and cannot be: your laptop's `.venv` (rebuild it from `requirements-*.txt` — or just use Codespaces where it's built automatically) and any un-pushed local files. So as long as you commit + push your changes, a teammate gets *exactly* the same behavior by cloning — they don't need access to your laptop.

## QA
`python qa_loop.py` must be green before shipping (writes `qa_report.json`).
