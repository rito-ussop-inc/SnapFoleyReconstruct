"""
SnapFoley Backend — WINNING fully-local version.
Target: i5-12450H | 8GB RAM | RTX 2050 4GB VRAM
No remote API, no web download, no keys. Offline-first.
"""
import os, json, time, re, tempfile, shutil, subprocess, base64
from datetime import datetime
from io import BytesIO
from typing import Optional, List
import pandas as pd
import cv2, numpy as np

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from app.schemas import Timeline
from app.detectors.footstep import FootstepDetector
from app.detectors.universal import UniversalDetector
from app.detectors.base import ScriptDetector
from app.vlm_local import LocalVLM
from app.audio.retrieval import RetrievalBackend
from app.audio.post import trim_silence, normalize_lufs, fit_to_duration
from app import mixer
from app.video import create_annotated_video

app = FastAPI(title="SnapFoley Backend", version="1.0.0-local")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

tasks = {}
vlm = LocalVLM(device="cpu")  # CPU only — saves VRAM for YOLO on RTX 2050 4GB
retriever = RetrievalBackend(bank_dir="audio_bank", top_k=3)
script_detector = ScriptDetector()
detector = UniversalDetector(sensitivity="medium", yolo_conf=0.5)  # any video, not just footsteps

class ProcessingConfig(BaseModel):
    sensitivity: str = "medium"
    yolo_conf: float = 0.5
    use_hybrid: bool = True
    mode: str = "video"  # video | script

def get_ffmpeg():
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

@app.get("/api/health")
def health():
    # no heavy model load — QA loop stays fast on 8GB
    return {
        "status": "ok", "mode": "offline", "profile": "laptop_2050_8gb",
        "offline_guarantee": True,
        "models": {"yolo": "yolov8n.pt", "vlm": "SmolVLM-500M", "retrieval": "CLAP-base"},
        "ffmpeg": get_ffmpeg(),
    }

@app.get("/")
def root():
    idx = Path(__file__).parent.parent / "frontend" / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"message": "SnapFoley Backend (fully local)", "version": "1.0.0-local"}

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...), config: Optional[str] = None):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be video")
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(await file.read()); tmp.close()
    cfg = ProcessingConfig(**(json.loads(config) if config else {}))
    tasks[task_id] = {"task_id": task_id, "status": "uploaded", "progress": 0.0,
                      "video_path": tmp.name, "config": cfg.dict(), "created_at": datetime.now().isoformat()}
    return {"task_id": task_id, "status": "uploaded"}

# ---- chunked upload ---------------------------------------------------------
# Tunnel proxies (GitHub Codespaces port forwarding, trycloudflare, etc.) reject
# large request bodies with "413 Request Entity Too Large". The frontend splits
# the video into ~4MB parts, posts each one here, then finalizes below.

UPLOAD_DIR = Path(tempfile.gettempdir()) / "snapfoley_uploads"
_ID_RE = re.compile(r"[A-Za-z0-9_-]{4,64}")

def _is_video_head(head: bytes) -> bool:
    """Accept ANY common container by magic bytes: mp4/mov, mkv/webm, avi, ogv, flv, mpeg."""
    if len(head) < 12: return False
    if head[4:8] == b"ftyp": return True                                    # mp4 / mov / m4v
    if head[:4] == b"\x1aE\xdf\xa3": return True                            # mkv / webm (EBML)
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ": return True           # avi
    if head[:4] == b"OggS": return True                                     # ogg / ogv
    if head[:3] == b"FLV": return True                                      # flv
    if head[:4] in (b"\x00\x00\x00\x01", b"\x00\x00\x01\xba"): return True  # mpeg-ts / mpeg-ps
    return False

def _prune_abandoned_uploads(max_age_s: float = 7200.0):
    now = time.time()
    for d in UPLOAD_DIR.iterdir() if UPLOAD_DIR.exists() else []:
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

@app.post("/api/upload-chunk")
async def upload_chunk(upload_id: str = Form(...), index: int = Form(...), chunk: UploadFile = File(...)):
    if not _ID_RE.fullmatch(upload_id): raise HTTPException(400, "bad upload_id")
    if index < 0 or index > 10000: raise HTTPException(400, "bad chunk index")
    data = await chunk.read()
    d = UPLOAD_DIR / upload_id
    is_new = not d.exists()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"chunk_{index:06d}").write_bytes(data)
    if is_new: _prune_abandoned_uploads()
    return {"ok": True, "index": index, "bytes": len(data)}

class UploadCompleteRequest(BaseModel):
    upload_id: str
    filename: str = "video.mp4"
    config: Optional[dict] = None

@app.post("/api/upload-complete")
async def upload_complete(req: UploadCompleteRequest):
    if not _ID_RE.fullmatch(req.upload_id): raise HTTPException(400, "bad upload_id")
    d = UPLOAD_DIR / req.upload_id
    parts = sorted(d.glob("chunk_*")) if d.exists() else []
    if not parts: raise HTTPException(404, "Unknown upload_id — no chunks received")
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    head = b""
    try:
        for p in parts:
            b = p.read_bytes()
            if not head: head = b[:64]
            tmp.write(b)
        tmp.close()
        if not _is_video_head(head):
            raise HTTPException(400, "File is not a recognized video container (mp4/mov/mkv/webm/avi/ogv/flv/mpeg)")
        cfg = ProcessingConfig(**(req.config or {}))
        tasks[task_id] = {"task_id": task_id, "status": "uploaded", "progress": 0.0,
                          "video_path": tmp.name, "config": cfg.dict(), "created_at": datetime.now().isoformat()}
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return {"task_id": task_id, "status": "uploaded"}

@app.post("/api/process/{task_id}")
def process(task_id: str, background_tasks: BackgroundTasks):
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    tasks[task_id]["status"] = "processing"
    background_tasks.add_task(_process_task, task_id)
    return {"task_id": task_id, "status": "processing"}

def _fallback_prompt(events) -> str:
    """Sound-design label derived from what was actually DETECTED (VLM is off on 8GB)."""
    types = set(e.type for e in events)
    n = {t: sum(1 for e in events if e.type == t) for t in types}
    top = max(n, key=n.get) if n else "none"
    if top == "typing":
        return "laptop keyboard typing, close-mic, foley"
    if top == "bounce":
        return "basketball dribbles and ball bounces, close-mic, foley"
    if top == "punch":
        return "gloved punches hitting a heavy bag, close-mic, foley"
    if top == "clap":
        return "hand claps, close-mic, foley"
    if top == "footstep":
        # prefer the detector's own surface-aware label (POV gait detects the
        # ground: pavement / grass / fallen leaves) over the generic string
        fps_prompts = [e.prompt for e in events if e.type == "footstep" and e.prompt]
        if fps_prompts:
            return max(set(fps_prompts), key=fps_prompts.count)
        return "footsteps on surface, close-mic, foley"
    if top == "ambience":
        return "ambient room tone, soft foley"
    if top == "activity":
        return "motion impacts and foley hits, close-mic"
    return "video foley, close-mic"

def _process_task(task_id: str):
    t = tasks[task_id]
    try:
        cfg = t["config"]
        events = detector.detect(t["video_path"], progress_callback=lambda p: t.update(progress=p))
        t["events"] = [e.dict() for e in events]
        # local VLM prompt enrichment (fast path — fallback string if model not cached)
        prompt = vlm.describe_video(t["video_path"], fallback=_fallback_prompt(events))
        stems = []
        for ev in events:
            # per-event prompt (footstep vs activity/ambience), fall back to VLM prompt
            path, score = retriever.search(ev.prompt or prompt)
            stems.append({"event_id": ev.event_id, "stem_path": path or "", "backend": "retrieval", "score": score,
                          "loop": ev.type == "ambience", "gain_db": 0.0, "fade_in": 0.05})
        t["stems"] = stems
        t["prompt"] = prompt
        tl = mixer.build_timeline(task_id, events, [type("C", (), s)() if False else _as_clip(s) for s in stems], duration_sec=(events[-1].end_sec if events else 0))
        t["timeline"] = tl.dict()
        t["status"] = "completed"; t["progress"] = 1.0
    except Exception as e:
        import traceback; traceback.print_exc()
        t["status"] = "failed"; t["error"] = str(e)

def _as_clip(d):
    from app.schemas import Clip
    return Clip(**{k: d[k] for k in ["event_id","stem_path","backend","score","loop","gain_db","fade_in"] if k in d})

@app.get("/api/status/{task_id}")
def status(task_id: str):
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    t = tasks[task_id]
    r = {"task_id": task_id, "status": t["status"], "progress": t.get("progress", 0)}
    if t["status"] == "completed": r.update({"events": t.get("events"), "prompt": t.get("prompt"), "timeline": t.get("timeline")})
    if "error" in t: r["error"] = t["error"]
    return r

@app.get("/api/timeline/{task_id}")
def timeline(task_id: str):
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    if tasks[task_id]["status"] != "completed": raise HTTPException(400, "Not ready")
    return tasks[task_id].get("timeline", {})

@app.post("/api/upload-script")
async def upload_script(file: UploadFile = File(...)):
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.write(await file.read()); tmp.close()
    events = script_detector.detect(tmp.name)
    stems = []
    for ev in events:
        path, score = retriever.search(ev.prompt)
        stems.append({"event_id": ev.event_id, "stem_path": path or "", "backend": "retrieval", "score": score, "loop": True, "gain_db": -6.0, "fade_in": 0.05})
    dur = events[-1].end_sec if events else 0
    tl = mixer.build_timeline(task_id, events, [_as_clip(s) for s in stems], duration_sec=dur)
    tasks[task_id] = {"task_id": task_id, "status": "completed", "progress": 1.0, "events": [e.dict() for e in events], "stems": stems, "timeline": tl.dict(), "prompt": "script ambience"}
    return {"task_id": task_id, "status": "completed", "events": len(events)}

@app.get("/api/export-csv/{task_id}")
def export_csv(task_id: str):
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    df = pd.DataFrame(tasks[task_id].get("events", []))
    buf = BytesIO(); df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=snapfoley_events_{task_id}.csv"})

@app.get("/api/export-json/{task_id}")
def export_json(task_id: str):
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    return tasks[task_id].get("timeline", {})

@app.post("/api/generate-audio-video/{task_id}")
def gen_audio_video(task_id: str, background_tasks: BackgroundTasks):
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    if tasks[task_id]["status"] != "completed": raise HTTPException(400, "Not completed")
    tasks[task_id]["final_generating"] = True; tasks[task_id]["final_ready"] = False
    background_tasks.add_task(_gen_final, task_id)
    return {"task_id": task_id, "message": "final video started"}

def _gen_final(task_id: str, annotate: bool = True):
    from app.schemas import Event
    t = tasks[task_id]
    try:
        video_path = t["video_path"]
        evs = [Event(**e) for e in t.get("events", [])]
        stems = [s.get("stem_path") for s in t.get("stems", [])]
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        duration = total / fps
        # 1. annotated temp (editor remixes skip this — the user wants clean video)
        if annotate:
            annotated = tempfile.mktemp(suffix="_annotated.mp4")
            ok = create_annotated_video(video_path, evs, annotated)
            if not ok: raise RuntimeError("annotate failed")
        else:
            annotated = video_path
        # 2. audio track
        audio_track = mixer.mix_track(evs, stems, duration, sr=44100, prompt=t.get("prompt", ""))
        # 3. merge (fallback to annotated if ffmpeg missing on this host)
        final = tempfile.mktemp(suffix="_final.mp4")
        merged = mixer.merge_with_ffmpeg(annotated, audio_track, 44100, final, ffmpeg=get_ffmpeg())
        if not merged:
            import shutil as _sh
            _sh.move(annotated, final)
            print("[gen] ffmpeg missing — returning annotated video as fallback (install ffmpeg for audio mux)")
        t["final_path"] = final; t["final_ready"] = True; t["final_generating"] = False; t["final_audio"] = merged
    except Exception as e:
        import traceback; traceback.print_exc()
        t["final_error"] = str(e); t["final_ready"] = False; t["final_generating"] = False

@app.get("/api/audio-video-status/{task_id}")
def audio_video_status(task_id: str):
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    t = tasks[task_id]
    return {"task_id": task_id, "audio_video_ready": t.get("final_ready", False),
            "audio_video_generating": t.get("final_generating", False), "audio_video_error": t.get("final_error")}

@app.get("/api/download-audio-video/{task_id}")
def download_audio_video(task_id: str):
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    t = tasks[task_id]
    if not t.get("final_ready"): raise HTTPException(400, "Video not ready")
    p = t.get("final_path")
    if not p or not os.path.exists(p): raise HTTPException(404, "File missing")
    return FileResponse(p, media_type="video/mp4", filename=f"snapfoley_{task_id}.mp4")

# ---- manual editor (re-render audio from edited events, no re-detection) ----

VALID_TYPES = {"footstep", "typing", "activity", "ambience", "bounce", "punch", "clap"}
_TYPE_PROMPT = {
    "footstep": "footsteps on surface, close-mic, foley",
    "typing":   "laptop keyboard typing, close-mic, foley",
    "activity": "motion impact hit, close-mic, foley",
    "ambience": "soft room ambience, indoor foley",
    "bounce":   "ball bounce impact, close-mic, foley",
    "punch":    "punch impact on heavy bag, foley",
    "clap":     "hand clap, close-mic, foley",
}
_TYPE_DUR = {"footstep": 0.4, "typing": 0.12, "activity": 0.35, "ambience": 1.0,
             "bounce": 0.35, "punch": 0.5, "clap": 0.3}

class RemixEvent(BaseModel):
    event_id: Optional[str] = None
    type: str = "activity"
    start_sec: float = 0.0
    gain: float = 1.0

class RemixRequest(BaseModel):
    events: List[RemixEvent]

def _timecode(sec: float, fps: float) -> str:
    ts = max(0.0, sec)
    return f"{int(ts//3600):02d}:{int((ts%3600)//60):02d}:{int(ts%60):02d}:{int((ts*fps)%fps):02d}"

@app.post("/api/remix/{task_id}")
def remix(task_id: str, req: RemixRequest, background_tasks: BackgroundTasks):
    """Apply manually edited events (time/type/volume, added & deleted) and
    re-render the audio. Skips detection AND annotation — usually a few seconds."""
    if task_id not in tasks: raise HTTPException(404, "Task not found")
    t = tasks[task_id]
    if t["status"] != "completed": raise HTTPException(400, "Process the video first")
    if t.get("final_generating"): raise HTTPException(409, "Still rendering — try again in a moment")
    cap = cv2.VideoCapture(t["video_path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()
    duration = total / fps

    from app.schemas import Event
    evs = []
    for i, r in enumerate(req.events):
        etype = r.type if r.type in VALID_TYPES else "activity"
        start = float(min(max(r.start_sec, 0.0), duration))
        ev = Event(
            event_id=r.event_id or f"edit_{i:04d}_{os.urandom(2).hex()}",
            type=etype, start_sec=start, end_sec=start + _TYPE_DUR[etype],
            frame=int(start * fps), timecode=_timecode(start, fps),
            confidence=0.9, prompt=_TYPE_PROMPT[etype],
            duration_hint=_TYPE_DUR[etype], gain=float(min(max(r.gain, 0.0), 2.0)))
        evs.append(ev)
    evs.sort(key=lambda e: e.start_sec)

    stems = []
    for ev in evs:
        path, score = retriever.search(ev.prompt)
        stems.append({"event_id": ev.event_id, "stem_path": path or "", "backend": "retrieval", "score": score,
                      "loop": ev.type == "ambience", "gain_db": 0.0, "fade_in": 0.05})

    t["events"] = [e.dict() for e in evs]
    t["stems"] = stems
    t["prompt"] = _fallback_prompt(evs)
    tl = mixer.build_timeline(task_id, evs, [_as_clip(s) for s in stems], duration_sec=duration)
    t["timeline"] = tl.dict()
    t["final_ready"] = False; t["final_generating"] = True; t.pop("final_error", None)
    background_tasks.add_task(_gen_final, task_id, False)
    return {"task_id": task_id, "status": "remixing", "events": len(evs)}

# Serve the test frontend (same origin, no CORS issues)
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Health for QA loop
@app.get("/api/ffmpeg-info")
def ffmpeg_info():
    ff = get_ffmpeg()
    try:
        proc = subprocess.run([ff, "-version"], capture_output=True, text=True, timeout=5)
        ver = proc.stdout.splitlines()[0] if proc.returncode == 0 else None
    except Exception: ver = None
    return {"ffmpeg_path": ff, "version": ver}
