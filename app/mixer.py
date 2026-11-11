"""Multi-track mixer — replaces single-track AudioGenerator."""
import numpy as np, os, tempfile, subprocess
try:
    import soundfile as sf
except ImportError:
    sf = None
from typing import List
from app.schemas import Event, Clip, Track, Timeline

def extract_1sec(path, sr=44100, second=0):
    try:
        import librosa
        y, _ = librosa.load(path, sr=sr, offset=second, duration=1.0)
        if len(y) < sr: y = np.pad(y, (0, sr-len(y)))
        return y
    except Exception:
        # fallback silent foley click
        t = np.linspace(0, 0.2, int(0.2*sr))
        return (np.sin(2*np.pi*100*t)*np.exp(-10*t)).astype(np.float32)

def build_timeline(task_id, events: List[Event], stems: List[Clip], duration_sec: float) -> Timeline:
    sfx, amb = [], []
    for e, s in zip(events, stems):
        (amb if s.loop or e.type=="ambience" else sfx).append(s)
    return Timeline(task_id=task_id, duration_sec=duration_sec, tracks=[Track(name="sfx", clips=sfx), Track(name="ambience", clips=amb)])

def mix_track(events, stem_paths, duration_sec, sr=44100):
    total = int(duration_sec*sr)
    track = np.zeros(total, dtype=np.float32)
    cache = {}
    # Fast path: if ANY typing/activity → continuous typing bed (dense clicks) instead of sparse event pops
    has_typing = any(e.type in ("typing", "activity") for e in events)
    if has_typing:
        click = _synth_key_click(sr=sr)
        # 4-6 clicks/sec across full duration, jittered — sounds like typing
        import random as _r
        t_sec = 0.0
        while t_sec < duration_sec - 0.1:
            start = int(t_sec * sr)
            end = min(start + len(click), total)
            if end > start:
                track[start:end] += click[:end-start] * 0.35
            t_sec += _r.uniform(0.14, 0.26)
        # add event accents slightly louder
        for e in events:
            if e.type in ("typing", "activity"):
                start = int(e.start_sec * sr)
                end = min(start + len(click), total)
                if end > start:
                    track[start:end] += click[:end-start] * 0.4
        peak = np.max(np.abs(track)) if len(track) else 0
        if peak > 1e-6: track = track/peak*0.7
        return track
    for i, (e, p) in enumerate(zip(events, stem_paths)):
        # Type-aware stem: use procedural synth for activity/ambience instead of footstep mp3
        if e.type in ("activity", "ambience"):
            if ("typing" not in cache):
                cache["typing"] = _synth_key_click(sr=sr)
            seg = cache["typing"]
            # alternate clicks
            if i % 2 == 1 and len(seg) > 1000:
                seg = seg[::-1].copy()
        else:
            if not p or not os.path.exists(p): continue
            if p not in cache:
                cache[p] = extract_1sec(p, sr, second=0)
            seg = cache[p]
        # pitch variation ±6% (resample by indexing)
        shift = 1.0 + (i % 5 - 2) * 0.03
        idx = np.clip((np.arange(len(seg))*shift), 0, len(seg)-1).astype(int)
        seg = seg[idx]
        start = int(e.start_sec*sr)
        end = min(start+len(seg), total)
        if end > start:
            track[start:end] += seg[:end-start]*0.5 if e.type in ("activity","ambience") else seg[:end-start]
    peak = np.max(np.abs(track)) if len(track) else 0
    if peak > 1e-6: track = track/peak*0.8
    return track

def _synth_key_click(sr=44100, dur=0.08):
    """Procedural keyboard click: short noise burst + 2kHz ping + fast decay. ~2KB, no librosa load."""
    t = np.linspace(0, dur, int(sr*dur))
    click = (np.random.randn(len(t)) * 0.5 + np.sin(2*np.pi*2000*t)*0.3) * np.exp(-60*t)
    return click.astype(np.float32)

def merge_with_ffmpeg(video_path, audio_track, sr, output_path, ffmpeg="ffmpeg"):
    tmp = tempfile.mktemp(suffix=".wav")
    if sf is not None:
        sf.write(tmp, audio_track, sr)
    else:
        from scipy.io.wavfile import write as wav_write
        wav_write(tmp, sr, (np.clip(audio_track, -1, 1) * 32767).astype(np.int16))
    cmd = [ffmpeg, "-nostdin", "-y", "-i", str(video_path), "-i", tmp,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
           "-vf", "scale='min(1280,iw)':-2", "-threads", "2",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return True
    except Exception as e:
        print(f"[mixer] ffmpeg failed: {e}")
        return False
    finally:
        if os.path.exists(tmp): os.remove(tmp)
