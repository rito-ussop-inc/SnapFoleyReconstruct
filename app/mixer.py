"""Multi-track mixer — real-sample foley renderer.

Design (v2, "matches the video" edition):
 - Footsteps: real CC0 step samples (Kenney, audio_bank/footsteps/<surface>/),
   one per detected step, with pitch/gain/timing variation so no two hits are
   identical. Surface picked from the prompt keywords, default concrete.
 - Typing: dual-transient keyboard click bed, only across the typing span.
 - Activity: soft real impact transients synced to each detected motion peak.
 - Ambience: quiet noise bed across ambience ranges only.
 - Everything is rendered at the EVENT times -> audio follows the video.
"""
import os, random, tempfile, subprocess
import numpy as np
from pathlib import Path
try:
    import soundfile as sf
except ImportError:
    sf = None
from typing import List
from app.schemas import Event, Clip, Track, Timeline

BANK = Path(__file__).resolve().parent.parent / "audio_bank"

SURFACE_KEYWORDS = {
    "concrete": ["concrete", "pavement", "paved", "stone", "brick", "sidewalk",
                 "street", "road", "tile", "marble", "asphalt", "cobble"],
    "grass":    ["grass", "gravel", "lawn", "field", "meadow", "garden", "dirt"],
    "wood":     ["wood", "wooden", "floorboard", "hardwood", "deck", "floor"],
    "carpet":   ["carpet", "rug", "indoors", "indoor", "room", "office", "home"],
    "snow":     ["snow", "ice", "frost", "winter"],
}
DEFAULT_SURFACE = "concrete"

_sample_cache = {}

def map_surface(prompt: str) -> str:
    p = (prompt or "").lower()
    for surface, kws in SURFACE_KEYWORDS.items():
        for kw in kws:
            if kw in p:
                return surface
    return DEFAULT_SURFACE

def _load_sample(path: Path, sr: int) -> np.ndarray:
    key = (str(path), sr)
    if key in _sample_cache:
        return _sample_cache[key]
    y = None
    try:
        import librosa
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception:
        pass
    if y is None or len(y) == 0:
        y = _synth_step(sr)
    _sample_cache[key] = y.astype(np.float32)
    return _sample_cache[key]

def _bank_files(kind: str, surface: str = "") -> List[Path]:
    if surface:
        files = sorted((BANK / kind / surface).glob("*.wav"))
        if files:
            return files
    return sorted((BANK / kind).rglob("*.wav"))

def _place(track: np.ndarray, y: np.ndarray, start: int, gain: float = 1.0):
    total = len(track)
    end = min(start + len(y), total)
    if start < total and end > start:
        s0 = max(0, start)
        track[s0:end] += y[s0 - start:end - start] * gain

def _vary(y: np.ndarray, pitch_range=0.07) -> np.ndarray:
    shift = 1.0 + random.uniform(-pitch_range, pitch_range)
    n = max(1, int(len(y) / shift))
    return np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)

# ---------------------------------------------------------------- renderers --

def render_footsteps(events, duration_sec, sr, prompt=""):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    surface = map_surface(prompt)
    files = _bank_files("footsteps", surface)
    for i, e in enumerate(events):
        if files:
            y = _vary(_load_sample(files[i % len(files)], sr))
        else:
            y = _vary(_synth_step(sr))
        jitter = random.uniform(-0.02, 0.02)
        _place(track, y, int((e.start_sec + jitter) * sr), random.uniform(0.65, 1.0))
    return track

def render_typing(events, duration_sec, sr):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    if not events: return track
    click = _synth_key_click(sr)
    t0 = min(e.start_sec for e in events)
    t1 = min(max((e.end_sec if e.end_sec and e.end_sec > e.start_sec else e.start_sec + 0.5)
                 for e in events), duration_sec)
    t = t0
    while t < t1 - 0.05:
        _place(track, click, int(t * sr), random.uniform(0.25, 0.5))
        t += random.uniform(0.13, 0.30)
    for e in events:  # accents on detected strikes
        _place(track, click, int(e.start_sec * sr), 0.65)
    return track

def render_activity(events, duration_sec, sr):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    files = _bank_files("impacts")
    for i, e in enumerate(events):
        y = _vary(_load_sample(files[i % len(files)], sr), 0.06) if files else _vary(_synth_step(sr), 0.06)
        _place(track, y, int(e.start_sec * sr), random.uniform(0.25, 0.45))
    return track

def render_ambience(events, duration_sec, sr):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    if not events: return track
    t0 = max(0.0, min(e.start_sec for e in events))
    t1 = min(duration_sec, max((e.end_sec if e.end_sec > e.start_sec else e.start_sec + 1.0) for e in events))
    if t1 <= t0: return track
    n = int((t1 - t0) * sr)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n).astype(np.float32)
    # crude lowpass -> soft room tone
    k = 9
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    fade = int(0.5 * sr)
    env = np.ones(n, dtype=np.float32)
    if n > 2 * fade:
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
    track[int(t0 * sr):int(t0 * sr) + n] += noise * env * 0.05
    return track

# ------------------------------------------------------------------ public --

def build_timeline(task_id, events: List[Event], stems: List[Clip], duration_sec: float) -> Timeline:
    sfx, amb = [], []
    for e, s in zip(events, stems):
        (amb if s.loop or e.type == "ambience" else sfx).append(s)
    return Timeline(task_id=task_id, duration_sec=duration_sec, tracks=[Track(name="sfx", clips=sfx), Track(name="ambience", clips=amb)])

def mix_track(events, stem_paths, duration_sec, sr=44100, prompt=""):
    total = int(duration_sec * sr)
    if total <= 0: return np.zeros(1, dtype=np.float32)
    track = np.zeros(total, dtype=np.float32)
    by = lambda t: [e for e in events if e.type == t]
    foot, typ, act, amb = by("footstep"), by("typing"), by("activity"), by("ambience")
    if foot: track += render_footsteps(foot, duration_sec, sr, prompt)
    if typ:  track += render_typing(typ, duration_sec, sr)
    if act:  track += render_activity(act, duration_sec, sr)
    if amb:  track += render_ambience(amb, duration_sec, sr)
    peak = np.max(np.abs(track))
    if peak > 1e-6: track = (track / peak * 0.85).astype(np.float32)
    return track

def _synth_step(sr=44100):
    """Procedural footstep fallback: heel thump + surface crunch."""
    dur = 0.09
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    thump = np.sin(2 * np.pi * (90 * np.exp(-t * 30)) * t) * np.exp(-t * 45)
    crunch = np.random.randn(len(t)) * np.exp(-t * 35)
    y = 0.6 * thump + 0.5 * crunch
    y /= (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)

def _synth_key_click(sr=44100):
    """Dual-transient mechanical key click (down + softer up)."""
    def burst(dur, f, decay):
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        return (np.random.randn(len(t)) * 0.8 + 0.25 * np.sin(2 * np.pi * f * t)) * np.exp(-decay * t)
    down, up = burst(0.03, 2600, 120), burst(0.02, 2100, 160)
    y = np.zeros(int(sr * 0.09), dtype=np.float32)
    y[:len(down)] += down
    o = int(0.045 * sr)
    y[o:o + len(up)] += up
    return y.astype(np.float32)

def extract_1sec(path, sr=44100, second=0):
    try:
        import librosa
        y, _ = librosa.load(path, sr=sr, offset=second, duration=1.0)
        if len(y) < sr: y = np.pad(y, (0, sr-len(y)))
        return y
    except Exception:
        t = np.linspace(0, 0.2, int(0.2*sr))
        return (np.sin(2*np.pi*100*t)*np.exp(-10*t)).astype(np.float32)

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
