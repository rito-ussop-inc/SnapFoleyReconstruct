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

# ------------------------------------------------------------ footstep v3 --
# A real footstep is THREE layers, not one sample:
#   1. body thump   — the weight of the person dropping into the ground
#                     (60-150Hz pitch-dropping sine, the part you FEEL)
#   2. surface      — the contact: heel click + material crunch (the sample)
#   3. scuff        — a short noise swish just BEFORE impact (the foot plant)
# plus a tiny room tail so it doesn't sound dry/game-y.

try:
    from scipy.signal import butter, sosfiltfilt, fftconvolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

def _band(y, sr, kind, f1, f2=None, order=2):
    """Zero-phase band filter; kind in {high, low, band}."""
    if not _HAS_SCIPY or len(y) < 32: return y
    try:
        fs = sr / 2.0
        if kind == "high":
            sos = butter(order, min(f1, fs * 0.95) / fs, btype="high", output="sos")
        elif kind == "low":
            sos = butter(order, min(f1, fs * 0.95) / fs, btype="low", output="sos")
        else:
            lo, hi = sorted((min(f1, fs * 0.95), min(f2 or f1 * 3, fs * 0.95)))
            sos = butter(order, [lo / fs, hi / fs], btype="band", output="sos")
        return sosfiltfilt(sos, y).astype(np.float32)
    except Exception:
        return y

def _synth_thump(sr, f_start=120.0, f_end=52.0, dur=0.20, tau=0.030):
    """Weight-drop thump: pitch falls fast, amplitude decays — the 'body' of a step."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    f = f_start + (f_end - f_start) * (1 - np.exp(-t / tau)) / (1 - np.exp(-dur / tau))
    phase = 2 * np.pi * np.cumsum(f) / sr
    env = np.exp(-t / 0.055)
    y = np.sin(phase) * env
    # soften the very onset (a real foot doesn't click on)
    a = int(0.004 * sr)
    y[:a] *= np.linspace(0.3, 1.0, a)
    return y.astype(np.float32)

def _synth_scuff(sr, dur=0.055):
    """The swish of the foot planting: bandpassed noise, fast attack."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    y = np.random.randn(n).astype(np.float32)
    y = _band(y, sr, "band", 700, 5200)
    env = np.minimum(t / 0.008, 1.0) * np.exp(-t / 0.02)
    return (y * env).astype(np.float32)

_SURFACE_SHAPE = {   # how the contact sample is voiced per material
    "concrete": dict(kind="high", f=320,  g=0.95),
    "grass":    dict(kind="low",  f=2400, g=0.85),
    "wood":     dict(kind="band", f=200,  f2=1100, g=1.0),
    "carpet":   dict(kind="low",  f=1100, g=0.55),
    "snow":     dict(kind="low",  f=1800, g=0.80),
}

# Per-material voicing of the SYNTHESIZED contact layers. The bundled CC0
# samples are dull low thuds (fine for texture), so the identity of a real
# step — sharp heel click + surface crunch — is synthesized per material.
_HIT_PARAMS = {
    "concrete": dict(thump=(115, 60, 0.40), click=(1500, 7500, 0.50), crunch=(300, 4000, 0.35),
                     sample_g=0.25, room=0.12),
    "wood":     dict(thump=(100, 55, 0.45), click=(800, 4500, 0.45),  crunch=(250, 2200, 0.30),
                     sample_g=0.30, room=0.12),
    "grass":    dict(thump=(85, 45, 0.50),  click=(700, 3500, 0.20),  crunch=(300, 2500, 0.40),
                     sample_g=0.30, room=0.07),
    "carpet":   dict(thump=(70, 40, 0.55),  click=(500, 2500, 0.10),  crunch=(250, 1000, 0.20),
                     sample_g=0.20, room=0.05),
    "snow":     dict(thump=(75, 40, 0.50),  click=(600, 3000, 0.25),  crunch=(300, 2000, 0.35),
                     sample_g=0.25, room=0.06),
}

def _noise_burst(sr, dur, f_lo, f_hi, tau, attack=0.004):
    """Bandpassed noise burst with fast attack + exponential decay."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    y = np.random.randn(n).astype(np.float32)
    y = _band(y, sr, "band", f_lo, f_hi)
    env = np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / tau)
    return (y * env).astype(np.float32)

def _room_tail(y, sr, dur=0.13, wet=0.11):
    """Tiny exponential room tail — dry game samples sound fake without it."""
    if not _HAS_SCIPY or len(y) < 64: return y
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    ir = np.random.randn(n).astype(np.float32) * np.exp(-t / 0.03)
    ir = _band(ir, sr, "low", 3200)
    ir /= (np.max(np.abs(ir)) + 1e-9)
    tail = fftconvolve(y, ir)[: len(y)]
    return (y * (1 - wet) + tail * wet).astype(np.float32)

def _footstep_hit(sr, surface, sample_path=None):
    """One complete footstep: scuff -> heel click + crunch + body thump -> texture."""
    p = _HIT_PARAMS.get(surface, _HIT_PARAMS["concrete"])
    dur = 0.42
    total = int(sr * dur)
    y = np.zeros(total, dtype=np.float32)

    # 0. scuff, ~30ms BEFORE the impact instant (t=0 is the impact)
    sc = _synth_scuff(sr)
    o = int(0.030 * sr)
    y[o:o + len(sc)] += sc * 0.25

    # 1. heel click — the sharp strike transient (a few ms, wide band)
    cl_lo, cl_hi, cl_g = p["click"]
    click = _noise_burst(sr, 0.012, cl_lo * random.uniform(0.85, 1.15),
                         cl_hi * random.uniform(0.8, 1.2), tau=0.006, attack=0.0005)
    y[:len(click)] += click * cl_g

    # 2. surface crunch — material grit right after the strike
    cr_lo, cr_hi, cr_g = p["crunch"]
    crunch = _noise_burst(sr, 0.075, cr_lo * random.uniform(0.8, 1.2),
                          cr_hi * random.uniform(0.8, 1.2), tau=0.022)
    y[:len(crunch)] += crunch * cr_g

    # 3. body thump at the impact
    f1, f2, tg = p["thump"]
    th = _synth_thump(sr, f_start=f1 * random.uniform(0.9, 1.15),
                      f_end=f2 * random.uniform(0.9, 1.1),
                      tau=0.030 * random.uniform(0.85, 1.2))
    y[:len(th)] += th * tg

    # heel-toe: a second, smaller weight transfer shortly after (60% of steps)
    if random.random() < 0.6:
        o2 = int(random.uniform(0.055, 0.085) * sr)
        th2 = _synth_thump(sr, f_start=f1 * 0.8, f_end=f2 * 0.9, dur=0.14, tau=0.024)
        y[o2:o2 + len(th2)] += th2 * tg * 0.45
        y[o2:o2 + len(click)] += click * cl_g * 0.25

    # 4. sample texture — the CC0 recording, quiet, adds sole/floor character
    if sample_path is not None and p["sample_g"] > 0:
        s = _load_sample(sample_path, sr)
        n = min(len(s), int(0.16 * sr))
        s = _vary(s[:n].copy(), 0.08)
        y[:len(s)] += s * p["sample_g"] * random.uniform(0.7, 1.1)

    y = _room_tail(y, sr, wet=p["room"])
    peak = np.max(np.abs(y))
    if peak > 1e-9:
        y = (y / peak * 0.92).astype(np.float32)
    return y

def _synth_step(sr=44100):
    """Procedural footstep fallback (kept for QA/imports): delegates to v3 hit."""
    return _footstep_hit(sr, "concrete", None)

# ---------------------------------------------------------------- renderers --

def render_footsteps(events, duration_sec, sr, prompt=""):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    surface = map_surface(prompt)
    files = _bank_files("footsteps", surface)
    for i, e in enumerate(events):
        g = float(getattr(e, "gain", 1.0))
        if g <= 0: continue
        y = _footstep_hit(sr, surface, files[i % len(files)] if files else None)
        jitter = random.uniform(-0.015, 0.015)
        _place(track, y, int((e.start_sec + jitter) * sr), random.uniform(0.8, 1.0) * g)
    return track

def render_typing(events, duration_sec, sr):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    if not events: return track
    gains = [float(getattr(e, "gain", 1.0)) for e in events]
    ag = float(np.mean(gains)) if gains else 1.0
    click = _synth_key_click(sr)
    t0 = min(e.start_sec for e in events)
    t1 = min(max((e.end_sec if e.end_sec and e.end_sec > e.start_sec else e.start_sec + 0.5)
                 for e in events), duration_sec)
    t = t0
    while t < t1 - 0.05:
        _place(track, click, int(t * sr), random.uniform(0.25, 0.5) * ag)
        t += random.uniform(0.13, 0.30)
    for e, g in zip(events, gains):  # accents on detected strikes
        if g <= 0: continue
        _place(track, click, int(e.start_sec * sr), 0.65 * g)
    return track

def _impact_hit(sr, kind=None):
    """Generic foley impact for 'activity' events — 3 physical models so
    unrelatable videos get varied, plausible hits instead of one bank loop.
    Kept QUIET and mostly 'solid': this is a catch-all, so it must stay
    unobtrusive even when it fires on an unusual video."""
    kind = kind or (random.choices(["solid", "wood", "metal"], weights=[70, 22, 8])[0])
    dur = random.uniform(0.16, 0.26)
    total = int(sr * dur)
    y = np.zeros(total, dtype=np.float32)
    if kind == "solid":
        # dense thud against a hard surface (wall, ground, body slam)
        th = _synth_thump(sr, f_start=random.uniform(90, 150), f_end=random.uniform(40, 60),
                          dur=dur * 0.6, tau=random.uniform(0.02, 0.035))
        y[:len(th)] += th * 0.8
        c = _noise_burst(sr, 0.012, random.uniform(600, 1200), random.uniform(2500, 4500), tau=0.006)
        y[:len(c)] += c * 0.30
    elif kind == "wood":
        # woody knock: mid-band resonance, very short decay
        f0 = random.uniform(160, 320)
        t3 = np.linspace(0, dur, total, endpoint=False)
        y[:len(t3)] += (np.sin(2 * np.pi * f0 * t3) * np.exp(-t3 / 0.045)).astype(np.float32) * 0.45
        c = _noise_burst(sr, 0.010, 800, 5000, tau=0.005)
        y[:len(c)] += c * 0.40
        th = _synth_thump(sr, f_start=110, f_end=55, dur=0.08, tau=0.018)
        y[:len(th)] += th * 0.40
    else:  # metal — several detuned partials, ringing
        base = random.uniform(400, 900)
        t3 = np.linspace(0, dur, total, endpoint=False)
        ring = np.zeros(total, dtype=np.float32)
        for ratio, amp in [(1.0, 1.0), (2.76, 0.5), (5.4, 0.3), (1.21, 0.6)]:
            ring += (np.sin(2 * np.pi * base * ratio * t3) * np.exp(-t3 / 0.09) * amp).astype(np.float32)
        y[:len(ring)] += ring * 0.26
        c = _noise_burst(sr, 0.008, 1500, 7000, tau=0.004)
        y[:len(c)] += c * 0.35
    y = _room_tail(y, sr, dur=0.10, wet=0.08)
    peak = np.max(np.abs(y))
    if peak > 1e-9:
        y = (y / peak * 0.82).astype(np.float32)
    return y

def render_activity(events, duration_sec, sr):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    for e in events:
        g = float(getattr(e, "gain", 1.0))
        if g <= 0: continue
        y = _impact_hit(sr)                      # fresh randomized hit every event
        _place(track, y, int(e.start_sec * sr), random.uniform(0.28, 0.42) * g)
    return track

# ------------------------------------------------- ball bounce & punch hits --
# Both are physical contact sounds; pure synthesis lets every hit vary
# naturally (the old generic-bank loop was the "same repetitive sounds" bug).

def _bounce_hit(sr, surface="hard"):
    """Basketball-style bounce: skin click -> rubber body -> hollow ring -> slap.

    A ball contact is a stiff, short collision: sharp attack, quick pitch drop,
    a hollow rubber resonance right after, and a faint slap off the floor.
    """
    dur = 0.32
    total = int(sr * dur)
    y = np.zeros(total, dtype=np.float32)

    # 1. skin/leather click at contact — leads the hit
    click = _noise_burst(sr, 0.014, 900 * random.uniform(0.85, 1.15),
                         3800 * random.uniform(0.8, 1.2), tau=0.007, attack=0.0004)
    y[:len(click)] += click * 0.95

    # 2. rubber body: fast pitch drop, THE bounce signature
    f1 = 130 * random.uniform(0.92, 1.10)
    th = _synth_thump(sr, f_start=f1, f_end=f1 * 0.62, dur=0.11, tau=0.020)
    y[:len(th)] += th * 0.50

    # 3. hollow resonance (ball shell ringing, two detuned modes)
    f0 = (95 * random.uniform(0.9, 1.1))
    t3 = np.linspace(0, 0.18, int(sr * 0.18), endpoint=False)
    ring = (np.sin(2 * np.pi * f0 * t3) + 0.6 * np.sin(2 * np.pi * f0 * 1.22 * t3)) * np.exp(-t3 / 0.045)
    y[:len(ring)] += ring.astype(np.float32) * 0.45

    # 4. floor slap
    slap = _noise_burst(sr, 0.032, 400, 1800, tau=0.013)
    y[:len(slap)] += slap * 0.50

    y = _room_tail(y, sr, dur=0.10, wet=0.07)
    peak = np.max(np.abs(y))
    if peak > 1e-9: y = (y / peak * 0.9).astype(np.float32)
    return y

def _punch_hit(sr, heavy=True):
    """Boxing punch on a heavy bag: glove slap -> deep thud -> bag wobble.

    A heavy-bag punch is: a short leather slap, a VERY fast low thud (the fist
    compressing the bag), then the whole bag swinging as a slow damped wave.
    About a third of punches also rattle the chains.
    """
    dur = 0.55 if heavy else 0.35
    total = int(sr * dur)
    y = np.zeros(total, dtype=np.float32)

    # 1. glove slap — you hear the leather crack FIRST
    slap = _noise_burst(sr, 0.024, 500 * random.uniform(0.8, 1.2),
                        3000 * random.uniform(0.8, 1.2), tau=0.010, attack=0.0006)
    y[:len(slap)] += slap * 0.95

    # 2. body thud — follows the slap
    th = _synth_thump(sr, f_start=170 * random.uniform(0.9, 1.12),
                      f_end=72 * random.uniform(0.9, 1.1), dur=0.10, tau=0.014)
    y[:len(th)] += th * 0.55

    # 3. bag wobble — big slow damped wave starting right after impact
    o = int(0.02 * sr)
    fw = 58 * random.uniform(0.85, 1.15)
    t3 = np.linspace(0, dur - 0.02, total - o, endpoint=False)
    wob = np.sin(2 * np.pi * fw * t3) * np.exp(-t3 / 0.16)
    wob *= (1 - np.minimum(t3 / 0.012, 1.0))          # soft onset
    y[o:] += wob.astype(np.float32) * 0.35

    # 4. chain rattle on some punches
    if random.random() < 0.35:
        o2 = int(random.uniform(0.08, 0.18) * sr)
        rattle = _noise_burst(sr, 0.12, 1800, 5200, tau=0.05, attack=0.002)
        y[o2:o2 + len(rattle)] += rattle * 0.12

    y = _room_tail(y, sr, dur=0.16, wet=0.10)
    peak = np.max(np.abs(y))
    if peak > 1e-9: y = (y / peak * 0.9).astype(np.float32)
    return y

def _synth_clap(sr):
    """Hand clap: two skin slaps with a tiny double-hit."""
    n = int(sr * 0.22)
    y = np.zeros(n, dtype=np.float32)
    for k, (o, g) in enumerate([(0, 1.0), (int(0.012 * sr), 0.5)]):
        c = _noise_burst(sr, 0.030, 800, 6500, tau=0.009, attack=0.0003)
        y[o:o + len(c)] += c * g * 0.6
    y = _room_tail(y, sr, dur=0.09, wet=0.13)
    peak = np.max(np.abs(y))
    if peak > 1e-9: y = (y / peak * 0.9).astype(np.float32)
    return y

def render_bounce(events, duration_sec, sr, prompt=""):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    for i, e in enumerate(events):
        g = float(getattr(e, "gain", 1.0))
        if g <= 0: continue
        y = _bounce_hit(sr)
        jitter = random.uniform(-0.012, 0.012)
        _place(track, y, int((e.start_sec + jitter) * sr), random.uniform(0.8, 1.0) * g)
    return track

def render_punch(events, duration_sec, sr, prompt=""):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    for i, e in enumerate(events):
        g = float(getattr(e, "gain", 1.0))
        if g <= 0: continue
        y = _punch_hit(sr)
        jitter = random.uniform(-0.012, 0.012)
        _place(track, y, int((e.start_sec + jitter) * sr), random.uniform(0.85, 1.05) * g)
    return track

def render_clap(events, duration_sec, sr, prompt=""):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    for i, e in enumerate(events):
        g = float(getattr(e, "gain", 1.0))
        if g <= 0: continue
        y = _synth_clap(sr)
        _place(track, y, int(e.start_sec * sr), random.uniform(0.8, 1.0) * g)
    return track

def render_ambience(events, duration_sec, sr):
    total = int(duration_sec * sr)
    track = np.zeros(total, dtype=np.float32)
    if not events: return track
    rng = np.random.default_rng(7)
    for e in events:
        g = float(getattr(e, "gain", 1.0))
        if g <= 0: continue
        t0 = max(0.0, e.start_sec)
        t1 = min(duration_sec, e.end_sec if e.end_sec > e.start_sec else e.start_sec + 1.0)
        if t1 <= t0: continue
        n = int((t1 - t0) * sr)
        noise = rng.standard_normal(n).astype(np.float32)
        # crude lowpass -> soft room tone
        k = 9
        noise = np.convolve(noise, np.ones(k) / k, mode="same")
        fade = int(0.5 * sr)
        env = np.ones(n, dtype=np.float32)
        if n > 2 * fade:
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
        track[int(t0 * sr):int(t0 * sr) + n] += noise * env * 0.05 * g
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
    bou, pnc, clp = by("bounce"), by("punch"), by("clap")
    if foot: track += render_footsteps(foot, duration_sec, sr, prompt)
    if typ:  track += render_typing(typ, duration_sec, sr)
    if act:  track += render_activity(act, duration_sec, sr)
    if amb:  track += render_ambience(amb, duration_sec, sr)
    if bou:  track += render_bounce(bou, duration_sec, sr, prompt)
    if pnc:  track += render_punch(pnc, duration_sec, sr, prompt)
    if clp:  track += render_clap(clp, duration_sec, sr, prompt)
    peak = np.max(np.abs(track))
    if peak > 1e-6: track = (track / peak * 0.85).astype(np.float32)
    return track

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
