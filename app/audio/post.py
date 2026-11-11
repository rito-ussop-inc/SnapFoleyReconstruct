"""Audio post — trim/normalize/loop. Fully local, CPU-only, 8GB safe."""
import numpy as np

def trim_silence(y, sr=44100, top_db=30):
    try:
        import librosa
        intervals = librosa.effects.split(y, top_db=top_db)
        if len(intervals) == 0: return y
        start, end = intervals[0][0], intervals[-1][1]
        return y[start:end]
    except Exception:
        return y

def normalize_lufs(y, target_db=-16):
    try:
        import soundfile as sf, pyloudnorm as pyln
        meter = pyln.Meter(sr=44100)
        loud = meter.integrated_loudness(y)
        if np.isinf(loud): return y
        return y * (10**((target_db - loud)/20.0))
    except Exception:
        peak = np.max(np.abs(y)) if len(y) else 0
        return y / peak * 0.8 if peak > 1e-6 else y

def fit_to_duration(y, sr, duration_sec, loop=False):
    need = int(duration_sec * sr)
    if len(y) >= need: return y[:need]
    if loop and len(y) > 0:
        reps = int(np.ceil(need / len(y)))
        return np.tile(y, reps)[:need]
    return np.pad(y, (0, need - len(y)))
