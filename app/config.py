"""Config loader for RITO laptop profile. Lazy, no torch at import."""
from pathlib import Path
import yaml

DEFAULT = {
    "app": {"mode": "offline", "port": 7860},
    "hardware": {"profile": "laptop_2050_8gb", "max_workers": 2},
    "models": {"yolo": "yolov8n.pt", "yolo_conf": 0.5, "sensitivity": "medium", "mediapipe_complexity": 1},
    "vlm": {"provider": "local", "model": "HuggingFaceTB/SmolVLM-500M-Instruct", "device": "cpu"},
    "audio": {
        "sample_rate": 44100,
        "retrieval": {"enabled": True, "bank_dir": "audio_bank", "top_k": 3, "score_threshold": 0.75},
        "generation": {"enabled": False, "model": "stable-audio-open", "device": "cpu"},
    },
}

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        # shallow merge
        merged = dict(DEFAULT)
        for k, v in cfg.items():
            merged[k] = {**DEFAULT.get(k, {}), **v} if isinstance(v, dict) else v
        return merged
    return DEFAULT

CONFIG = load_config()
