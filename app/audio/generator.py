"""Generation backend — Stable Audio Open local. DISABLED by default on 8GB/4GB VRAM."""
import numpy as np

class GenerationBackend:
    def __init__(self, enabled=False, device="cpu"):
        self.enabled = enabled
        self.device = device

    def generate(self, prompt: str, duration: float = 4.0):
        if not self.enabled:
            return None  # retrieval handles it
        try:
            # import heavy deps lazily
            from transformers import pipeline  # placeholder for stable-audio-open
            return None
        except Exception:
            return None
