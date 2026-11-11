"""
Retrieval backend — REPLACES sound_agent.py (YouTube download).
Winning for 8GB: CLAP base (~150MB) + FAISS CPU. No internet.
Bank: audio_bank/*.wav with metadata.json (prompt, type).
If bank missing, falls back to bundled footstep mp3 in audio/.
"""
import os, json, numpy as np
from pathlib import Path

class RetrievalBackend:
    def __init__(self, bank_dir="audio_bank", top_k=3):
        self.bank_dir = Path(bank_dir)
        self.top_k = top_k
        self._model = None
        self._processor = None
        self._index = None
        self._metas = []

    def _load(self):
        if self._index is not None: return True
        try:
            import faiss
            idx_path = str(self.bank_dir / "index.faiss")
            meta_path = str(self.bank_dir / "metadata.json")
            if os.path.exists(idx_path) and os.path.exists(meta_path):
                self._index = faiss.read_index(idx_path)
                self._metas = json.loads(Path(meta_path).read_text())
                return True
        except Exception as e:
            print(f"[retrieval] no index: {e}")
        return False

    def search(self, prompt: str):
        """Return (path, score). Falls back to bundled footstep."""
        if self._load() and self._index is not None:
            try:
                from transformers import ClapProcessor, ClapModel
                import torch
                if self._processor is None:
                    self._processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
                    self._model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
                inputs = self._processor(text=[prompt], return_tensors="pt")
                with torch.no_grad():
                    emb = self._model.get_text_features(**inputs)
                scores, idx = self._index.search(emb.cpu().numpy(), self.top_k)
                best = int(idx[0][0])
                meta = self._metas[best]
                p = str(self.bank_dir / meta["file"])
                return p, float(scores[0][0])
            except Exception as e:
                print(f"[retrieval] search failed: {e}")
        # Fallback — bundled file, always present on this laptop
        for cand in ["audio/Footsteps on Gravel Path Outdoor.mp3", "audio_bank/footsteps.wav"]:
            if os.path.exists(cand):
                return cand, 0.6
        return None, 0.0
