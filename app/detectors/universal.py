"""Universal detector — works on ANY video within 4GB VRAM / 8GB RAM.

Strategy (lightweight, no VideoMAE on this laptop):
 1. Try footstep pipeline (YOLO+MediaPipe). If events -> return them.
 2. Fallback: pure-CV motion detector (frame diff + MOG2). No VRAM, CPU only.
    Segments high motion into Event(type="activity") with VLM/CLAP retrieval prompt.
 This covers: typing, door, car, water, crowd, generic object motion.
"""
import cv2, numpy as np
from typing import List
from app.schemas import Event
from app.detectors.footstep import FootstepDetector

PROMPTS = ["person", "door", "car", "dog", "water", "glass", "crowd", "rain", "typing", "keyboard"]

class UniversalDetector:
    name = "universal"
    def __init__(self, sensitivity="medium", yolo_conf=0.5, **kw):
        self.footstep = FootstepDetector(sensitivity=sensitivity, yolo_conf=yolo_conf, **kw)
        from app.detectors.typing import TypingDetector
        self.typing = TypingDetector()
        self.mog = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=25, detectShadows=False)

    def detect(self, video_path: str, fps: float = 30.0, **kw) -> List[Event]:
        # progress_callback is reused across stages — keep it flowing so UI doesn't stick at 99%
        cb = kw.get("progress_callback")
        footstep_events = self.footstep.detect(video_path, fps=fps, **kw)
        # Require alternating gait: at least 2 left+right or >=4 steps; otherwise it's not walking -> try typing
        if len(footstep_events) >= 4:
            lefts = sum(1 for e in footstep_events if "LEFT" in e.event_id)
            rights = sum(1 for e in footstep_events if "RIGHT" in e.event_id)
            if lefts >= 2 and rights >= 2:
                self._release_gpu()
                return footstep_events
        # Typing/hands check BEFORE the POV gait fallback — a hands-on-keyboard
        # clip has rhythmic finger motion that could masquerade as gait otherwise
        typing_events = self.typing.detect(video_path, fps=fps, progress_callback=cb)
        if typing_events:
            self._release_gpu()
            return typing_events
        # POV walking (feet-only selfie shots): gait-locked rhythm, gated against
        # static-camera/hands scenes so it can't hijack non-walking clips
        pov_events = self.footstep.pov_gait_fallback(video_path, fps, cb)
        if pov_events:
            self._release_gpu()
            return pov_events
        # Fallback: motion segmentation (CPU-only, ~50MB RAM)
        motion_events = self._motion_detect(video_path, fps=fps, progress_callback=cb)
        self._release_gpu()
        return motion_events

    def _motion_detect(self, video_path: str, fps: float = 30.0, progress_callback=None) -> List[Event]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        motion_vals = []
        frame_idx = 0
        prev = None
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = self.mog.apply(gray)
            # motion energy = mean of fg mask
            energy = float(np.mean(fg > 0))
            motion_vals.append(energy)
            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(min(frame_idx / total, 1.0))
        cap.release()
        if not motion_vals:
            return []
        # smooth energy, find peaks = activity bursts
        try:
            from scipy.signal import find_peaks
            vals = np.array(motion_vals, dtype=float)
            # normalise
            peaks, _ = find_peaks(vals, prominence=0.02, distance=int(fps*0.4))
            if len(peaks) == 0:
                # No bursts — emit single ambience event for whole clip
                return [Event(event_id="amb_0", type="ambience", start_sec=0.0,
                              end_sec=len(vals)/fps, frame=0, confidence=0.5,
                              prompt="quiet room ambience, foley", duration_hint=len(vals)/fps)]
            events = []
            for i, p in enumerate(peaks):
                sec = p / fps
                events.append(Event(event_id=f"act_{i:04d}", type="activity",
                                    start_sec=float(sec), end_sec=float(sec+0.5), frame=int(p),
                                    timecode=_smpte(p, fps), confidence=float(min(1.0, vals[p]*2)),
                                    prompt="ambient activity sounds, foley, close-mic", duration_hint=0.5))
            return sorted(events, key=lambda e: e.frame)
        except Exception as e:
            print(f"[universal] motion fallback failed: {e}")
            return [Event(event_id="amb_0", type="ambience", start_sec=0.0,
                          end_sec=len(motion_vals)/fps, frame=0, confidence=0.5,
                          prompt="quiet room ambience, foley", duration_hint=len(motion_vals)/fps)]

    def _release_gpu(self):
        """Free VRAM ASAP — critical for RTX 2050 4GB."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import gc; gc.collect()
        except Exception:
            pass

def _smpte(frame, fps):
    ts = frame/fps
    return f"{int(ts//3600):02d}:{int((ts%3600)//60):02d}:{int(ts%60):02d}:{int((ts*fps)%fps):02d}"
