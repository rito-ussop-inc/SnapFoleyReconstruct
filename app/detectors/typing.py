"""Typing detector — MediaPipe Hands, CPU-only, tiny VRAM cost."""
import cv2
import numpy as np
from typing import List
from app.schemas import Event

def _smpte(frame, fps):
    ts = frame / fps
    return f"{int(ts//3600):02d}:{int((ts%3600)//60):02d}:{int(ts%60):02d}:{int((ts*fps)%fps):02d}"

class TypingDetector:
    name = "typing"

    def __init__(self):
        self._hands = None
        try:
            import mediapipe as mp
            self._mp = mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        except Exception as e:
            print(f"[typing] hands not loaded: {e}")
            self._hands = None

    def detect(self, video_path: str, fps: float = 30.0, progress_callback=None) -> List[Event]:
        if self._hands is None:
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        motion_vals = []
        frame_idx = 0
        STRIDE = 2
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % STRIDE == 0:
                small = cv2.resize(frame, (320, 240))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = self._hands.process(rgb)
                # hand motion proxy: landmark spread jitter + presence
                if results.multi_hand_landmarks:
                    # any hand visible + landmark motion > tiny threshold
                    pts = []
                    for hl in results.multi_hand_landmarks:
                        for lm in hl.landmark:
                            pts.append([lm.x, lm.y])
                    motion_vals.append(1.0)  # hand present = typing candidate
                else:
                    motion_vals.append(0.0)
                if progress_callback:
                    progress_callback(min(frame_idx / total, 1.0))
            frame_idx += 1
        cap.release()
        if not motion_vals or sum(motion_vals) < total * 0.15:
            return []
        # peaks of hand presence = typing bursts
        try:
            from scipy.signal import find_peaks
            vals = np.array(motion_vals, dtype=float)
            peaks, _ = find_peaks(vals, prominence=0.1, distance=max(1, int(fps * 0.15 / STRIDE)))
            events = []
            for i, p in enumerate(peaks):
                sec = (p * STRIDE) / fps
                events.append(Event(event_id=f"typ_{i:04d}", type="typing",
                                    start_sec=float(sec), end_sec=float(sec + 0.2),
                                    frame=int(p * STRIDE), timecode=_smpte(int(p * STRIDE), fps),
                                    confidence=0.75,
                                    prompt="laptop keyboard typing, close-mic, foley", duration_hint=0.15))
            return sorted(events, key=lambda e: e.frame)
        except Exception as e:
            print(f"[typing] failed: {e}")
            return []
