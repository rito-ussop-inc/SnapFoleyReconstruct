"""Typing detector — MediaPipe Hands + landmark-velocity keystrikes, CPU-only.

Hands visible >15% of the clip = hands-dominant scene. Keystrike events come
from per-frame landmark velocity peaks (fingers accelerating between keys), so
events line up with actual strikes. A still, visible hand produces no strikes.
"""
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
        presence, motions, prev_pts = [], [], None
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
            if results.multi_hand_landmarks:
                pts = []
                for hl in results.multi_hand_landmarks:
                    for lm in hl.landmark:
                        pts.append((lm.x, lm.y))
                m = 0.0
                if prev_pts is not None and len(prev_pts) == len(pts):
                    d = np.abs(np.asarray(pts, dtype=float) - np.asarray(prev_pts, dtype=float)).sum(axis=1)
                    m = float(d.mean())
                motions.append(m); presence.append(1.0); prev_pts = pts
            else:
                motions.append(0.0); presence.append(0.0); prev_pts = None
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(min(frame_idx / total, 1.0))
            frame_idx += 1
        cap.release()
        if not presence or float(np.mean(presence)) < 0.15:
            return []  # hands not a consistent subject -> not a typing/hands clip
        mv = np.asarray(motions, dtype=float)
        mv_s = np.convolve(mv, np.ones(3) / 3, mode="same")
        active = mv_s[mv_s > 0]
        if active.size == 0:
            return []
        from scipy.signal import find_peaks
        prom = max(0.01, float(np.percentile(active, 50)) * 0.6)
        peaks, props = find_peaks(mv_s, prominence=prom, distance=max(2, int(fps * 0.08)))
        if len(peaks) < 6:
            return []  # hands visible but no strike rhythm (resting/holding)
        proms = props.get("prominences", [0.05] * len(peaks))
        events = []
        for i, p in enumerate(peaks):
            sec = p / fps
            events.append(Event(event_id=f"typ_{i:04d}", type="typing",
                                start_sec=float(sec), end_sec=float(sec + 0.15),
                                frame=int(p), timecode=_smpte(int(p), fps),
                                confidence=float(min(1.0, 0.65 + float(proms[i]) * 2.0)),
                                prompt="laptop keyboard typing, close-mic, foley", duration_hint=0.12))
        return sorted(events, key=lambda e: e.frame)
