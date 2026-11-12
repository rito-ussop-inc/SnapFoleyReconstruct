"""Universal detector — works on ANY video within 4GB VRAM / 8GB RAM.

Strategy (lightweight, no VideoMAE on this laptop):
 1. Try footstep pipeline (YOLO+MediaPipe). If walking gait -> return footsteps.
 2. Ball check (YOLO sports-ball): visible ball -> hands-on-ball play; typing
    is skipped and motion events become "bounce" hits.
 3. Typing (MediaPipe hands + fingertip-velocity strikes), travel-gated.
 4. POV walking gait fallback (gated).
 5. Motion segmentation -> impact classifier picks the sound family:
    ball bounce / punch / generic activity / ambience.
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
        # Ball check ONCE (20 cheap YOLO frames at conf 0.10 — dark balls are
        # nearly invisible at standard conf). A visible ball means dribbling/
        # juggling: typing must not fire and impacts must sound like ball hits.
        ball_ratio = self.footstep.ball_present(video_path)
        if ball_ratio < 0.08:
            typing_events = self.typing.detect(video_path, fps=fps, progress_callback=cb)
            if typing_events:
                self._release_gpu()
                return typing_events
        # POV walking (feet-only selfie shots): gait-locked rhythm, gated against
        # static-camera/hands scenes so it can't hijack non-walking clips
        if ball_ratio < 0.08:
            pov_events = self.footstep.pov_gait_fallback(video_path, fps, cb)
            if pov_events:
                self._release_gpu()
                return pov_events
        # Fallback: motion segmentation (CPU-only, ~50MB RAM) + impact classifier
        motion_events, curves = self._motion_detect(video_path, fps=fps, progress_callback=cb)
        events = self._classify_impacts(motion_events, curves, ball_ratio, fps)
        self._release_gpu()
        return events

    def _classify_impacts(self, events: List[Event], curves: dict, ball_ratio: float, fps: float) -> List[Event]:
        """Pick the sound family for motion bursts: ball bounce / punch / activity.

        - ball visible            -> bounce, timed to LOWER-FRAME ground contacts
        - rhythmic bursts 1-8/s   -> punch (boxing bag work, knocks, impact play)
        - anything else           -> generic activity
        """
        if not events:
            return events
        if ball_ratio >= 0.08:
            # Juggling/dribbling: the total-motion curve washes contacts out
            # (the whole body moves), but the LOWER-FRAME curve spikes on every
            # ground contact / low catch. Build bounce events from ITS peaks.
            low_e = curves.get("lower")
            lp = None
            if low_e is not None and len(low_e) > 12:
                try:
                    from scipy.signal import find_peaks
                    # 0.25: handheld-camera motion lifts the whole lower-frame
                    # baseline, so contacts only clear ~25% of the p90 band;
                    # below this the peak set is stable (not threshold noise).
                    prom = max(0.008, float(np.percentile(low_e, 90)) * 0.25)
                    cand, _ = find_peaks(low_e, prominence=prom, distance=max(2, int(fps * 0.22)))
                    if len(cand) >= 3:
                        lp = cand
                except Exception:
                    lp = None
            if lp is not None:
                out = []
                for i, p in enumerate(lp):
                    sec = float(p / fps)
                    out.append(Event(event_id=f"bou_{i:04d}", type="bounce",
                                     start_sec=sec, end_sec=sec + 0.35, frame=int(p),
                                     timecode=_smpte(int(p), fps), confidence=0.8,
                                     prompt="ball bounce impact, close-mic, foley",
                                     duration_hint=0.35))
                return out
            # fallback: snap the total-motion events to lower-frame energy max
            low_e = curves.get("lower")
            for i, e in enumerate(events):
                e.type = "bounce"
                e.prompt = "ball bounce impact, close-mic, foley"
                e.event_id = f"bou_{i:04d}"
                e.duration_hint = 0.35
                if low_e is not None:
                    e.frame = self._snap(e.frame, low_e, int(0.25 * fps))
                    e.start_sec = e.frame / fps
                    e.end_sec = e.start_sec + 0.35
                    e.timecode = _smpte(e.frame, fps)
            return sorted(events, key=lambda e: e.frame)
        ts = [e.start_sec for e in events]
        if len(ts) >= 3:
            iv = np.diff(ts)
            rate = 1.0 / max(float(np.mean(iv)), 1e-6)
            cv = float(np.std(iv) / max(np.mean(iv), 1e-6))
            if 1.0 <= rate <= 8.0 and cv < 0.75:
                for i, e in enumerate(events):
                    e.type = "punch"
                    e.prompt = "punch impact on heavy bag, foley"
                    e.event_id = f"pnc_{i:04d}"
                    e.duration_hint = 0.5
                return sorted(events, key=lambda e: e.frame)
        return events

    def _snap(self, frame: int, energy: np.ndarray, half_win: int) -> int:
        """Snap a frame to the sharpest energy extremum within +-half_win."""
        lo, hi = max(0, frame - half_win), min(len(energy), frame + half_win + 1)
        if hi - lo < 3:
            return frame
        seg = energy[lo:hi]
        return int(lo + int(np.argmax(seg)))

    def _motion_detect(self, video_path: str, fps: float = 30.0, progress_callback=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], {}
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        motion_vals, lower_vals = [], []
        frame_idx = 0
        prev = None
        prev_low = None
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (192, int(192 * frame.shape[0] / max(1, frame.shape[1]))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            h = gray.shape[0]
            low = gray[int(h * 0.62):, :]
            fg = self.mog.apply(gray)
            motion_vals.append(float(np.mean(fg > 0)))
            if prev_low is not None:
                lower_vals.append(float(np.mean(cv2.absdiff(low, prev_low))) / 255.0)
            else:
                lower_vals.append(0.0)
            prev_low = low
            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(min(frame_idx / total, 1.0))
        cap.release()
        if not motion_vals:
            return [], {}
        curves = {"lower": np.asarray(lower_vals, dtype=float)}
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
                              prompt="quiet room ambience, foley", duration_hint=len(vals)/fps)], curves
            events = []
            for i, p in enumerate(peaks):
                sec = p / fps
                events.append(Event(event_id=f"act_{i:04d}", type="activity",
                                    start_sec=float(sec), end_sec=float(sec+0.5), frame=int(p),
                                    timecode=_smpte(p, fps), confidence=float(min(1.0, vals[p]*2)),
                                    prompt="ambient activity sounds, foley, close-mic", duration_hint=0.5))
            return sorted(events, key=lambda e: e.frame), curves
        except Exception as e:
            print(f"[universal] motion fallback failed: {e}")
            return [Event(event_id="amb_0", type="ambience", start_sec=0.0,
                          end_sec=len(motion_vals)/fps, frame=0, confidence=0.5,
                          prompt="quiet room ambience, foley", duration_hint=len(motion_vals)/fps)], curves

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
