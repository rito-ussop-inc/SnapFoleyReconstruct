"""
Footstep detector — slimmed version of real.py HybridFootstepDetectionPipeline.
Winning tuning for 8GB RAM / RTX 2050 4GB:
 - YOLOv8n only, MediaPipe complexity 1
 - max_workers=2 elsewhere, ffmpeg threads 2
 - no streamlit, lazy imports so QA loop can import without GPU
"""
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from app.schemas import Event

try:
    import mediapipe as mp
except ImportError:
    mp = None

THRESHOLDS = {
    "low":    {"prominence": 0.04, "min_interval": 0.5},
    "medium": {"prominence": 0.03, "min_interval": 0.45},
    "high":   {"prominence": 0.025, "min_interval": 0.4},
}

class FootstepDetector:
    name = "footstep"
    LEFT_HEEL, RIGHT_HEEL = 29, 30

    def __init__(self, sensitivity="medium", yolo_conf=0.5, model="yolov8n.pt", mp_complexity=1):
        self.thresh = THRESHOLDS.get(sensitivity, THRESHOLDS["medium"])
        self.yolo_conf = yolo_conf
        self.model_name = model
        self.mp_complexity = mp_complexity
        self._yolo = None
        self._pose = None
        self._hand_gate = None
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self.model_name)
        except Exception:
            self._yolo = None
        if mp is not None:
            try:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False, model_complexity=self.mp_complexity,
                    smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except Exception:
                self._pose = None
            try:
                self._hand_gate = mp.solutions.hands.Hands(
                    static_image_mode=True, max_num_hands=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except Exception:
                self._hand_gate = None

    def _estimate_pose(self, frame, bbox):
        if self._pose is None: return None
        x1,y1,x2,y2 = bbox
        pad = 20
        x1, y1 = max(0,x1-pad), max(0,y1-pad)
        x2, y2 = min(frame.shape[1],x2+pad), min(frame.shape[0],y2+pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark:
                lm.x = (lm.x*(x2-x1)+x1)/frame.shape[1]
                lm.y = (lm.y*(y2-y1)+y1)/frame.shape[0]
        return res

    def detect(self, video_path: str, fps: float = 30.0, progress_callback=None) -> list[Event]:
        """Skeleton-based footsteps only (third-person full-body views).

        POV/selfie walking has no usable skeleton; that case is handled by
        `pov_gait_fallback`, which the UniversalDetector runs AFTER the typing
        check so hands-on-keyboard clips can't be misread as footsteps.
        """
        if self._yolo is None or self._pose is None:
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        left_pos, right_pos, confs = {}, {}, {}
        frame_idx = 0
        STRIDE = 3  # process every 3rd frame — 3x faster on RTX 2050 4GB
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % STRIDE == 0:
                frame_small = cv2.resize(frame, (640, 480))
                res = self._yolo.track(frame_small, persist=True, conf=self.yolo_conf, classes=[0], verbose=False, imgsz=640)
                if len(res)>0 and res[0].boxes.id is not None:
                    for box, tid in zip(res[0].boxes.xyxy, res[0].boxes.id):
                        t = int(tid.item())
                        if t not in left_pos:
                            left_pos[t]=[np.nan]*total; right_pos[t]=[np.nan]*total; confs[t]=[0.0]*total
                        pose_res = self._estimate_pose(frame_small, tuple(map(int, box.cpu().numpy())))
                        if pose_res and pose_res.pose_landmarks:
                            lm = pose_res.pose_landmarks.landmark
                            left_pos[t][frame_idx] = lm[self.LEFT_HEEL].y
                            right_pos[t][frame_idx] = lm[self.RIGHT_HEEL].y
                            confs[t][frame_idx] = (lm[self.LEFT_HEEL].visibility+lm[self.RIGHT_HEEL].visibility)/2
                if progress_callback:
                    progress_callback(min(frame_idx/total, 1.0))
            frame_idx += 1
        cap.release()
        events = []
        for t in left_pos:
            # Skip if no valid pose detections (all NaN / very few frames) — common on non-walking clips
            valid = sum(1 for v in left_pos[t] if not (v != v))  # NaN check without importing numpy NaN pitfalls
            if valid < 5:  # NaN-safe recount
                import numpy as _np
                valid = int(_np.count_nonzero(~_np.isnan(_np.array(left_pos[t], dtype=float))))
            if valid < 5:
                continue
            lp = pd.Series(left_pos[t]).interpolate().bfill().ffill().values
            rp = pd.Series(right_pos[t]).interpolate().bfill().ffill().values
            # If still NaN (e.g. zero detections), skip instead of crashing savgol
            try:
                import numpy as _np
                if _np.isnan(lp).any() or _np.isnan(rp).any():
                    lp = _np.nan_to_num(lp, nan=0.5)
                    rp = _np.nan_to_num(rp, nan=0.5)
                win = min(11, len(lp) if len(lp)%2==1 else len(lp)-1)
                if win >= 3:
                    lp = savgol_filter(lp, win, 2); rp = savgol_filter(rp, win, 2)
            except Exception:
                pass
            for name, series in [("LEFT", lp), ("RIGHT", rp)]:
                # Require heel near ground (y>0.6) to avoid hand/closeup false positives (e.g. typing)
                peaks, _ = find_peaks(series, prominence=self.thresh["prominence"], distance=int(fps*self.thresh["min_interval"]), height=0.6)
                for p in peaks:
                    conf = float(confs[t][int(p)]) if confs[t][int(p)]>0 else 0.5
                    if conf < 0.6:
                        continue
                    events.append(Event(
                        event_id=f"{t}_{name}_{int(p)}", type="footstep",
                        start_sec=p/fps, end_sec=p/fps+0.3, frame=int(p),
                        timecode=_smpte(p, fps), confidence=conf,
                        prompt="footsteps on surface, close-mic, foley", duration_hint=0.5, person_id=t))
        events = sorted(events, key=lambda e: e.frame)
        return events

    def ball_present(self, video_path: str, samples: int = 20, conf: float = 0.10) -> float:
        """Fraction of sampled frames containing a sports ball (YOLO COCO class 32).

        Used as a hard gate: a visible ball means hands-on-ball play (dribbling,
        juggling), which must never be classified as typing or walking.
        """
        if self._yolo is None:
            self._load()
        if self._yolo is None:
            return 0.0
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        hits, seen = 0, 0
        for idx in np.linspace(0, max(0, total - 1), samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok: continue
            seen += 1
            try:
                res = self._yolo.predict(frame, conf=conf, verbose=False)[0]
                for box in res.boxes:
                    cls = int(box.cls[0]) if hasattr(box, "cls") else -1
                    if cls == 32:  # COCO: sports ball
                        hits += 1
                        break
            except Exception:
                continue
        cap.release()
        return (hits / seen) if seen else 0.0

    def pov_gait_fallback(self, video_path: str, fps: float, progress_callback=None) -> list[Event]:
        """Detect footsteps in POV walking videos via gait-rhythm locking.

        POV shots (camera pointing at the walker's own feet) have no full-body
        skeleton, but the camera bounces once per step. We extract the per-frame
        global vertical shift (phase correlation) + motion energy, find the
        dominant cadence in 0.8–3.0 Hz, phase-align a step grid to it, and snap
        every step to the nearest vertical-bounce extremum.

        Guarded so it can't fire on hands-on-keyboard close-ups:
          - the scene must move coherently (low phase-correlation response —
            static-camera clips with small moving objects score ~0.9+, walking
            POVs ~0.5), and
          - MediaPipe Hands must NOT be seeing hands (checked every 8th frame).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return []
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        energies, vertical, prev = [], [], None
        resps, hand_hits, hand_samples = [], 0, 0
        # ground-region stats (sampled every 10th frame): pick the surface so
        # the foley voicing matches what the walker is standing on, and add a
        # leaf-crunch layer when fallen leaves are visible (autumn POV walks).
        green_frac, leaf_frac, surf_samples = 0.0, 0.0, 0
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            small = cv2.resize(frame, (160, int(160 * frame.shape[0] / max(1, frame.shape[1]))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                energies.append(float(np.mean(cv2.absdiff(gray, prev))) / 255.0)
                try:
                    (dx, dy), resp = cv2.phaseCorrelate(prev, gray)
                    vertical.append(float(dy))
                    resps.append(float(resp))
                except Exception:
                    vertical.append(0.0); resps.append(0.0)
            else:
                energies.append(0.0); vertical.append(0.0); resps.append(0.0)
            # hand gate: a hands close-up (typing etc.) is not a walking POV
            if frame_idx % 8 == 0 and self._hand_gate is not None:
                hs = cv2.resize(frame, (256, int(256 * frame.shape[0] / max(1, frame.shape[1]))))
                rgb = cv2.cvtColor(hs, cv2.COLOR_BGR2RGB)
                try:
                    hres = self._hand_gate.process(rgb)
                    if hres.multi_hand_landmarks: hand_hits += 1
                except Exception:
                    pass
                hand_samples += 1
            # ground look: bottom third of the frame, sampled sparsely
            if frame_idx % 10 == 0:
                h = small.shape[0]
                ground = small[int(h * 0.65):, :].astype(np.float32)
                b, gch, r = ground[:, :, 0], ground[:, :, 1], ground[:, :, 2]
                mx = np.maximum(b, np.maximum(gch, r)) + 1e-6
                sat = (mx - np.minimum(b, np.minimum(gch, r))) / mx
                green = float(np.mean((gch > r * 1.12) & (gch > b * 1.12) & (sat > 0.15)))
                leafy = float(np.mean((r > b * 1.25) & (r > 70) & (gch > b) & (sat > 0.25)))
                green_frac += green; leaf_frac += leafy; surf_samples += 1
            prev = gray
            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(min(frame_idx / total, 1.0))
        cap.release()
        dur = len(energies) / fps
        if dur < 3.0:  # need >= ~3s to lock a gait rhythm
            return []
        if hand_samples and hand_hits / hand_samples > 0.10:
            return []  # hands dominate the frame -> not a walking POV
        if resps and float(np.mean(resps)) > 0.75:
            return []  # mostly-static camera with localized motion -> not POV walking
        # surface from the ground stats
        if surf_samples:
            gf, lf = green_frac / surf_samples, leaf_frac / surf_samples
        else:
            gf = lf = 0.0
        if gf > 0.22:
            step_prompt = "footsteps on grass, close-mic, foley"
        elif lf > 0.015:
            step_prompt = "footsteps on brick pavement with fallen leaves, close-mic, foley"
        else:
            step_prompt = "footsteps on pavement, close-mic, foley"

        def cadence(x, fmin=0.8, fmax=3.0):
            x = np.asarray(x, dtype=float)
            x = x - np.convolve(x, np.ones(max(3, int(fps * 0.8))) / max(3, int(fps * 0.8)), mode="same")
            w = np.hanning(len(x))
            F = np.abs(np.fft.rfft(x * w))
            freqs = np.fft.rfftfreq(len(x), 1.0 / fps)
            band = (freqs >= fmin) & (freqs <= fmax)
            if not band.any() or len(F) < 4: return None
            fb, Fb = freqs[band], F[band]
            k = int(np.argmax(Fb))
            snr = float(Fb[k]) / (float(np.median(Fb)) + 1e-9)
            return float(fb[k]), snr

        cv_ = cadence(vertical)   # vertical bounce: the strong, clean signal
        ce = cadence(energies)    # motion energy: corroboration
        cands = []
        if cv_ and cv_[1] >= 2.5: cands.append((cv_[0], cv_[1], 3.0))       # weight vertical x3
        if ce and ce[1] >= 2.5: cands.append((ce[0], ce[1], 1.0))
        if not cands: return []
        f0 = max(cands, key=lambda c: c[1] * c[2])[0]

        det_v = np.asarray(vertical, dtype=float)
        det_v = det_v - np.convolve(det_v, np.ones(max(3, int(fps * 0.8))) / max(3, int(fps * 0.8)), mode="same")
        period = 1.0 / f0
        cyc = int(round(period * fps))
        if cyc < 2 or len(det_v) < 3 * cyc: return []
        # fold by period -> phase of maximum bounce
        folds = [det_v[i::cyc] for i in range(cyc)]
        L = min(len(f) for f in folds)
        fold = np.mean([f[:L] for f in folds], axis=0)
        phase = int(np.argmax(fold))  # step lands at the biggest bounce extremum
        t0 = phase / fps
        n_steps = int((len(det_v) / fps - t0) / period)
        if n_steps < 4: return []

        events = []
        half_win = max(2, int(0.15 * fps))
        lead = 2  # frames: place the strike just before the dip peak (perceived-sync correction)
        for k in range(n_steps):
            t_k = t0 + k * period
            c = int(round(t_k * fps))
            lo, hi = max(0, c - half_win), min(len(det_v), c + half_win + 1)
            if hi - lo < 3: continue
            seg = det_v[lo:hi]
            # snap to the sharpest vertical extremum (impact) inside the window
            p = lo + int(np.argmax(np.abs(seg - np.mean(seg))))
            conf = float(min(1.0, 0.55 + abs(float(det_v[p])) * 2.0))
            lr = "LEFT" if k % 2 == 0 else "RIGHT"
            sf = max(0, p - lead)
            events.append(Event(
                event_id=f"pov_{lr}_{p}", type="footstep",
                start_sec=sf / fps, end_sec=sf / fps + 0.25, frame=int(p),
                timecode=_smpte(p, fps), confidence=conf,
                prompt=step_prompt, duration_hint=0.4,
                person_id=0))
        # de-dup steps that snapped to the same frame
        seen, out = set(), []
        for e in events:
            if e.frame in seen: continue
            seen.add(e.frame); out.append(e)
        return sorted(out, key=lambda e: e.frame)

def _smpte(frame, fps):
    ts = frame/fps
    return f"{int(ts//3600):02d}:{int((ts%3600)//60):02d}:{int(ts%60):02d}:{int((ts*fps)%fps):02d}"
