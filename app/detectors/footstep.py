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
        return sorted(events, key=lambda e: e.frame)

def _smpte(frame, fps):
    ts = frame/fps
    return f"{int(ts//3600):02d}:{int((ts%3600)//60):02d}:{int(ts%60):02d}:{int((ts*fps)%fps):02d}"
