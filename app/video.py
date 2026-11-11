"""Annotated video writer — lightweight, CPU-safe. Draws footstep strikes."""
import cv2
from app.schemas import Event
from typing import List

def create_annotated_video(input_path, events: List[Event], output_path, progress_callback=None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        return False
    by_frame = {}
    for e in events:
        by_frame.setdefault(int(e.frame), []).append(e)
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in by_frame:
            cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
            for i, e in enumerate(by_frame[frame_idx]):
                color = (0, 255, 0) if e.type == "footstep" else (255, 255, 0)
                label = f"t={e.start_sec:.2f}s {e.type.upper()} {e.timecode}"
                cv2.putText(frame, label, (12, 22 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.circle(frame, (width//2, height-60), 18, color, -1)
        cv2.putText(frame, f"frame {frame_idx}/{total}", (10, height-14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        out.write(frame)
        frame_idx += 1
        if progress_callback and frame_idx % 5 == 0:
            progress_callback(min(frame_idx/total, 1.0))
    cap.release(); out.release()
    return True
