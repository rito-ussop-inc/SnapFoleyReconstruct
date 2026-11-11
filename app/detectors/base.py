"""Base detector interface — footstep is just one plugin."""
from typing import List
from app.schemas import Event

class EventDetectorBase:
    name = "base"
    def detect(self, video_path: str, fps: float = 30.0, **kwargs) -> List[Event]:
        raise NotImplementedError

class ScriptDetector(EventDetectorBase):
    """Podcast script mode: lines -> events, no video needed."""
    name = "script"
    def detect(self, script_path: str, secs_per_line: float = 3.0, **kwargs) -> List[Event]:
        from pathlib import Path
        text = Path(script_path).read_text(errors="ignore")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        events = []
        for i, line in enumerate(lines[:200]):
            events.append(Event(
                event_id=f"evt_{i:04d}",
                type="ambience",
                start_sec=i*secs_per_line,
                end_sec=i*secs_per_line+secs_per_line,
                frame=0,
                confidence=0.6,
                prompt=line[:120],
                duration_hint=secs_per_line,
            ))
        return events
