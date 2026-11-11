"""Universal schemas — replaces FootstepEvent with generic Event/Timeline."""
from pydantic import BaseModel, Field
from typing import List, Optional

class Event(BaseModel):
    event_id: str = Field(default="", description="unique id")
    type: str = Field(default="footstep", description="footstep|door_creak|car_pass|splash|ambience|generic")
    start_sec: float = 0.0
    end_sec: float = 0.0
    frame: int = 0
    timecode: str = ""
    confidence: float = 0.5
    prompt: str = ""          # text prompt for retrieval/generation
    duration_hint: float = 0.5
    person_id: Optional[int] = None

class Clip(BaseModel):
    event_id: str
    stem_path: str = ""
    gain_db: float = 0.0
    fade_in: float = 0.05
    loop: bool = False
    backend: str = "retrieval"  # retrieval|generation|fallback
    score: float = 0.0

class Track(BaseModel):
    name: str = "sfx"  # sfx|ambience|foley
    clips: List[Clip] = []

class Timeline(BaseModel):
    task_id: str
    duration_sec: float = 0.0
    tracks: List[Track] = []

class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str = "offline"
    profile: str = "laptop_2050_8gb"
    models: dict = {}
    offline_guarantee: bool = True
