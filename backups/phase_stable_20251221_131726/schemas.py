from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class TrackingRow:
    frame_id: int
    track_id: int
    team_id: Optional[int]
    x_pixel: float
    y_pixel: float
    x_meter: Optional[float]
    y_meter: Optional[float]
    object_class: int # 0=ball, 1=gk, 2=player, 3=ref

@dataclass
class EventRow:
    event_type: str # pass, dribble, shot, etc.
    timestamp: float
    player_id: Optional[str] # "TeamA_10" or just track_id if not identified
    outcome: str # success, fail
    details: Dict[str, Any]

@dataclass
class PlayerStats:
    player_id: str
    team_id: Optional[int]
    minutes_played: float
    distance_covered: float
    sprints: int
    passes_attempted: int
    passes_completed: int
    shots: int
    goals: int
    assists: int
    tackles: int
    interceptions: int
    dribbles: int
    xg: float
