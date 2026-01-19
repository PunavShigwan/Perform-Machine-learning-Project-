from pydantic import BaseModel
from typing import List

class RepDetail(BaseModel):
    rep: int
    form: int
    rating: str
    time: float

class PushupAnalysisResponse(BaseModel):
    pushup_count: int
    fatigue: int
    fatigue_level: str
    estimated_range: str
    reps: List[RepDetail]
    output_video_path: str
