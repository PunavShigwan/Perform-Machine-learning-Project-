from pydantic import BaseModel
from typing import Dict, Optional

class CleanJerkAnalysisResponse(BaseModel):
    total_frames: int
    bad_frames: int
    form_accuracy_percent: float
    phase_distribution: Dict[str, int]
    processed_video_path: Optional[str]