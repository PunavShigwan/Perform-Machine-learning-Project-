from pydantic import BaseModel
from typing import Dict, List, Optional

class StageSummary(BaseModel):
    label: str
    frame_count: int
    percentage: float
    detected: bool
    issues: List[str]
    form_ok: bool

class CleanJerkAnalysisResponse(BaseModel):
    lift_verdict: str
    verdict_reason: str
    total_frames: int
    bad_frames: int
    form_accuracy_percent: float
    stage_summary: Dict[str, StageSummary]
    processed_video_path: Optional[str] 