"""
app/schema/clean_jerk_schema.py
"""
from pydantic import BaseModel
from typing import Dict, List, Optional


class StageSummary(BaseModel):
    label:            str
    frame_count:      int
    percentage:       float
    detected:         bool
    top_issues:       List[str]
    form_ok:          bool
    duration_seconds: float
    start_frame:      Optional[int] = None
    end_frame:        Optional[int] = None


class PhaseTimingAnalysis(BaseModel):
    phase_durations_seconds:     Dict[str, float]
    phase_percentages:           Dict[str, float]
    total_lift_duration_seconds: float
    phase_transitions_seconds:   Dict[str, Optional[float]]
    fastest_phase:               str
    slowest_phase:               str


class PhaseFormBreakdown(BaseModel):
    clean_to_shoulder: float
    jerk_overhead:     float
    release_finish:    float


class CleanJerkAnalysisResponse(BaseModel):
    machine_call:          str     # "GREEN LIGHT" | "MAJORITY GREEN" | "RED LIGHT"
    machine_reason:        str
    form_accuracy_percent: float
    form_frozen:           bool    # True = form % was frozen at RELEASE_FINISH
    phase_form:            PhaseFormBreakdown
    pass_threshold:        float
    total_frames:          int
    active_lift_frames:    int
    bad_frames:            int
    stage_summary:         Dict[str, StageSummary]
    phase_timing:          PhaseTimingAnalysis
    processed_video_path:  Optional[str] = None

    class Config:
        extra = "allow"   # prevents crash if service returns extra debug keys