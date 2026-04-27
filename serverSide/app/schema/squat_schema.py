"""
app/schema/squat_schema.py
Pydantic models for squat API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# ─────────────────────────────────────────────
#  PER-REP DETAIL
# ─────────────────────────────────────────────
class RepDetail(BaseModel):
    rep:        int   = Field(..., description="Rep number (1-indexed)")
    form_score: float = Field(..., description="Average model confidence for this rep (0–100)")
    form_tag:   str   = Field(..., description="'GOOD' or 'BAD'")
    faults:     List[str] = Field(default_factory=list, description="Form faults detected during this rep")


# ─────────────────────────────────────────────
#  VIDEO UPLOAD RESPONSE
# ─────────────────────────────────────────────
class SquatAnalysisResponse(BaseModel):
    """Returned by POST /squat/analyze after full video processing."""

    total_reps:          int   = Field(...,  description="Total squat reps detected")
    good_reps:           int   = Field(...,  description="Reps with good form")
    bad_reps:            int   = Field(...,  description="Reps with bad form")
    form_rate_percent:   float = Field(...,  description="Percentage of reps with good form")
    predicted_max_reps:  str   = Field(...,  description="Estimated max reps predictor string e.g. '~12'")

    rep_log:      List[RepDetail]      = Field(default_factory=list,
                                               description="Per-rep breakdown")
    fault_summary: Dict[str, int]      = Field(default_factory=dict,
                                               description="Fault name → occurrence count across all reps")

    processed_video_url: str           = Field("", description="URL to the annotated output video")
    output_video_path:   str           = Field("", description="Absolute local path to the processed video")


# ─────────────────────────────────────────────
#  LIVE SESSION — STATS SNAPSHOT
# ─────────────────────────────────────────────
class SquatLiveStatsResponse(BaseModel):
    """Returned by GET /squat/live/stats while a session is active."""

    status:               str   = Field("running")
    total_reps:           int   = Field(0)
    good_reps:            int   = Field(0)
    bad_reps:             int   = Field(0)
    form_rate_percent:    float = Field(0.0)
    predicted_max_reps:   str   = Field("—")
    current_phase:        str   = Field("WAIT")
    person_detected:      bool  = Field(False)
    squat_valid:          bool  = Field(False)
    current_form_score:   float = Field(0.0, description="Current frame model confidence (0–100)")
    live_faults:          List[str] = Field(default_factory=list)
    knee_angle:           float = Field(170.0, description="Smoothed average knee angle (degrees)")
    session_duration_sec: float = Field(0.0)


# ─────────────────────────────────────────────
#  LIVE SESSION — STOP / FINAL SUMMARY
# ─────────────────────────────────────────────
class SquatLiveStopResponse(BaseModel):
    """Returned by POST /squat/live/stop after the session ends."""

    status:               str   = Field("stopped")
    total_reps:           int   = Field(0)
    good_reps:            int   = Field(0)
    bad_reps:             int   = Field(0)
    form_rate_percent:    float = Field(0.0)
    predicted_max_reps:      str   = Field("—")
    session_duration_sec: float = Field(0.0)
    rep_log:              List[RepDetail]   = Field(default_factory=list)
    fault_summary:        Dict[str, int]    = Field(default_factory=dict)