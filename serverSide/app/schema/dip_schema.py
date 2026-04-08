from pydantic import BaseModel
from typing  import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# PER-REP
# ─────────────────────────────────────────────────────────────────────────────

class RepAnalysis(BaseModel):
    rep_number:          int
    start_time_sec:      float   # when descent began
    bottom_time_sec:     float   # deepest point timestamp
    end_time_sec:        float   # arms extended again
    duration_sec:        float   # total rep duration
    descent_sec:         float   # start → bottom
    ascent_sec:          float   # bottom → end
    min_elbow_angle:     float   # lowest elbow angle (smaller = deeper dip)
    elbow_symmetry:      float   # |left_elbow - right_elbow| at deepest point
    torso_lean_angle:    float   # forward lean angle at deepest point
    shoulder_abduction:  float   # avg shoulder abduction at deepest point
    wrist_deviation:     float   # avg wrist deviation at deepest point
    form_score:          int     # 0–100 geometry score
    issues:              List[str]
    advice:              str     # coaching cue for this rep
    fatigue_at_rep:      str     # none | early | moderate | severe


# ─────────────────────────────────────────────────────────────────────────────
# OVERALL
# ─────────────────────────────────────────────────────────────────────────────

class OverallAnalysis(BaseModel):
    total_reps:           int
    estimated_max_reps:   int

    # form scores
    avg_form_score:       int
    min_form_score:       int
    max_form_score:       int

    # depth
    avg_depth_angle:      float   # avg min elbow angle across reps
    best_depth_angle:     float   # lowest (deepest) min elbow angle
    depth_consistency:    float   # std-dev of min elbow angles (lower = more consistent)

    # tempo
    avg_duration_sec:     float
    min_duration_sec:     float
    max_duration_sec:     float
    avg_descent_sec:      float
    avg_ascent_sec:       float

    # posture
    avg_torso_lean:       float
    avg_elbow_symmetry:   float

    # summary
    recurring_issues:     List[str]
    overall_advice:       str

    # fatigue
    fatigue_level:        str    # none | early | moderate | severe
    fatigue_score:        float  # 0–100
    fatigue_tip:          str


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

class DipAnalysisResponse(BaseModel):
    processed_video_url: str
    overall_analysis:    OverallAnalysis
    per_rep_analysis:    List[RepAnalysis]
    summary:             str