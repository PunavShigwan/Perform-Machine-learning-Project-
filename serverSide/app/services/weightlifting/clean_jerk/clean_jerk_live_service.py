"""
clean_jerk_live_service.py
===========================
LiveCleanJerkSession — manages webcam capture, pose analysis,
phase detection (bar-touch → clean → jerk → release),
form scoring, fatigue tracking, and wrong-exercise detection
in a background thread.

Best webcam setup:
  - Camera angle : SIDE-ON (90° to athlete) — front/back breaks pose estimation
  - Framing      : Full body head-to-feet visible throughout the lift
  - Lighting     : Bright, even — no strong backlight

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① SETUP_BAR_TOUCH   — athlete grips bar on the floor (wrists ≥ knee level)
② CLEAN_TO_SHOULDER — bar pulled from floor, racked on shoulders, stand complete
③ JERK_OVERHEAD     — bar driven overhead, elbows locked out
④ RELEASE_FINISH    — bar held overhead for judge signal (form FROZEN here)

Once RELEASE_FINISH is detected the form score is frozen; no further
updates occur even if the athlete lowers the bar.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
import logging
from collections import deque, Counter
from typing import Optional, Dict, List, Tuple, Any

print("📦 clean_jerk_live_service.py loaded")

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | cj_live | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_jerk_live_service")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG — stage detection gates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Stage detection
CLEAN_ELBOW_MIN    = 40
CLEAN_ELBOW_MAX    = 105
CLEAN_KNEE_MIN     = 55
CLEAN_HIP_MIN      = 30
JERK_RAISE_MIN     = 135
JERK_ELBOW_MIN     = 135
RELEASE_RAISE_MIN  = 148
RELEASE_ELBOW_MIN  = 150

# Form-fault gates
CLEAN_HIP_FAULT_LO   = 70
CLEAN_HIP_FAULT_HI   = 148
CLEAN_KNEE_FAULT_LO  = 70
CLEAN_KNEE_FAULT_HI  = 145
JERK_ELBOW_FAULT     = 150
JERK_RAISE_FAULT     = 135
RELEASE_ELBOW_FAULT  = 148
RELEASE_RAISE_FAULT  = 145

# Smoothing / confirmation
STAGE_WINDOW         = 7          # frames for majority-vote stage smoothing
FAULT_HOLD           = 12         # consecutive frames before a fault is logged
BAR_TOUCH_CONFIRM    = 3          # votes in deque(5) to confirm bar touch

# Scoring thresholds
PASS_THRESHOLD     = 65.0
MAJORITY_THRESHOLD = 50.0

# Wrong-exercise detection
WRONG_EXERCISE_FRAMES   = 20      # consecutive non-lift frames → warn
WRONG_EXERCISE_COOLDOWN = 60      # frames before next warning is allowed

# Fatigue
FATIGUE_WINDOW = 5                # last N reps to track fatigue trend

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE META
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STAGES       = ["SETUP_BAR_TOUCH", "CLEAN_TO_SHOULDER", "JERK_OVERHEAD", "RELEASE_FINISH"]
SCORED_STAGES = {"CLEAN_TO_SHOULDER", "JERK_OVERHEAD", "RELEASE_FINISH"}

STAGE_LABELS = {
    "SETUP_BAR_TOUCH":   "① Bar Touch / Setup",
    "CLEAN_TO_SHOULDER": "② Clean to Shoulder",
    "JERK_OVERHEAD":     "③ Jerk Overhead",
    "RELEASE_FINISH":    "④ Release / Finish",
}

STAGE_ORDER = {
    "SETUP_BAR_TOUCH":   0,
    "CLEAN_TO_SHOULDER": 1,
    "JERK_OVERHEAD":     2,
    "RELEASE_FINISH":    3,
}

PHASE_WEIGHTS = {
    "CLEAN_TO_SHOULDER": 0.35,
    "JERK_OVERHEAD":     0.40,
    "RELEASE_FINISH":    0.25,
}

JOINT_LANDMARKS = {
    "L.Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "R.Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "L.Elbow":    mp_pose.PoseLandmark.LEFT_ELBOW,
    "R.Elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW,
    "L.Wrist":    mp_pose.PoseLandmark.LEFT_WRIST,
    "R.Wrist":    mp_pose.PoseLandmark.RIGHT_WRIST,
    "L.Hip":      mp_pose.PoseLandmark.LEFT_HIP,
    "R.Hip":      mp_pose.PoseLandmark.RIGHT_HIP,
    "L.Knee":     mp_pose.PoseLandmark.LEFT_KNEE,
    "R.Knee":     mp_pose.PoseLandmark.RIGHT_KNEE,
    "L.Ankle":    mp_pose.PoseLandmark.LEFT_ANKLE,
    "R.Ankle":    mp_pose.PoseLandmark.RIGHT_ANKLE,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEOMETRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _angle(a, b, c) -> float:
    """Interior angle at point B (degrees), 2-D only."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba = a - b; bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _extract_angles(lm) -> Optional[Dict[str, Any]]:
    """Extract key joint angles from landmarks. Returns None if visibility too low."""
    P = mp_pose.PoseLandmark
    needed = [P.LEFT_HIP, P.LEFT_KNEE, P.LEFT_ANKLE,
              P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST]
    low = [p.name for p in needed if lm[p.value].visibility < 0.35]
    if low:
        return None

    def pt(p): return [lm[p.value].x, lm[p.value].y]

    hip   = pt(P.LEFT_HIP);    knee  = pt(P.LEFT_KNEE);   ankle = pt(P.LEFT_ANKLE)
    shldr = pt(P.LEFT_SHOULDER); elbow = pt(P.LEFT_ELBOW); wrist = pt(P.LEFT_WRIST)

    return {
        "knee_angle":           round(_angle(hip,   knee,  ankle), 1),
        "hip_angle":            round(_angle(shldr, hip,   knee),  1),
        "elbow_angle":          round(_angle(shldr, elbow, wrist), 1),
        "raise_angle":          round(_angle(hip,   shldr, wrist), 1),
        "wrist_above_shoulder": lm[P.LEFT_WRIST.value].y < lm[P.LEFT_SHOULDER.value].y,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BAR TOUCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _detect_bar_touch(lm) -> bool:
    """Wrists together + at/below knee level = athlete gripping bar on floor."""
    P  = mp_pose.PoseLandmark
    lw = lm[P.LEFT_WRIST.value]; rw = lm[P.RIGHT_WRIST.value]; lk = lm[P.LEFT_KNEE.value]
    if lw.visibility < 0.5 or rw.visibility < 0.5 or lk.visibility < 0.4:
        return False
    wrist_dist  = abs(lw.x - rw.x)
    avg_wrist_y = (lw.y + rw.y) / 2
    return avg_wrist_y >= lk.y - 0.10 and wrist_dist < 0.40


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE DETECTION  (heuristic-only, no external ML model needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _heuristic_stage(a: Dict) -> Optional[str]:
    """
    Priority order: RELEASE > JERK > CLEAN.
    Returns None for transition frames.
    """
    ea  = a["elbow_angle"]
    ra  = a["raise_angle"]
    ka  = a["knee_angle"]
    ha  = a["hip_angle"]
    wab = a["wrist_above_shoulder"]

    if ra >= RELEASE_RAISE_MIN and ea >= RELEASE_ELBOW_MIN and wab:
        return "RELEASE_FINISH"
    if ra >= JERK_RAISE_MIN and ea >= JERK_ELBOW_MIN and wab:
        return "JERK_OVERHEAD"
    if CLEAN_ELBOW_MIN <= ea <= CLEAN_ELBOW_MAX and ka >= CLEAN_KNEE_MIN and ha >= CLEAN_HIP_MIN:
        return "CLEAN_TO_SHOULDER"
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WRONG-EXERCISE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _classify_activity(lm) -> str:
    """Very rough classifier for non-lift postures."""
    P   = mp_pose.PoseLandmark
    def y(p): return lm[p.value].y
    def x(p): return lm[p.value].x

    wrist_avg_y = (y(P.LEFT_WRIST) + y(P.RIGHT_WRIST)) / 2
    hip_y       = y(P.LEFT_HIP)
    knee_y      = y(P.LEFT_KNEE)
    ankle_y     = y(P.LEFT_ANKLE)

    # Wrists well above head → overhead press / snatch attempt
    if wrist_avg_y < y(P.LEFT_SHOULDER) - 0.15:
        return "overhead_movement"

    # Hips near knee height and knees bent → squat-like
    hip_knee_diff = abs(hip_y - knee_y)
    if hip_knee_diff < 0.12 and knee_y < ankle_y:
        return "squat"

    # Torso roughly horizontal → deadlift setup / bent-over
    torso_angle = abs(y(P.LEFT_SHOULDER) - hip_y)
    if torso_angle < 0.05:
        return "bent_over"

    # Sitting or lying
    if hip_y > ankle_y - 0.05:
        return "sitting_or_lying"

    return "standing_idle"


def _wrong_exercise_msg(activity: str) -> str:
    return {
        "squat":              "Wrong exercise! Looks like a SQUAT — prepare for CLEAN & JERK.",
        "overhead_movement":  "Different overhead move detected — ensure correct CLEAN & JERK technique.",
        "bent_over":          "Bent-over position detected — please start from correct SETUP position.",
        "sitting_or_lying":   "Sitting/lying detected — stand up and get into CLEAN & JERK position.",
        "standing_idle":      "Just standing — please approach the bar and begin your lift.",
    }.get(activity, "Unexpected movement — please perform a CLEAN & JERK.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORM EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _evaluate_form(stage: str, angles: Dict, consec: Dict) -> List[str]:
    """Return active fault strings for this frame (fires after FAULT_HOLD frames)."""
    issues = []
    ea  = angles.get("elbow_angle",  180)
    ra  = angles.get("raise_angle",    0)
    ka  = angles.get("knee_angle",   180)
    ha  = angles.get("hip_angle",    180)
    wab = angles.get("wrist_above_shoulder", True)

    def check(key: str, bad: bool, msg: str):
        if bad:
            consec[key] = consec.get(key, 0) + 1
            if consec[key] >= FAULT_HOLD:
                issues.append(msg)
        else:
            consec[key] = 0

    if stage == "CLEAN_TO_SHOULDER":
        check("clean_hip",
              CLEAN_HIP_FAULT_LO < ha < CLEAN_HIP_FAULT_HI,
              f"Incomplete hip extension after catch ({ha:.0f}°) — drive hips to full stand")
        check("clean_knee",
              CLEAN_KNEE_FAULT_LO < ka < CLEAN_KNEE_FAULT_HI,
              f"Incomplete knee extension ({ka:.0f}°) — push to full stand")

    elif stage == "JERK_OVERHEAD":
        check("jerk_elbow",
              ea < JERK_ELBOW_FAULT,
              f"Elbow not locked overhead ({ea:.0f}° — need ≥{JERK_ELBOW_FAULT}°)")
        check("jerk_raise",
              ra < JERK_RAISE_FAULT,
              f"Bar not fully overhead ({ra:.0f}° — need ≥{JERK_RAISE_FAULT}°)")
        check("jerk_wrist",
              not wab,
              "Wrists below shoulder — bar must clear overhead")

    elif stage == "RELEASE_FINISH":
        check("release_elbow",
              ea < RELEASE_ELBOW_FAULT,
              f"Elbow unlocking during hold ({ea:.0f}° — need ≥{RELEASE_ELBOW_FAULT}°)")
        check("release_raise",
              ra < RELEASE_RAISE_FAULT,
              f"Bar dipping during hold ({ra:.0f}° — need ≥{RELEASE_RAISE_FAULT}°)")
        check("release_wrist",
              not wab,
              "Bar dropped below overhead during hold")

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORM SCORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_phase_form(stage: str, stage_counter: Dict,
                        stage_issues_log: Dict) -> float:
    total = stage_counter.get(stage, 0)
    if total == 0:
        return 0.0
    fault_frames = len(stage_issues_log.get(stage, []))
    if stage == "CLEAN_TO_SHOULDER":
        scored = max(total // 2, 1)
        fault_in_scored = min(fault_frames, scored)
        pct = (1 - fault_in_scored / scored) * 100
    else:
        pct = (1 - fault_frames / total) * 100
    return round(max(0.0, min(100.0, pct)), 2)


def _compute_overall_form(stage_counter: Dict, stage_issues_log: Dict) -> Dict[str, Any]:
    """Weighted overall form + per-phase breakdown."""
    phase_form = {
        st: _compute_phase_form(st, stage_counter, stage_issues_log)
        for st in SCORED_STAGES
    }
    detected = [st for st in PHASE_WEIGHTS if stage_counter.get(st, 0) > 0]
    if not detected:
        overall = 100.0
    else:
        weight_sum = sum(PHASE_WEIGHTS[st] for st in detected)
        overall    = sum(phase_form[st] * PHASE_WEIGHTS[st] for st in detected) / weight_sum
    return {
        "overall":             round(max(0.0, min(100.0, overall)), 2),
        "CLEAN_TO_SHOULDER":   phase_form.get("CLEAN_TO_SHOULDER", 0.0),
        "JERK_OVERHEAD":       phase_form.get("JERK_OVERHEAD",     0.0),
        "RELEASE_FINISH":      phase_form.get("RELEASE_FINISH",    0.0),
    }


def _form_grade(score: float) -> Tuple[str, Tuple[int, int, int]]:
    if score >= 88: return "EXCELLENT", (0, 220, 0)
    if score >= 72: return "GOOD",      (0, 200, 120)
    if score >= 55: return "FAIR",      (0, 165, 255)
    if score >= 38: return "POOR",      (0, 80, 255)
    return "BAD", (0, 0, 220)


def _machine_call(form: Dict, stage_counter: Dict,
                  stage_issues_log: Dict) -> Tuple[str, str]:
    """
    GREEN LIGHT    : overall ≥ 65% + all 3 scored stages detected + no phase < 50%
    MAJORITY GREEN : overall ≥ 50%  OR only 1 concern
    RED LIGHT      : otherwise
    """
    overall  = form["overall"]
    concerns = []

    for st, label in [("CLEAN_TO_SHOULDER", "Clean phase"),
                      ("JERK_OVERHEAD",     "Jerk phase"),
                      ("RELEASE_FINISH",    "Release phase")]:
        if stage_counter.get(st, 0) == 0:
            concerns.append(f"{label} not detected — incomplete lift")
        elif form.get(st, 100.0) < 50.0:
            flat  = [i for sub in stage_issues_log.get(st, []) for i in sub]
            worst = Counter(i.split(" (")[0] for i in flat).most_common(1)
            msg   = worst[0][0] if worst else "Sustained form fault"
            concerns.append(f"{label} form too low ({form[st]:.0f}%) — {msg}")

    n = len(concerns)
    if n == 0 and overall >= PASS_THRESHOLD:
        return (
            "GREEN LIGHT",
            f"Lift confirmed — {overall:.1f}% "
            f"(clean={form['CLEAN_TO_SHOULDER']:.0f}% "
            f"jerk={form['JERK_OVERHEAD']:.0f}% "
            f"release={form['RELEASE_FINISH']:.0f}%). All phases completed.",
        )
    if n <= 1 and overall >= MAJORITY_THRESHOLD:
        concern = f" Concern: {concerns[0]}" if concerns else ""
        return ("MAJORITY GREEN", f"Lift narrowly confirmed — {overall:.1f}%.{concern}")
    reason = (
        f"Lift not confirmed — {overall:.1f}%. Faults: " + " | ".join(concerns)
        if concerns else
        f"Lift not confirmed — {overall:.1f}% (below {MAJORITY_THRESHOLD}% threshold)."
    )
    return ("RED LIGHT", reason)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FATIGUE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fatigue_index(lift_log: List[Dict]) -> int:
    """0-100 fatigue estimate based on form trend across lifts."""
    if len(lift_log) < 2:
        return 0
    recent = lift_log[-FATIGUE_WINDOW:]
    scores = [r["overall_form"] for r in recent]
    trend  = max(0, scores[0] - scores[-1])         # form drop
    bad_rt = sum(1 for r in recent if r["overall_form"] < 50) / len(recent) * 100
    return int(np.clip(0.6 * trend + 0.4 * bad_rt, 0, 100))


def _fatigue_level(v: int) -> str:
    return "LOW" if v < 30 else ("MODERATE" if v < 60 else "HIGH")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DRAWING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BODY_CONNECTIONS = [c for c in mp_pose.POSE_CONNECTIONS if c[0] > 10 and c[1] > 10]


def _draw_skeleton(frame, landmarks):
    mp_draw.draw_landmarks(
        frame, landmarks, BODY_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 60),  thickness=2, circle_radius=3),
        mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2),
    )


def _draw_panel(frame, form: Dict, stage: Optional[str],
                issues: List[str], lift_count: int, machine_call: str,
                machine_reason: str, form_frozen: bool):
    """Left-side HUD panel."""
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (310, h), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    overall = form.get("overall", 0.0)
    grade, gc = _form_grade(overall)

    cv2.putText(frame, "CLEAN & JERK LIVE",
                (8, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 200, 255), 1)
    cv2.line(frame, (8, 34), (298, 34), (80, 80, 80), 1)

    # Lift count
    cv2.putText(frame, f"Lifts: {lift_count}",
                (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Current phase
    phase_txt = STAGE_LABELS.get(stage, "Waiting...") if stage else "Waiting for lift..."
    phase_col = (0, 255, 180) if stage else (160, 160, 160)
    cv2.putText(frame, phase_txt[:28],
                (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, phase_col, 1)
    cv2.line(frame, (8, 96), (298, 96), (80, 80, 80), 1)

    # Overall form bar
    bw = int(270 * min(overall, 100) / 100)
    cv2.rectangle(frame, (8, 108), (278, 126), (50, 50, 50), -1)
    cv2.rectangle(frame, (8, 108), (8 + bw, 126), gc, -1)
    label = f"{overall:.1f}%  {grade}" + ("  [FINAL]" if form_frozen else "")
    cv2.putText(frame, label, (8, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gc, 2)
    cv2.line(frame, (8, 153), (298, 153), (80, 80, 80), 1)

    # Per-phase mini bars
    phase_rows = [
        ("CLEAN_TO_SHOULDER", "Clean", (0, 255, 180)),
        ("JERK_OVERHEAD",     "Jerk",  (255, 180,   0)),
        ("RELEASE_FINISH",    "Rel.",  (255,  80, 200)),
    ]
    y = 164
    for st, lbl, col in phase_rows:
        pct = form.get(st, 0.0); mbw = int(120 * pct / 100)
        cv2.putText(frame, lbl, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        cv2.rectangle(frame, (72, y - 11), (192, y + 2), (50, 50, 50), -1)
        cv2.rectangle(frame, (72, y - 11), (72 + mbw, y + 2), col, -1)
        cv2.putText(frame, f"{pct:.0f}%",
                    (197, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        y += 22
    cv2.line(frame, (8, y), (298, y), (80, 80, 80), 1); y += 14

    # Form issues
    if issues:
        cv2.putText(frame, "FORM ISSUES:",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1); y += 18
        for iss in issues[:3]:
            cv2.putText(frame, f"  {iss[:36]}",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 140, 255), 1); y += 16
    else:
        cv2.putText(frame, "Form OK", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        y += 18

    # Machine call (shown once form is frozen)
    if form_frozen and machine_call:
        cv2.line(frame, (8, y + 2), (298, y + 2), (80, 80, 80), 1); y += 14
        col_map = {
            "GREEN LIGHT":    (0, 200, 0),
            "MAJORITY GREEN": (0, 180, 100),
            "RED LIGHT":      (50, 50, 220),
        }
        mc_col = col_map.get(machine_call, (200, 200, 200))
        cv2.putText(frame, machine_call,
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mc_col, 2); y += 20
        reason_short = machine_reason[:38] + "..." if len(machine_reason) > 38 else machine_reason
        cv2.putText(frame, reason_short,
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)


def _draw_angles(frame, lm, w, h, a: Dict):
    """Overlay joint angles near the relevant landmark."""
    P = mp_pose.PoseLandmark
    for enum, txt in [
        (P.LEFT_KNEE,     f"Knee {a.get('knee_angle',   0):.0f}°"),
        (P.LEFT_HIP,      f"Hip  {a.get('hip_angle',    0):.0f}°"),
        (P.LEFT_ELBOW,    f"Elbow {a.get('elbow_angle', 0):.0f}°"),
        (P.LEFT_SHOULDER, f"Raise {a.get('raise_angle', 0):.0f}°"),
    ]:
        lmk = lm[enum.value]
        if lmk.visibility < 0.4:
            continue
        cv2.putText(frame, txt,
                    (int(lmk.x * w) - 65, int(lmk.y * h) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1)


def _draw_timer(frame, elapsed: float, w: int):
    m, s = int(elapsed) // 60, elapsed % 60
    cv2.putText(frame, f"Lift: {m:01d}:{s:05.2f}",
                (w - 200, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 100), 2)


def _draw_fatigue(frame, fatigue: int, level: str, w: int):
    col = (0, 220, 0) if fatigue < 30 else ((0, 165, 255) if fatigue < 60 else (0, 50, 255))
    cv2.putText(frame, f"Fatigue: {fatigue}% ({level})",
                (w - 310, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)


def _draw_wrong_banner(frame, activity: str):
    h, w = frame.shape[:2]
    msg  = f"WRONG MOVEMENT: {activity.upper().replace('_', ' ')} — Prepare CLEAN & JERK"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(msg, font, 0.62, 2)
    cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 180), -1)
    cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 255), 3)
    cv2.putText(frame, msg, (max(8, (w - tw) // 2), 34), font, 0.62, (255, 255, 255), 2)


def _draw_machine_call_banner(frame, machine_call: str, machine_reason: str, w: int):
    """Prominent top-right machine call banner (shown after form is frozen)."""
    col_map = {
        "GREEN LIGHT":    ((0, 140, 0),   (255, 255, 255)),
        "MAJORITY GREEN": ((0, 120, 70),  (220, 255, 220)),
        "RED LIGHT":      ((30, 30, 200), (255, 180, 180)),
    }
    bg, fg = col_map.get(machine_call, ((80, 80, 80), (255, 255, 255)))
    bx, by, bw, bh = w - 400, 8, 390, 54
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), bg, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
    cv2.putText(frame, f"MACHINE: {machine_call}",
                (bx + 8, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2)
    short = machine_reason[:52] + "..." if len(machine_reason) > 52 else machine_reason
    cv2.putText(frame, short,
                (bx + 8, by + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.34, fg, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIVE SESSION CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LiveCleanJerkSession:
    """
    One instance = one live webcam session.
    Call start() → returns immediately, processing runs in background thread.
    Call get_frame() → JPEG bytes of the latest annotated frame.
    Call get_stats() → live stats dict.
    Call stop()  → finalises and returns a complete summary dict.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._lock        = threading.Lock()
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[bytes] = None

        # ── Public stats (written by background thread, read externally) ──────
        self.lift_count             = 0
        self.lift_log: List[Dict]   = []      # one entry per completed lift
        self.wrong_events: List[Dict] = []
        self.current_stage: Optional[str]  = None
        self.current_form: Dict[str, Any]  = {"overall": 100.0,
                                               "CLEAN_TO_SHOULDER": 100.0,
                                               "JERK_OVERHEAD": 100.0,
                                               "RELEASE_FINISH": 100.0}
        self.current_grade            = "EXCELLENT"
        self.current_issues: List[str] = []
        self.current_machine_call     = ""
        self.current_machine_reason   = ""
        self.form_frozen              = False
        self.current_fatigue          = 0
        self.current_fatigue_level    = "LOW"
        self.wrong_warning_active     = False
        self.wrong_warning_msg: Optional[str] = None
        self.session_start_time: Optional[float] = None

        print(f"  ℹ️   LiveCleanJerkSession created (camera={camera_index})")

    # ──────────────────────────────────────────────────────────────────────────
    def start(self) -> Dict:
        if self._running:
            print("  ⚠️   Session already running")
            return {"status": "already_running"}
        print(f"\n  ▶   Starting live C&J session (camera {self.camera_index})...")
        self._running           = True
        self.session_start_time = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("  ✅  Background thread started")
        return {"status": "started"}

    # ──────────────────────────────────────────────────────────────────────────
    def stop(self) -> Dict:
        if not self._running:
            return {"status": "not_running"}
        print("\n  ⏹   Stopping C&J session...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=6)
        print("  ✅  Session stopped")
        return self.get_stats(final=True)

    # ──────────────────────────────────────────────────────────────────────────
    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self, final: bool = False) -> Dict:
        elapsed = (time.time() - self.session_start_time
                   if self.session_start_time else 0)

        # Aggregate form across all lifts
        all_overall = [r["overall_form"] for r in self.lift_log]
        avg_overall  = round(float(np.mean(all_overall)), 2) if all_overall else 0.0
        avg_grade, _ = _form_grade(avg_overall)

        # Top issues across all lifts
        all_issues = [i for r in self.lift_log for i in r.get("top_issues", [])]
        top_issues = [{"issue": k, "occurrences": v}
                      for k, v in Counter(all_issues).most_common(5)]

        return {
            "status":                 "final" if final else "live",
            "lift_count":             self.lift_count,
            "session_duration_sec":   round(elapsed, 1),
            "current_stage":          self.current_stage,
            "current_form":           self.current_form,
            "current_form_grade":     self.current_grade,
            "current_form_issues":    self.current_issues,
            "form_frozen":            self.form_frozen,
            "machine_call":           self.current_machine_call,
            "machine_reason":         self.current_machine_reason,
            "average_form_score":     avg_overall,
            "average_form_grade":     avg_grade,
            "top_form_issues":        top_issues,
            "fatigue":                self.current_fatigue,
            "fatigue_level":          self.current_fatigue_level,
            "wrong_exercise": {
                "active":          self.wrong_warning_active,
                "warning_message": self.wrong_warning_msg,
                "total_warnings":  len(self.wrong_events),
                "events":          self.wrong_events,
            },
            "lifts":      self.lift_log,
            "stream_url": "http://localhost:8000/cleanjerk/live/stream",
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _run_loop(self):
        print(f"  🎥  Opening camera {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"  ❌  Cannot open camera {self.camera_index}")
            self._running = False
            return

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"  ✅  Camera opened — {w}×{h} @ {fps:.0f}fps")

        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Per-lift state (reset on each new lift attempt) ───────────────────
        def _reset_lift_state():
            return {
                "stage_counter":     {},
                "stage_issues_log":  {},
                "stage_window":      deque(maxlen=STAGE_WINDOW),
                "current_stage":     None,
                "highest_reached":   -1,
                "consecutive_bad":   {},
                "lift_started":      False,
                "touch_window":      deque(maxlen=5),
                "lift_start_frame":  None,
                "form_frozen":       False,
                "frozen_form":       {},
                "frozen_mc":         ("", ""),
                "running_phase_form": {
                    "CLEAN_TO_SHOULDER": 100.0,
                    "JERK_OVERHEAD":     100.0,
                    "RELEASE_FINISH":    100.0,
                },
            }

        ls = _reset_lift_state()   # lift state

        # ── Session-level tracking ────────────────────────────────────────────
        frame_count      = 0
        wrong_frames     = 0
        wrong_cd         = 0
        last_wrong_type  = "unknown"

        # Track inter-lift idle to detect when athlete steps back (end of lift)
        post_release_idle = 0
        IDLE_RESET_FRAMES = 90    # ~3s idle after release → reset for next lift

        print("  🔄  Processing loop running...")

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05); continue

            frame_count += 1
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            current_issues: List[str] = []
            angles: Dict              = {}

            if res.pose_landmarks:
                lm   = res.pose_landmarks.landmark
                flat_list = []
                for lmk in lm:
                    flat_list.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])

                # ── Wrong-exercise / idle detection (before lift starts) ───────
                if not ls["lift_started"]:
                    activity = _classify_activity(lm)
                    is_approach = activity in ("bent_over",) or _detect_bar_touch(lm)

                    if not is_approach and activity not in ("overhead_movement",):
                        wrong_frames += 1
                        last_wrong_type = activity
                        if wrong_frames >= WRONG_EXERCISE_FRAMES and wrong_cd <= 0:
                            msg = _wrong_exercise_msg(activity)
                            self.wrong_events.append({
                                "frame":    frame_count,
                                "activity": activity,
                                "message":  msg,
                                "time_sec": round(time.time() - self.session_start_time, 1),
                            })
                            wrong_cd                  = WRONG_EXERCISE_COOLDOWN
                            self.wrong_warning_active = True
                            self.wrong_warning_msg    = msg
                            print(f"  ⚠️   WRONG MOVEMENT @ frame {frame_count} — {activity}")
                    else:
                        wrong_frames = max(0, wrong_frames - 2)
                        if wrong_frames == 0:
                            self.wrong_warning_active = False
                    if wrong_cd > 0:
                        wrong_cd -= 1

                # ── Bar touch detection → start lift ──────────────────────────
                if not ls["lift_started"]:
                    ls["touch_window"].append(_detect_bar_touch(lm))
                    if sum(ls["touch_window"]) >= BAR_TOUCH_CONFIRM:
                        ls["lift_started"]     = True
                        ls["lift_start_frame"] = frame_count
                        ls["stage_counter"]["SETUP_BAR_TOUCH"] = 0
                        self.wrong_warning_active = False
                        print(f"  🏋️   Bar touch confirmed @ frame {frame_count} — lift #{self.lift_count + 1} starting")

                # ── Stage detection & form scoring ────────────────────────────
                if ls["lift_started"]:
                    angles = _extract_angles(lm) or {}

                    if angles and not ls["form_frozen"]:
                        detected = _heuristic_stage(angles)
                        if detected:
                            ls["stage_window"].append(detected)
                            smoothed = max(set(ls["stage_window"]),
                                          key=list(ls["stage_window"]).count)
                            order = STAGE_ORDER.get(smoothed, -1)
                            if order >= ls["highest_reached"]:
                                if smoothed != ls["current_stage"]:
                                    ls["consecutive_bad"].clear()
                                    print(f"  📍  Phase → {smoothed} @ frame {frame_count}")
                                ls["highest_reached"] = order
                                ls["current_stage"]   = smoothed

                        if ls["current_stage"]:
                            st = ls["current_stage"]
                            ls["stage_counter"][st] = ls["stage_counter"].get(st, 0) + 1
                            current_issues = _evaluate_form(
                                st, angles, ls["consecutive_bad"])
                            if current_issues:
                                ls["stage_issues_log"].setdefault(st, []).append(current_issues)

                            # Update live phase form
                            if st in ls["running_phase_form"]:
                                ls["running_phase_form"][st] = _compute_phase_form(
                                    st, ls["stage_counter"], ls["stage_issues_log"])

                            # Freeze on first RELEASE_FINISH frame
                            if st == "RELEASE_FINISH" and not ls["form_frozen"]:
                                ls["form_frozen"]  = True
                                ls["frozen_form"]  = _compute_overall_form(
                                    ls["stage_counter"], ls["stage_issues_log"])
                                ls["frozen_mc"]    = _machine_call(
                                    ls["frozen_form"],
                                    ls["stage_counter"],
                                    ls["stage_issues_log"])
                                print(f"  🔒  Form FROZEN @ frame {frame_count}  "
                                      f"overall={ls['frozen_form']['overall']}%  "
                                      f"call={ls['frozen_mc'][0]}")

                    # ── Post-release idle → commit lift + reset ───────────────
                    if ls["form_frozen"]:
                        post_release_idle += 1
                        if post_release_idle >= IDLE_RESET_FRAMES:
                            self._commit_lift(ls, frame_count)
                            ls = _reset_lift_state()
                            post_release_idle = 0
                    else:
                        post_release_idle = 0

                # ── Push stats to public attrs ────────────────────────────────
                display_form = (ls["frozen_form"] if ls["form_frozen"]
                                else _compute_overall_form(
                                    ls["stage_counter"], ls["stage_issues_log"]))
                grade, gc    = _form_grade(display_form.get("overall", 100.0))
                mc, mr       = ls["frozen_mc"] if ls["form_frozen"] else ("", "")
                fatigue      = _fatigue_index(self.lift_log)

                self.current_stage          = ls["current_stage"]
                self.current_form           = display_form
                self.current_grade          = grade
                self.current_issues         = current_issues
                self.form_frozen            = ls["form_frozen"]
                self.current_machine_call   = mc
                self.current_machine_reason = mr
                self.current_fatigue        = fatigue
                self.current_fatigue_level  = _fatigue_level(fatigue)

                # ── Draw ─────────────────────────────────────────────────────
                _draw_skeleton(frame, res.pose_landmarks)
                if angles:
                    _draw_angles(frame, lm, w, h, angles)
                _draw_panel(frame, display_form, ls["current_stage"],
                            current_issues, self.lift_count, mc, mr,
                            ls["form_frozen"])
                if ls["lift_started"]:
                    elapsed_lift = (frame_count - ls["lift_start_frame"]) / fps
                    _draw_timer(frame, elapsed_lift, w)
                _draw_fatigue(frame, fatigue, _fatigue_level(fatigue), w)
                if self.wrong_warning_active and wrong_cd > WRONG_EXERCISE_COOLDOWN - 60:
                    _draw_wrong_banner(frame, last_wrong_type)
                if ls["form_frozen"] and mc:
                    _draw_machine_call_banner(frame, mc, mr, w)

            # Session timer top-right
            elapsed_sess = time.time() - self.session_start_time
            m, s = int(elapsed_sess) // 60, int(elapsed_sess) % 60
            cv2.putText(frame, f"Session: {m:02d}:{s:02d}",
                        (w - 200, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._latest_frame = jpeg.tobytes()

        # Commit any in-progress lift before exit
        if ls["lift_started"] and ls["stage_counter"]:
            self._commit_lift(ls, frame_count, partial=True)

        cap.release()
        pose.close()
        print(f"  ✅  Camera released | frames={frame_count} | "
              f"lifts={self.lift_count} | wrong_events={len(self.wrong_events)}")

    # ──────────────────────────────────────────────────────────────────────────
    def _commit_lift(self, ls: Dict, frame_count: int, partial: bool = False):
        """Finalise current lift and append to lift_log."""
        self.lift_count += 1
        form = (ls["frozen_form"] if ls["form_frozen"]
                else _compute_overall_form(ls["stage_counter"], ls["stage_issues_log"]))
        mc, mr = (ls["frozen_mc"] if ls["form_frozen"]
                  else _machine_call(form, ls["stage_counter"], ls["stage_issues_log"]))

        # Aggregate top issues across all phases
        all_issues_flat = [i for sub in ls["stage_issues_log"].values() for lst in sub for i in lst]
        top_issues = [m for m, _ in Counter(
            i.split(" (")[0] for i in all_issues_flat).most_common(5)]

        entry = {
            "lift_number":    self.lift_count,
            "overall_form":   form["overall"],
            "form_grade":     _form_grade(form["overall"])[0],
            "machine_call":   mc,
            "machine_reason": mr,
            "phase_form": {
                "clean_to_shoulder": form.get("CLEAN_TO_SHOULDER", 0.0),
                "jerk_overhead":     form.get("JERK_OVERHEAD",     0.0),
                "release_finish":    form.get("RELEASE_FINISH",    0.0),
            },
            "phases_detected": {
                st: ls["stage_counter"].get(st, 0) > 0 for st in STAGES
            },
            "phase_frame_counts": dict(ls["stage_counter"]),
            "top_issues":     top_issues,
            "form_frozen":    ls["form_frozen"],
            "partial":        partial,
            "end_frame":      frame_count,
        }
        self.lift_log.append(entry)
        print(f"  ✅  Lift #{self.lift_count} committed — "
              f"overall={form['overall']}%  call={mc}  partial={partial}")