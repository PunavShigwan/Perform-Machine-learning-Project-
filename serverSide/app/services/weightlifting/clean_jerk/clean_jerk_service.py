"""
clean_jerk_service.py
======================
Best video specs:
  - FPS        : 30 fps minimum | 60 fps ideal
  - Resolution : 720p or 1080p
  - Camera     : SIDE-ON (90° to athlete) — front/back breaks pose estimation
  - Framing    : Full body head-to-feet visible throughout
  - Lighting   : Bright, even — no strong backlight

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① CLEAN_TO_SHOULDER
    Bar leaves floor → racks on shoulders → athlete stands upright
    Form measured: quality of the STAND-UP (hip & knee extension)
    NOT penalised: deep squat catch (correct technique)

② JERK_OVERHEAD
    Bar leaves shoulders → fully pressed / caught overhead
    Form measured: elbow lockout + raise angle at the TOP position
    NOT penalised: the jerk dip (brief knee bend before drive)

③ RELEASE_FINISH
    Bar held overhead while judges signal
    Form measured: maintaining lockout throughout the hold
    NOTE: Once RELEASE_FINISH is detected, form % is FROZEN — no further
          updates to scoring after the lift is complete.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANGLE REFERENCE  (measured AT the named joint, 2-D x/y)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  knee_angle   hip–knee–ankle          full extension ≈ 160-180  | squat ≈ 50-90
  hip_angle    shoulder–hip–knee       standing       ≈ 160-180  | squat ≈ 50-90
  elbow_angle  shoulder–elbow–wrist    straight arm   ≈ 160-180  | rack  ≈ 50-90
  raise_angle  hip–shoulder–wrist      arm at side    ≈  10-30   | overhead ≈ 150-175

PASS THRESHOLD  : 65%  (each phase + overall)
MACHINE CALL    : GREEN LIGHT ≥ 65% | MAJORITY GREEN 50-64% | RED LIGHT < 50%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import os
import logging
import traceback
import numpy as np
import mediapipe as mp
import joblib
from collections import deque, Counter
from typing import Dict, List, Optional, Any, Tuple

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | clean_jerk | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_jerk_service")


def _ckpt(step: str, detail: str = ""):
    logger.info(f"[CHECKPOINT] {step}" + (f" | {detail}" if detail else ""))


def _make_error(context: str, exc: Exception) -> dict:
    tb = traceback.format_exc()
    logger.error(f"[ERROR] {context} → {exc}\n{tb}")
    return {"error": True, "context": context, "message": str(exc), "traceback": tb}


# ─── MediaPipe ────────────────────────────────────────────────────────────────
try:
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    _ckpt("MediaPipe import OK")
except Exception as _e:
    logger.critical(f"[FATAL] MediaPipe import failed: {_e}")
    raise

# ─── ML Models ────────────────────────────────────────────────────────────────
MODEL_PATH     = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models\best_model.pkl"
LABEL_MAP_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models\label_map.pkl"

_ml_model  = None
_label_map = None


def load_ml_models() -> str:
    global _ml_model, _label_map
    if _ml_model is not None:
        return "already_loaded"
    for tag, path in [("model", MODEL_PATH), ("label_map", LABEL_MAP_PATH)]:
        if not os.path.exists(path):
            logger.warning(f"[ML] {tag} not found: {path} — heuristic-only mode")
            return "missing_files"
    try:
        _ml_model  = joblib.load(MODEL_PATH)
        _label_map = joblib.load(LABEL_MAP_PATH)
        _ckpt("ML models loaded", f"type={type(_ml_model).__name__}")
        return "loaded"
    except Exception as e:
        logger.error(f"[ML] Load failed: {e}\n{traceback.format_exc()}")
        _ml_model = _label_map = None
        return f"load_error: {e}"


def ml_predict(flat: list) -> Optional[str]:
    if _ml_model is None:
        return None
    try:
        idx   = _ml_model.predict([flat])[0]
        rev   = {v: k for k, v in _label_map.items()}
        label = rev.get(idx)
        logger.debug(f"[ML] idx={idx} → {label}")
        return label
    except Exception as e:
        logger.warning(f"[ML] predict failed: {e}")
        return None


def landmarks_to_flat(lm) -> list:
    out = []
    for lmk in lm:
        out.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])
    return out


# ─── Stage config ─────────────────────────────────────────────────────────────
STAGES = ["SETUP_BAR_TOUCH", "CLEAN_TO_SHOULDER", "JERK_OVERHEAD", "RELEASE_FINISH"]

STAGE_LABELS = {
    "SETUP_BAR_TOUCH":   "① Bar Touch / Setup",
    "CLEAN_TO_SHOULDER": "② Clean to Shoulder",
    "JERK_OVERHEAD":     "③ Jerk Overhead",
    "RELEASE_FINISH":    "④ Release / Finish",
}

SUCCESS_STAGES = {"CLEAN_TO_SHOULDER", "JERK_OVERHEAD", "RELEASE_FINISH"}

STAGE_ORDER = {
    "SETUP_BAR_TOUCH":   0,
    "CLEAN_TO_SHOULDER": 1,
    "JERK_OVERHEAD":     2,
    "RELEASE_FINISH":    3,
}

ML_LABEL_MAP = {
    "clean":              "CLEAN_TO_SHOULDER",
    "clean_shoulder":     "CLEAN_TO_SHOULDER",
    "clean_to_shoulder":  "CLEAN_TO_SHOULDER",
    "jerk":               "JERK_OVERHEAD",
    "jerk_overhead":      "JERK_OVERHEAD",
    "overhead":           "JERK_OVERHEAD",
    "release":            "RELEASE_FINISH",
    "release_finish":     "RELEASE_FINISH",
    "finish":             "RELEASE_FINISH",
    "lockout":            "RELEASE_FINISH",
    "setup":              "SETUP_BAR_TOUCH",
    "bar_touch":          "SETUP_BAR_TOUCH",
}

BODY_CONNECTIONS = [c for c in mp_pose.POSE_CONNECTIONS if c[0] > 10 and c[1] > 10]

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
# TUNABLE THRESHOLDS  (all in one place)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Stage detection gates ─────────────────────────────────────
CLEAN_ELBOW_MIN   = 40
CLEAN_ELBOW_MAX   = 105
CLEAN_KNEE_MIN    = 55
CLEAN_HIP_MIN     = 30

JERK_RAISE_MIN    = 135
JERK_ELBOW_MIN    = 135
RELEASE_RAISE_MIN = 148
RELEASE_ELBOW_MIN = 150

# ── Form fault gates ──────────────────────────────────────────
CLEAN_HIP_FAULT_LO   = 70
CLEAN_HIP_FAULT_HI   = 148
CLEAN_KNEE_FAULT_LO  = 70
CLEAN_KNEE_FAULT_HI  = 145

JERK_ELBOW_FAULT     = 150
JERK_RAISE_FAULT     = 135

RELEASE_ELBOW_FAULT  = 148
RELEASE_RAISE_FAULT  = 145

# ── Fault hold: frames before a fault counts ─────────────────
FAULT_HOLD = 12

# ── Pass / machine call thresholds ───────────────────────────
PASS_THRESHOLD       = 65.0
MAJORITY_THRESHOLD   = 50.0
JUDGE_FAULT_PCT      = 35.0


# ─── Geometry ─────────────────────────────────────────────────────────────────

def _angle(a, b, c) -> float:
    """Interior angle at point B (degrees), 2-D only."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba = a - b;  bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def extract_angles(lm) -> Optional[Dict[str, Any]]:
    P = mp_pose.PoseLandmark
    needed = [P.LEFT_HIP, P.LEFT_KNEE, P.LEFT_ANKLE,
              P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST]

    low = [p.name for p in needed if lm[p.value].visibility < 0.4]
    if low:
        logger.debug(f"[angles] skipped — low-vis: {low}")
        return None

    def pt(p): return [lm[p.value].x, lm[p.value].y]

    hip   = pt(P.LEFT_HIP);    knee  = pt(P.LEFT_KNEE);   ankle = pt(P.LEFT_ANKLE)
    shldr = pt(P.LEFT_SHOULDER); elbow = pt(P.LEFT_ELBOW); wrist = pt(P.LEFT_WRIST)

    a = {
        "knee_angle":           round(_angle(hip,   knee,  ankle), 1),
        "hip_angle":            round(_angle(shldr, hip,   knee),  1),
        "elbow_angle":          round(_angle(shldr, elbow, wrist), 1),
        "raise_angle":          round(_angle(hip,   shldr, wrist), 1),
        "wrist_above_shoulder": lm[P.LEFT_WRIST.value].y < lm[P.LEFT_SHOULDER.value].y,
    }
    logger.debug(
        f"[angles] knee={a['knee_angle']} hip={a['hip_angle']} "
        f"elbow={a['elbow_angle']} raise={a['raise_angle']} wab={a['wrist_above_shoulder']}"
    )
    return a


# ─── Bar touch ────────────────────────────────────────────────────────────────

def detect_bar_touch(lm) -> bool:
    """Wrists together + at/below knee = athlete gripping bar on the floor."""
    P  = mp_pose.PoseLandmark
    lw = lm[P.LEFT_WRIST.value];  rw = lm[P.RIGHT_WRIST.value]
    lk = lm[P.LEFT_KNEE.value]
    if lw.visibility < 0.5 or rw.visibility < 0.5 or lk.visibility < 0.4:
        return False
    wrist_dist  = abs(lw.x - rw.x)
    avg_wrist_y = (lw.y + rw.y) / 2
    result = avg_wrist_y >= lk.y - 0.10 and wrist_dist < 0.40
    logger.debug(
        f"[bar_touch] dist={wrist_dist:.3f} avg_y={avg_wrist_y:.3f} "
        f"knee_y={lk.y:.3f} → {result}"
    )
    return result


# ─── Stage detection ──────────────────────────────────────────────────────────

def heuristic_stage(a: Dict) -> Optional[str]:
    """
    Priority: RELEASE > JERK > CLEAN
    Each stage requires ALL conditions to be true simultaneously.
    Transition frames (arm mid-pull, dip) return None and are safely ignored.
    """
    ea  = a["elbow_angle"]
    ra  = a["raise_angle"]
    ka  = a["knee_angle"]
    ha  = a["hip_angle"]
    wab = a["wrist_above_shoulder"]

    if ra >= RELEASE_RAISE_MIN and ea >= RELEASE_ELBOW_MIN and wab:
        logger.debug(f"[heuristic] RELEASE_FINISH  raise={ra} elbow={ea}")
        return "RELEASE_FINISH"

    if ra >= JERK_RAISE_MIN and ea >= JERK_ELBOW_MIN and wab:
        logger.debug(f"[heuristic] JERK_OVERHEAD  raise={ra} elbow={ea}")
        return "JERK_OVERHEAD"

    if CLEAN_ELBOW_MIN <= ea <= CLEAN_ELBOW_MAX and ka >= CLEAN_KNEE_MIN and ha >= CLEAN_HIP_MIN:
        logger.debug(f"[heuristic] CLEAN_TO_SHOULDER  elbow={ea} knee={ka} hip={ha}")
        return "CLEAN_TO_SHOULDER"

    logger.debug(f"[heuristic] None  raise={ra} elbow={ea} knee={ka} hip={ha} wab={wab}")
    return None


def detect_stage(lm, flat: list) -> Tuple[Optional[str], Dict]:
    """Fuse ML + heuristic. Heuristic wins on conflict."""
    angles = extract_angles(lm)
    if angles is None:
        return None, {}

    h_stage = heuristic_stage(angles)
    ml_raw  = ml_predict(flat)
    m_stage = ML_LABEL_MAP.get(ml_raw.lower().replace(" ", "_"), None) if ml_raw else None

    logger.debug(f"[detect_stage] heuristic={h_stage}  ml={ml_raw}→{m_stage}")

    if m_stage and m_stage == h_stage: return m_stage, angles
    if m_stage and not h_stage:        return m_stage, angles
    if h_stage:                         return h_stage, angles
    return None, angles


# ─── Per-phase form scoring ────────────────────────────────────────────────────

def evaluate_form(
    stage: str,
    angles: Dict,
    consec: Dict,
) -> List[str]:
    """
    Returns list of active fault strings for this frame.
    Only fires a fault after FAULT_HOLD consecutive bad frames.
    """
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
                logger.debug(f"[fault] {key} fired @ {consec[key]} frames")
        else:
            consec[key] = 0

    if stage == "CLEAN_TO_SHOULDER":
        check(
            "clean_hip_stand",
            CLEAN_HIP_FAULT_LO < ha < CLEAN_HIP_FAULT_HI,
            f"Incomplete hip extension after catch ({ha:.0f}°) — drive hips to full stand",
        )
        check(
            "clean_knee_stand",
            CLEAN_KNEE_FAULT_LO < ka < CLEAN_KNEE_FAULT_HI,
            f"Incomplete knee extension after catch ({ka:.0f}°) — push to full stand",
        )

    elif stage == "JERK_OVERHEAD":
        check(
            "jerk_elbow_lock",
            ea < JERK_ELBOW_FAULT,
            f"Elbow not locked overhead ({ea:.0f}° — need ≥{JERK_ELBOW_FAULT}°)",
        )
        check(
            "jerk_raise",
            ra < JERK_RAISE_FAULT,
            f"Bar not fully overhead ({ra:.0f}° raise — need ≥{JERK_RAISE_FAULT}°)",
        )
        check(
            "jerk_wrist_over",
            not wab,
            "Wrists below shoulder — bar must clear overhead",
        )

    elif stage == "RELEASE_FINISH":
        check(
            "release_elbow",
            ea < RELEASE_ELBOW_FAULT,
            f"Elbow unlocking during hold ({ea:.0f}° — need ≥{RELEASE_ELBOW_FAULT}°)",
        )
        check(
            "release_raise",
            ra < RELEASE_RAISE_FAULT,
            f"Bar dipping during hold ({ra:.0f}° — need ≥{RELEASE_RAISE_FAULT}°)",
        )
        check(
            "release_wrist",
            not wab,
            "Bar dropped below overhead during hold",
        )

    return issues


# ─── Phase-accurate form percentage calculation ────────────────────────────────

def compute_phase_form(
    stage: str,
    stage_counter: Dict,
    stage_issues_log: Dict,
) -> float:
    total  = stage_counter.get(stage, 0)
    if total == 0:
        return 0.0

    fault_frames = len(stage_issues_log.get(stage, []))

    if stage == "CLEAN_TO_SHOULDER":
        scored_frames = max(total // 2, 1)
        fault_in_scored = min(fault_frames, scored_frames)
        pct = (1 - fault_in_scored / scored_frames) * 100
    else:
        pct = (1 - fault_frames / total) * 100

    return round(max(0.0, min(100.0, pct)), 2)


def compute_overall_form(
    stage_counter: Dict,
    stage_issues_log: Dict,
) -> Dict[str, Any]:
    """
    Returns dict with per-phase form % and weighted overall %.

    Weights:
      CLEAN_TO_SHOULDER : 35%
      JERK_OVERHEAD     : 40%
      RELEASE_FINISH    : 25%
    """
    WEIGHTS = {
        "CLEAN_TO_SHOULDER": 0.35,
        "JERK_OVERHEAD":     0.40,
        "RELEASE_FINISH":    0.25,
    }

    phase_form = {}
    for st in ["CLEAN_TO_SHOULDER", "JERK_OVERHEAD", "RELEASE_FINISH"]:
        phase_form[st] = compute_phase_form(st, stage_counter, stage_issues_log)

    detected = [st for st in WEIGHTS if stage_counter.get(st, 0) > 0]
    if not detected:
        overall = 100.0
    else:
        weight_sum = sum(WEIGHTS[st] for st in detected)
        overall    = sum(phase_form[st] * WEIGHTS[st] for st in detected) / weight_sum

    overall = round(max(0.0, min(100.0, overall)), 2)

    logger.info(
        f"[form] clean={phase_form.get('CLEAN_TO_SHOULDER')}%  "
        f"jerk={phase_form.get('JERK_OVERHEAD')}%  "
        f"release={phase_form.get('RELEASE_FINISH')}%  "
        f"overall={overall}%"
    )

    return {
        "overall":             overall,
        "CLEAN_TO_SHOULDER":   phase_form.get("CLEAN_TO_SHOULDER", 0.0),
        "JERK_OVERHEAD":       phase_form.get("JERK_OVERHEAD",     0.0),
        "RELEASE_FINISH":      phase_form.get("RELEASE_FINISH",    0.0),
    }


# ─── Machine call (replaces judge call) ───────────────────────────────────────

def compute_machine_call(
    form: Dict[str, Any],
    stage_counter: Dict,
    stage_issues_log: Dict,
) -> Tuple[str, str]:
    """
    Automated machine call — replaces the IWF 3-judge panel simulation.

    GREEN LIGHT    : overall form ≥ 65%  AND all 3 stages detected, no major faults
    MAJORITY GREEN : overall form ≥ 50%  OR only 1 concern
    RED LIGHT      : overall form < 50%  OR 2+ concerns

    Concerns are raised for:
      1. A mandatory stage completely missing
      2. A phase form % below 50% (that phase clearly failed)
    """
    overall  = form["overall"]
    concerns = []

    # ── Mandatory stage completion checks ─────────────────────────────────────
    if stage_counter.get("CLEAN_TO_SHOULDER", 0) == 0:
        concerns.append("Clean phase not detected — bar never reached shoulder rack")

    if stage_counter.get("JERK_OVERHEAD", 0) == 0:
        concerns.append("Jerk phase not completed — bar never reached overhead lockout")

    if stage_counter.get("RELEASE_FINISH", 0) == 0:
        concerns.append("Release not detected — lift not held to completion")

    # ── Phase-level form quality checks ───────────────────────────────────────
    for st, label in [
        ("CLEAN_TO_SHOULDER", "Clean phase"),
        ("JERK_OVERHEAD",     "Jerk phase"),
        ("RELEASE_FINISH",    "Release phase"),
    ]:
        phase_pct = form.get(st, 100.0)
        if stage_counter.get(st, 0) > 0 and phase_pct < 50.0:
            flat  = [i for sub in stage_issues_log.get(st, []) for i in sub]
            worst = Counter(i.split(" (")[0] for i in flat).most_common(1)
            msg   = worst[0][0] if worst else "Sustained form fault"
            concerns.append(f"{label} form too low ({phase_pct:.0f}%) — {msg}")

    _ckpt("Machine call", f"overall={overall}% n_concerns={len(concerns)} | {concerns}")

    n = len(concerns)

    if n == 0 and overall >= PASS_THRESHOLD:
        return (
            "GREEN LIGHT",
            f"Lift confirmed — overall form {overall:.1f}% "
            f"(clean={form['CLEAN_TO_SHOULDER']:.0f}% "
            f"jerk={form['JERK_OVERHEAD']:.0f}% "
            f"release={form['RELEASE_FINISH']:.0f}%). "
            "All phases completed with good form.",
        )

    if n <= 1 and overall >= MAJORITY_THRESHOLD:
        concern = f" Concern: {concerns[0]}" if concerns else ""
        return (
            "MAJORITY GREEN",
            f"Lift narrowly confirmed — overall form {overall:.1f}%.{concern}",
        )

    reason = (
        f"Lift not confirmed — overall form {overall:.1f}%. "
        "Faults detected: " + " | ".join(concerns)
    ) if concerns else (
        f"Lift not confirmed — overall form {overall:.1f}% (below {MAJORITY_THRESHOLD}% threshold)."
    )
    return ("RED LIGHT", reason)


# ─── Draw helpers ─────────────────────────────────────────────────────────────

def _draw_skeleton(frame, landmarks):
    mp_draw.draw_landmarks(
        frame, landmarks, BODY_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 60),  thickness=2, circle_radius=3),
        mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2),
    )


def _draw_joints(frame, lm, w, h):
    for name, enum in JOINT_LANDMARKS.items():
        lmk = lm[enum.value]
        if lmk.visibility < 0.4:
            continue
        cx, cy = int(lmk.x * w), int(lmk.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, name, (cx + 5, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)


def _draw_angles(frame, lm, w, h, a: Dict):
    P = mp_pose.PoseLandmark
    for enum, txt in [
        (P.LEFT_KNEE,     f"Knee {a.get('knee_angle', 0):.0f}°"),
        (P.LEFT_HIP,      f"Hip  {a.get('hip_angle',  0):.0f}°"),
        (P.LEFT_ELBOW,    f"Elbow {a.get('elbow_angle',0):.0f}°"),
        (P.LEFT_SHOULDER, f"Raise {a.get('raise_angle',0):.0f}°"),
    ]:
        lmk = lm[enum.value]
        if lmk.visibility < 0.4:
            continue
        cv2.putText(frame, txt,
                    (int(lmk.x * w) - 65, int(lmk.y * h) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1)


def _draw_gauge(frame, overall: float, phase_form: Dict, w: int, h: int, active: bool,
                frozen: bool = False):
    """
    Bottom bar: overall form + per-phase breakdown + machine call verdict.
    Pass threshold line drawn at 65%.
    When frozen=True (lift complete), shows a 'FINAL' badge and freezes the bar.
    """
    bx, by, bw, bh = 30, h - 90, 300, 26

    if not active:
        cv2.putText(frame, "Waiting for bar touch ...",
                    (bx, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)
        return

    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    fill = int(min(overall, 100) / 100.0 * bw)

    if overall >= PASS_THRESHOLD:
        color, vtext, vcol = (0, 210, 0),   "GREEN LIGHT",    (100, 255, 100)
    elif overall >= MAJORITY_THRESHOLD:
        color, vtext, vcol = (0, 180, 100), "MAJORITY GREEN", (120, 255, 180)
    else:
        color, vtext, vcol = (40, 40, 220), "RED LIGHT",      (100, 100, 255)

    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)

    form_label = f"Form: {overall:.1f}%"
    if frozen:
        form_label += " [FINAL]"
    cv2.putText(frame, form_label,
                (bx + 8, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(frame, vtext, (bx + bw + 12, by + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, vcol, 2)

    # Pass line at 65%
    tx = bx + int(PASS_THRESHOLD / 100.0 * bw)
    cv2.line(frame, (tx, by - 4), (tx, by + bh + 4), (255, 255, 0), 2)
    cv2.putText(frame, f"{int(PASS_THRESHOLD)}%", (tx - 14, by - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 0), 1)

    # Per-phase mini bars
    mini_labels = [
        ("CLEAN_TO_SHOULDER", "CL", (0, 255, 180)),
        ("JERK_OVERHEAD",     "JK", (255, 180,   0)),
        ("RELEASE_FINISH",    "RL", (255,  80, 200)),
    ]
    for i, (st, tag, col) in enumerate(mini_labels):
        pct  = phase_form.get(st, 0.0)
        mx   = bx + i * (bw // 3 + 2)
        mbw  = bw // 3 - 2
        mby  = by + bh + 6
        mbh  = 10
        cv2.rectangle(frame, (mx, mby), (mx + mbw, mby + mbh), (40, 40, 40), -1)
        cv2.rectangle(frame, (mx, mby), (mx + int(pct / 100 * mbw), mby + mbh), col, -1)
        cv2.rectangle(frame, (mx, mby), (mx + mbw, mby + mbh), (130, 130, 130), 1)
        cv2.putText(frame, f"{tag} {pct:.0f}%",
                    (mx + 2, mby + mbh - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)


def _draw_phases(frame, stage_counter: Dict, fps: int, w: int):
    total = sum(stage_counter.values()) or 1
    x0, y0 = w - 375, 50
    cols = {
        "SETUP_BAR_TOUCH":   (180, 255, 180),
        "CLEAN_TO_SHOULDER": (0,   255, 180),
        "JERK_OVERHEAD":     (255, 180,   0),
        "RELEASE_FINISH":    (255,  80, 200),
    }
    cv2.putText(frame, "Phase Timing", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    for i, st in enumerate(STAGES):
        cnt = stage_counter.get(st, 0)
        cv2.putText(
            frame,
            f"{STAGE_LABELS[st]}: {cnt/max(fps,1):.1f}s  {cnt/total*100:.0f}%",
            (x0, y0 + 22*(i+1)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, cols[st], 1,
        )


def _draw_timer(frame, elapsed: float):
    m, s = int(elapsed) // 60, elapsed % 60
    cv2.putText(frame, f"Lift: {m:01d}:{s:05.2f}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 100), 2)


def _draw_issues(frame, issues: List[str], h: int):
    for i, msg in enumerate(issues[:3]):
        cv2.putText(frame, f"! {msg}", (30, h - 120 - i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 80, 255), 2)


def _draw_machine_call_overlay(frame, machine_call: str, machine_reason: str, w: int, h: int):
    """
    Draw a prominent final machine call banner once the lift is complete.
    Shown after RELEASE_FINISH is detected and form is frozen.
    """
    if machine_call == "GREEN LIGHT":
        bg_color   = (0, 160, 0)
        text_color = (255, 255, 255)
    elif machine_call == "MAJORITY GREEN":
        bg_color   = (0, 140, 80)
        text_color = (220, 255, 220)
    else:
        bg_color   = (30, 30, 200)
        text_color = (255, 180, 180)

    # Banner background
    bx, by, bw, bh = w - 420, 10, 410, 56
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), bg_color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)

    cv2.putText(frame, f"MACHINE: {machine_call}",
                (bx + 10, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2)

    # Truncate reason to fit
    reason_short = machine_reason[:55] + "..." if len(machine_reason) > 55 else machine_reason
    cv2.putText(frame, reason_short,
                (bx + 10, by + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.34, text_color, 1)


def _top_issues(log: List[List[str]], n: int = 5) -> List[str]:
    flat = [i.split(" (")[0] for sub in log for i in sub]
    return [m for m, _ in Counter(flat).most_common(n)]


# ─── Main ─────────────────────────────────────────────────────────────────────

def analyze_clean_jerk_video(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Analyse a clean & jerk video.
    Always returns a plain dict (JSON-serialisable).
    On internal error: {"error": True, "context": ..., "message": ..., "traceback": ...}
    """
    _ckpt("analyze_clean_jerk_video START", f"input={input_path}")

    # ── Step 1: ML models ─────────────────────────────────────────────────────
    ml_status = load_ml_models()
    _ckpt("ML status", ml_status)

    # ── Step 2: Validate input ────────────────────────────────────────────────
    _ckpt("Validating input file")
    if not os.path.exists(input_path):
        return _make_error("Step 2 – input validation",
                           FileNotFoundError(f"Not found: {input_path}"))
    if os.path.getsize(input_path) == 0:
        return _make_error("Step 2 – input validation",
                           ValueError(f"Empty file: {input_path}"))
    _ckpt("Input OK", f"{os.path.getsize(input_path)//1024} KB")

    # ── Step 3: VideoCapture ──────────────────────────────────────────────────
    _ckpt("Opening VideoCapture")
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed: {input_path}")
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps     = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width == 0 or height == 0:
            raise RuntimeError(f"Invalid dimensions {width}×{height}")
        _ckpt("VideoCapture OK", f"{width}×{height} @ {fps}fps ~{n_total} frames")
    except Exception as e:
        try: cap.release()
        except: pass
        return _make_error("Step 3 – VideoCapture", e)

    # ── Step 4: VideoWriter ───────────────────────────────────────────────────
    _ckpt("Opening VideoWriter", output_path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not out.isOpened():
            raise RuntimeError(f"VideoWriter failed: {output_path}")
        _ckpt("VideoWriter OK")
    except Exception as e:
        cap.release()
        return _make_error("Step 4 – VideoWriter", e)

    # ── Step 5: MediaPipe Pose ────────────────────────────────────────────────
    _ckpt("Init MediaPipe Pose")
    try:
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        _ckpt("Pose OK")
    except Exception as e:
        cap.release(); out.release()
        return _make_error("Step 5 – MediaPipe Pose", e)

    # ── State ─────────────────────────────────────────────────────────────────
    stage_counter:     Dict[str, int]             = {}
    stage_issues_log:  Dict[str, List[List[str]]] = {}
    stage_start_frame: Dict[str, Optional[int]]   = {s: None for s in STAGES}
    stage_end_frame:   Dict[str, Optional[int]]   = {s: None for s in STAGES}
    stage_transitions: Dict[str, Optional[float]] = {s: None for s in STAGES}

    stage_window           = deque(maxlen=7)
    current_stage: Optional[str] = None
    highest_reached         = -1
    consecutive_bad: Dict[str, int] = {}

    lift_started     = False
    lift_start_frame = None
    touch_window     = deque(maxlen=5)

    total_frames = 0
    bad_frames   = 0

    # ── Form freeze state — set once RELEASE_FINISH is first detected ─────────
    form_frozen      = False          # True after bar is released / lift complete
    frozen_form: Dict[str, Any] = {}  # snapshot of form at freeze moment
    frozen_machine_call   = ""
    frozen_machine_reason = ""

    # Running per-phase form for the live gauge
    running_phase_form: Dict[str, float] = {
        "CLEAN_TO_SHOULDER": 100.0,
        "JERK_OVERHEAD":     100.0,
        "RELEASE_FINISH":    100.0,
    }

    # ── Step 6: Frame loop ────────────────────────────────────────────────────
    _ckpt("Frame loop START")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            if total_frames % 150 == 0:
                _ckpt("Progress",
                      f"frame={total_frames}/{n_total} stage={current_stage} "
                      f"lift={lift_started} bad={bad_frames} frozen={form_frozen}")

            try:
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logger.warning(f"[frame {total_frames}] pose error: {e}")
                out.write(frame); continue

            current_issues: List[str] = []
            angles: Dict              = {}

            if res.pose_landmarks:
                lm   = res.pose_landmarks.landmark
                flat = landmarks_to_flat(lm)

                # Bar touch
                if not lift_started:
                    try:
                        touch_window.append(detect_bar_touch(lm))
                        if sum(touch_window) >= 3:
                            lift_started     = True
                            lift_start_frame = total_frames
                            stage_counter["SETUP_BAR_TOUCH"]     = 0
                            stage_start_frame["SETUP_BAR_TOUCH"] = total_frames
                            stage_transitions["SETUP_BAR_TOUCH"] = round(total_frames / fps, 2)
                            _ckpt("BAR TOUCH", f"frame={total_frames}")
                    except Exception as e:
                        logger.warning(f"[frame {total_frames}] bar_touch: {e}")

                # Stage + form — only update if form is NOT yet frozen
                if lift_started:
                    try:
                        detected, angles = detect_stage(lm, flat)
                        if detected:
                            stage_window.append(detected)
                            smoothed = max(set(stage_window), key=list(stage_window).count)
                            order    = STAGE_ORDER.get(smoothed, -1)
                            if order >= highest_reached:
                                if smoothed != current_stage:
                                    consecutive_bad.clear()
                                    _ckpt("Stage →", f"{current_stage} → {smoothed} frame={total_frames}")

                                    # ── Freeze form when RELEASE_FINISH is first entered ──
                                    if smoothed == "RELEASE_FINISH" and not form_frozen:
                                        # Complete the RELEASE_FINISH phase scoring first,
                                        # then freeze on the NEXT stage tick.
                                        pass  # freeze happens below after counting

                                highest_reached = order
                                current_stage   = smoothed
                    except Exception as e:
                        logger.warning(f"[frame {total_frames}] detect_stage: {e}")

                    if current_stage and not form_frozen:
                        stage_counter[current_stage] = stage_counter.get(current_stage, 0) + 1
                        if stage_start_frame[current_stage] is None:
                            stage_start_frame[current_stage] = total_frames
                            stage_transitions[current_stage] = round(total_frames / fps, 2)
                        stage_end_frame[current_stage] = total_frames

                        try:
                            current_issues = evaluate_form(
                                current_stage, angles, consecutive_bad
                            )
                            if current_issues:
                                bad_frames += 1
                                stage_issues_log.setdefault(current_stage, []).append(current_issues)
                        except Exception as e:
                            logger.warning(f"[frame {total_frames}] evaluate_form: {e}")

                        # Update live phase form
                        try:
                            if current_stage in running_phase_form:
                                running_phase_form[current_stage] = compute_phase_form(
                                    current_stage, stage_counter, stage_issues_log
                                )
                        except Exception as e:
                            logger.warning(f"[frame {total_frames}] phase_form update: {e}")

                        # ── Freeze form once RELEASE_FINISH has been scored ────────────
                        # Freeze after the first frame in RELEASE_FINISH has been counted
                        # so the phase itself is reflected in the final score.
                        if current_stage == "RELEASE_FINISH" and not form_frozen:
                            form_frozen   = True
                            frozen_form   = compute_overall_form(stage_counter, stage_issues_log)
                            frozen_machine_call, frozen_machine_reason = compute_machine_call(
                                frozen_form, stage_counter, stage_issues_log
                            )
                            _ckpt(
                                "FORM FROZEN",
                                f"frame={total_frames} "
                                f"overall={frozen_form['overall']}% "
                                f"call={frozen_machine_call}",
                            )

                # Draw
                try:
                    _draw_skeleton(frame, res.pose_landmarks)
                    _draw_joints(frame, lm, width, height)
                    if angles:
                        _draw_angles(frame, lm, width, height, angles)
                except Exception as e:
                    logger.warning(f"[frame {total_frames}] draw: {e}")

            # HUD
            try:
                if current_stage:
                    cv2.putText(frame, STAGE_LABELS[current_stage],
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 60), 2)

                if lift_started and lift_start_frame:
                    _draw_timer(frame, (total_frames - lift_start_frame) / fps)

                # Use frozen form once available, otherwise compute live
                if form_frozen:
                    display_form    = frozen_form
                    display_overall = frozen_form["overall"]
                else:
                    display_form    = compute_overall_form(stage_counter, stage_issues_log)
                    display_overall = display_form["overall"]

                _draw_gauge(
                    frame, display_overall, running_phase_form,
                    width, height, lift_started,
                    frozen=form_frozen,
                )
                _draw_phases(frame, stage_counter, fps, width)
                _draw_issues(frame, current_issues if not form_frozen else [], height)

                # Overlay final machine call banner once lift is complete
                if form_frozen:
                    _draw_machine_call_overlay(
                        frame, frozen_machine_call, frozen_machine_reason, width, height
                    )

            except Exception as e:
                logger.warning(f"[frame {total_frames}] HUD: {e}")

            out.write(frame)

    except Exception as e:
        logger.error(f"[loop] Fatal at frame {total_frames}:\n{traceback.format_exc()}")
        cap.release(); out.release(); pose.close()
        return _make_error(f"Step 6 – Frame loop (frame {total_frames})", e)
    finally:
        cap.release()
        out.release()
        pose.close()
        _ckpt("Frame loop END", f"total={total_frames} bad={bad_frames} frozen={form_frozen}")

    # ── Step 7: Final scoring ─────────────────────────────────────────────────
    _ckpt("Final scoring", f"stage_counter={stage_counter}")
    try:
        # Use frozen form/call if available (lift was completed normally),
        # otherwise compute from whatever was detected.
        if form_frozen:
            form         = frozen_form
            machine_call   = frozen_machine_call
            machine_reason = frozen_machine_reason
        else:
            form           = compute_overall_form(stage_counter, stage_issues_log)
            machine_call, machine_reason = compute_machine_call(
                form, stage_counter, stage_issues_log
            )

        _ckpt("Score done",
              f"overall={form['overall']}% "
              f"clean={form['CLEAN_TO_SHOULDER']}% "
              f"jerk={form['JERK_OVERHEAD']}% "
              f"release={form['RELEASE_FINISH']}% "
              f"call={machine_call}")
    except Exception as e:
        return _make_error("Step 7 – Scoring", e)

    # ── Step 8: Build response ────────────────────────────────────────────────
    _ckpt("Building response")
    try:
        active_total = sum(stage_counter.get(s, 0) for s in SUCCESS_STAGES)

        stage_summary = {}
        for st in STAGES:
            cnt = stage_counter.get(st, 0)
            phase_pct = form.get(st, 0.0) if st in SUCCESS_STAGES else None
            stage_summary[st] = {
                "label":              STAGE_LABELS[st],
                "frame_count":        cnt,
                "percentage":         round(cnt / max(total_frames, 1) * 100, 1),
                "detected":           cnt > 0,
                "top_issues":         _top_issues(stage_issues_log.get(st, []), n=5),
                "form_ok":            len(stage_issues_log.get(st, [])) == 0,
                "phase_form_percent": phase_pct,
                "duration_seconds":   round(cnt / max(fps, 1), 2),
                "start_frame":        stage_start_frame[st],
                "end_frame":          stage_end_frame[st],
            }

        total_detected = sum(stage_counter.get(s, 0) for s in STAGES)
        phase_durs = {s: round(stage_counter.get(s, 0) / max(fps, 1), 2) for s in STAGES}
        phase_pcts = {
            s: round(stage_counter.get(s, 0) / max(total_detected, 1) * 100, 1)
            for s in STAGES
        }
        sorted_ph = sorted(STAGES, key=lambda s: phase_durs[s])

        result = {
            "machine_call":          machine_call,    # "GREEN LIGHT" | "MAJORITY GREEN" | "RED LIGHT"
            "machine_reason":        machine_reason,
            # Overall weighted form (frozen at RELEASE_FINISH)
            "form_accuracy_percent": form["overall"],
            "form_frozen":           form_frozen,      # True = scored up to release only
            # Per-phase form breakdown
            "phase_form": {
                "clean_to_shoulder": form["CLEAN_TO_SHOULDER"],
                "jerk_overhead":     form["JERK_OVERHEAD"],
                "release_finish":    form["RELEASE_FINISH"],
            },
            "pass_threshold":     PASS_THRESHOLD,
            "total_frames":       total_frames,
            "active_lift_frames": active_total,
            "bad_frames":         bad_frames,
            "stage_summary":      stage_summary,
            "phase_timing": {
                "phase_durations_seconds":     phase_durs,
                "phase_percentages":           phase_pcts,
                "total_lift_duration_seconds": round(total_detected / max(fps, 1), 2),
                "phase_transitions_seconds":   dict(stage_transitions),
                "fastest_phase":               sorted_ph[0],
                "slowest_phase":               sorted_ph[-1],
            },
            "processed_video_path": output_path,
        }
    except Exception as e:
        return _make_error("Step 8 – Response dict", e)

    # ── Step 9: JSON check ────────────────────────────────────────────────────
    try:
        import json
        json.dumps(result)
        _ckpt("JSON OK")
    except (TypeError, ValueError) as e:
        return _make_error("Step 9 – JSON check", e)

    _ckpt("COMPLETE", f"call={machine_call} form={form['overall']}% frozen={form_frozen}")
    return result
