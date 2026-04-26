"""
app/services/squat/squat_live_service.py
Live webcam session service for squat analysis.

Mirrors the structure of pushup_live_service.py, but runs the
Gradient Boosting squat analyser (v5 logic) in a background thread.

Public API
──────────
    session = LiveSquatSession(camera_index=0)
    session.start()
    frame_bytes = session.get_frame()   # JPEG bytes or None
    stats       = session.get_stats()   # dict snapshot
    packet      = session.stop()        # final summary dict
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import pickle
import threading
import time
import warnings
from pathlib import Path
from collections import deque, Counter
from typing import Optional, List, Dict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  MODEL PATHS
# ─────────────────────────────────────────────
_BASE     = Path(__file__).resolve().parents[3]
GB_MODEL  = _BASE / "ML_Model" / "squat_model" / "saved_models_v2" / "GradientBoosting.pkl"
META_JSON = _BASE / "ML_Model" / "squat_model" / "saved_models_v2" / "best_model_meta.json"

# ─────────────────────────────────────────────
#  THRESHOLDS  (same as offline service)
# ─────────────────────────────────────────────
KNEE_ANGLE_DOWN     = 110
KNEE_ANGLE_UP       = 155
FRAME_CONFIRM       = 4
SMOOTHING_WINDOW    = 7
GOOD_FORM_THRESHOLD = 0.65
MAX_SPINE_LEAN_DEG  = 45.0
MAX_KNEE_CAVE_RATIO = 0.80
MAX_KNEE_ANGLE_DIFF = 18.0
MAX_HIP_Y_SYMMETRY  = 0.06
MIN_DEPTH_ANGLE     = 115.0
MAX_KNEE_TOE_X_DIFF = 0.12
FATIGUE_WINDOW      = 5

JPEG_QUALITY = 80   # MJPEG compression (0-100)

mp_pose = mp.solutions.pose


# ─────────────────────────────────────────────
#  GEOMETRY  (identical to squat_service.py)
# ─────────────────────────────────────────────
def _angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def _xy(lm, i):
    return [lm[i].x, lm[i].y]

def _avg_knee_angle(lm):
    L = _angle(_xy(lm,23), _xy(lm,25), _xy(lm,27))
    R = _angle(_xy(lm,24), _xy(lm,26), _xy(lm,28))
    return (L + R) / 2.0


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────
def _extract_features(lm):
    f = {}
    msx = (lm[11].x + lm[12].x) / 2;  msy = (lm[11].y + lm[12].y) / 2
    mhx = (lm[23].x + lm[24].x) / 2;  mhy = (lm[23].y + lm[24].y) / 2

    f["left_knee_angle"]   = _angle(_xy(lm,23), _xy(lm,25), _xy(lm,27))
    f["right_knee_angle"]  = _angle(_xy(lm,24), _xy(lm,26), _xy(lm,28))
    f["avg_knee_angle"]    = (f["left_knee_angle"] + f["right_knee_angle"]) / 2
    f["knee_angle_diff"]   = abs(f["left_knee_angle"] - f["right_knee_angle"])

    f["left_hip_angle"]    = _angle(_xy(lm,11), _xy(lm,23), _xy(lm,25))
    f["right_hip_angle"]   = _angle(_xy(lm,12), _xy(lm,24), _xy(lm,26))
    f["avg_hip_angle"]     = (f["left_hip_angle"] + f["right_hip_angle"]) / 2
    f["hip_angle_diff"]    = abs(f["left_hip_angle"] - f["right_hip_angle"])

    f["left_trunk_angle"]  = _angle(_xy(lm,11), _xy(lm,23), [lm[23].x, lm[23].y+0.1])
    f["right_trunk_angle"] = _angle(_xy(lm,12), _xy(lm,24), [lm[24].x, lm[24].y+0.1])
    f["avg_trunk_angle"]   = (f["left_trunk_angle"] + f["right_trunk_angle"]) / 2

    f["left_ankle_angle"]  = _angle(_xy(lm,25), _xy(lm,27), _xy(lm,31))
    f["right_ankle_angle"] = _angle(_xy(lm,26), _xy(lm,28), _xy(lm,32))
    f["avg_ankle_angle"]   = (f["left_ankle_angle"] + f["right_ankle_angle"]) / 2

    dx, dy = msx - mhx, msy - mhy
    f["spine_lean_angle"]  = float(np.degrees(np.arctan2(abs(dx), abs(dy)+1e-6)))
    f["left_elbow_angle"]  = _angle(_xy(lm,11), _xy(lm,13), _xy(lm,15))
    f["right_elbow_angle"] = _angle(_xy(lm,12), _xy(lm,14), _xy(lm,16))

    f["left_hip_knee_y_diff"]  = lm[23].y - lm[25].y
    f["right_hip_knee_y_diff"] = lm[24].y - lm[26].y
    f["avg_hip_knee_y_diff"]   = (f["left_hip_knee_y_diff"] + f["right_hip_knee_y_diff"]) / 2
    f["hip_below_knee"]        = float(f["avg_hip_knee_y_diff"] > 0)

    f["left_knee_toe_x_diff"]  = lm[25].x - lm[31].x
    f["right_knee_toe_x_diff"] = lm[26].x - lm[32].x
    f["avg_knee_toe_x_diff"]   = (f["left_knee_toe_x_diff"] + f["right_knee_toe_x_diff"]) / 2

    knee_w  = abs(lm[25].x - lm[26].x)
    ankle_w = abs(lm[27].x - lm[28].x)
    hip_w   = abs(lm[23].x - lm[24].x)
    shldr_w = abs(lm[11].x - lm[12].x)

    f["knee_ankle_width_ratio"]   = knee_w  / (ankle_w + 1e-6)
    f["shoulder_hip_width_ratio"] = shldr_w / (hip_w   + 1e-6)
    f["stance_width"]             = ankle_w
    f["hip_y_symmetry"]           = abs(lm[23].y - lm[24].y)
    f["shoulder_y_symmetry"]      = abs(lm[11].y - lm[12].y)
    f["knee_y_symmetry"]          = abs(lm[25].y - lm[26].y)
    f["ankle_y_symmetry"]         = abs(lm[27].y - lm[28].y)

    for name, idx in [
        ("nose",0),("l_shoulder",11),("r_shoulder",12),
        ("l_hip",23),("r_hip",24),("l_knee",25),("r_knee",26),
        ("l_ankle",27),("r_ankle",28),("l_foot",31),("r_foot",32),
    ]:
        f[name + "_rel_x"] = lm[idx].x - mhx
        f[name + "_rel_y"] = lm[idx].y - mhy

    ys = [lm[i].y for i in [0,11,12,23,24,25,26,27,28]]
    f["body_height_range"] = max(ys) - min(ys)
    return f


# ─────────────────────────────────────────────
#  SQUAT VALIDATOR
# ─────────────────────────────────────────────
def _is_squat_position(lm):
    try:
        l_sh,  r_sh  = lm[11], lm[12]
        l_hip, r_hip = lm[23], lm[24]
        l_kn,  r_kn  = lm[25], lm[26]
        l_an,  r_an  = lm[27], lm[28]
        nose         = lm[0]
        mid_sh_y  = (l_sh.y  + r_sh.y)  / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2
        mid_kn_y  = (l_kn.y  + r_kn.y)  / 2
        mid_an_y  = (l_an.y  + r_an.y)  / 2
        mid_sh_x  = (l_sh.x  + r_sh.x)  / 2
        mid_hip_x = (l_hip.x + r_hip.x) / 2
        if not (nose.y < mid_sh_y < mid_hip_y < mid_kn_y < mid_an_y):
            return False
        if mid_sh_y > mid_hip_y - 0.05:
            return False
        vert = abs(mid_hip_y - mid_sh_y) + 1e-6
        if abs(mid_sh_x - mid_hip_x) / vert > 0.60:
            return False
        if hasattr(l_an, "visibility") and l_an.visibility < 0.3 and r_an.visibility < 0.3:
            return False
        if abs(l_kn.x - r_kn.x) < 0.01:
            return False
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
#  FORM FAULTS
# ─────────────────────────────────────────────
def _detect_faults(feats, min_knee_in_rep=999.0, check_depth=False):
    faults = []
    if feats.get("spine_lean_angle", 0) > MAX_SPINE_LEAN_DEG:
        faults.append("Leaning too far forward")
    if feats.get("knee_ankle_width_ratio", 1.0) < MAX_KNEE_CAVE_RATIO:
        faults.append("Knees caving in")
    if feats.get("knee_angle_diff", 0) > MAX_KNEE_ANGLE_DIFF:
        faults.append("Uneven knee bend")
    if feats.get("hip_y_symmetry", 0) > MAX_HIP_Y_SYMMETRY:
        faults.append("Hips tilting sideways")
    if abs(feats.get("avg_knee_toe_x_diff", 0)) > MAX_KNEE_TOE_X_DIFF:
        faults.append("Knees too far over toes")
    if check_depth and min_knee_in_rep > MIN_DEPTH_ANGLE:
        faults.append("Squat too shallow")
    return faults


# ─────────────────────────────────────────────
#  PREDICT HELPERS
# ─────────────────────────────────────────────
def _predict_prob(pipeline, feature_row, feature_names):
    try:
        X = (pd.DataFrame([feature_row])[feature_names]
             if feature_names else pd.DataFrame([feature_row]))
        classes = list(pipeline.classes_)
        proba   = pipeline.predict_proba(X)[0]
        return float(proba[classes.index(1)] if 1 in classes else proba[-1])
    except Exception:
        return 0.5

def _predict_max_reps(rep_history, total_reps):
    if len(rep_history) < 2:
        return f"~{max(total_reps + 4, 5)}" if total_reps > 0 else "—"
    confs  = [c for _, c in rep_history]
    recent = confs[-min(FATIGUE_WINDOW, len(confs)):]
    avg_r  = float(np.mean(recent))
    slope  = float(np.polyfit(range(len(recent)), recent, 1)[0]) if len(recent) >= 3 else 0.0
    if slope < -0.005 and avg_r > GOOD_FORM_THRESHOLD:
        extra = int((avg_r - GOOD_FORM_THRESHOLD) / (abs(slope) + 1e-6))
        return f"~{total_reps + max(0, extra)}"
    elif avg_r >= GOOD_FORM_THRESHOLD:
        return f"~{total_reps + max(5, int(total_reps * 0.5))}"
    else:
        return f"~{total_reps + max(1, int(total_reps * 0.1))}"


# ─────────────────────────────────────────────
#  OVERLAY (compact live version)
# ─────────────────────────────────────────────
def _draw_banner(frame, text, bg_color, text_color):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    fsc  = max(0.8, w / 900)
    ts   = cv2.getTextSize(text, font, fsc, 2)[0]
    pad  = 24
    bh   = ts[1] + pad * 2
    by   = (h - bh) // 2
    ov   = frame.copy()
    cv2.rectangle(ov, (0, by), (w, by + bh), bg_color, -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    tx = (w - ts[0]) // 2
    ty = by + pad + ts[1]
    cv2.putText(frame, text, (tx+2, ty+2), font, fsc, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (tx,   ty),   font, fsc, text_color, 2, cv2.LINE_AA)

def _draw_overlay(frame, state: dict):
    h, w = frame.shape[:2]

    total_reps      = state["total_reps"]
    good_reps       = state["good_reps"]
    bad_reps        = state["bad_reps"]
    cur_prob        = state["cur_prob"]
    phase           = state["phase"]
    squat_valid     = state["squat_valid"]
    person_detected = state["person_detected"]
    max_pred        = state["max_pred"]
    live_faults     = state["live_faults"]
    smooth_k        = state["smooth_k"]
    phase_steps     = state["phase_steps"]

    form_active = person_detected and squat_valid and (phase_steps >= 2)
    form_good   = form_active and (cur_prob >= GOOD_FORM_THRESHOLD)
    accent      = (0, 220, 80) if form_good else (0, 60, 230)

    # Rep block
    cv2.rectangle(frame, (0, 0), (220, 200), (0,0,0), -1)
    cv2.rectangle(frame, (0, 0), (220, 200), accent, 2)
    cv2.putText(frame, str(total_reps),
                (15, 130), cv2.FONT_HERSHEY_DUPLEX, 4.8, accent, 7, cv2.LINE_AA)
    cv2.putText(frame, "REPS",
                (15, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"G:{good_reps}",
                (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,220,80), 2, cv2.LINE_AA)
    cv2.putText(frame, f"B:{bad_reps}",
                (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,60,230), 2, cv2.LINE_AA)

    if person_detected:
        phase_colors = {
            "WAIT":              (150,150,150),
            "1: Stand UP":       (0,200,255),
            "2: Squat DOWN":     (0,100,255),
            "3: Stand UP \u2713":(0,220,80),
        }
        pcol = phase_colors.get(phase, (200,200,200))
        pts  = cv2.getTextSize(phase, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
        px   = (w - pts[0]) // 2
        cv2.putText(frame, phase, (px+2, 52), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0,0,0), 6, cv2.LINE_AA)
        cv2.putText(frame, phase, (px,   50), cv2.FONT_HERSHEY_DUPLEX, 0.85, pcol,    2, cv2.LINE_AA)
        cv2.putText(frame, f"Knee: {smooth_k:.1f}",
                    (15, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2, cv2.LINE_AA)

        step_labels = ["\u25bc DOWN", "\u25b2 UP", "\u25bc DOWN", "\u25b2 UP = REP"]
        bar_x=240; bar_y=18; box_w=130; box_h=30; gap=8
        for i, lbl in enumerate(step_labels):
            bx = bar_x + i * (box_w + gap)
            done = i < phase_steps
            cv2.rectangle(frame, (bx, bar_y), (bx+box_w, bar_y+box_h),
                          (0,160,60) if done else (40,40,40), -1)
            cv2.rectangle(frame, (bx, bar_y), (bx+box_w, bar_y+box_h),
                          (0,220,80) if done else (80,80,80), 1)
            ts = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            cv2.putText(frame, lbl,
                        (bx+(box_w-ts[0])//2, bar_y+(box_h+ts[1])//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255,255,255) if done else (120,120,120), 1, cv2.LINE_AA)

    if form_active and live_faults:
        px = w - 360; py = 10
        ph = 36 + len(live_faults) * 32
        cv2.rectangle(frame, (px-10, py), (w-10, py+ph), (10,10,40), -1)
        cv2.rectangle(frame, (px-10, py), (w-10, py+ph), (0,60,230), 2)
        cv2.putText(frame, "FORM ISSUES", (px, py+26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,120,255), 2, cv2.LINE_AA)
        for i, fault in enumerate(live_faults):
            cv2.putText(frame, f"  * {fault}", (px, py+26+(i+1)*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,180,255), 1, cv2.LINE_AA)

    bm=16; bh2=18
    by2 = h - bm - bh2
    bw  = w - 2*bm
    cv2.rectangle(frame, (bm, by2), (bm+bw, by2+bh2), (30,30,30), -1)
    if form_active:
        fw = int(bw * cur_prob)
        cv2.rectangle(frame, (bm, by2), (bm+fw, by2+bh2), accent, -1)
        tx2 = bm + int(bw * GOOD_FORM_THRESHOLD)
        cv2.line(frame, (tx2, by2-5), (tx2, by2+bh2+5), (255,255,0), 2)
        lbl = f"{cur_prob*100:.0f}%  {'GOOD FORM' if form_good else 'BAD FORM'}"
        cv2.putText(frame, lbl, (bm+4, by2-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 2, cv2.LINE_AA)
    else:
        msg = ("Form bar inactive — no person detected" if not person_detected
               else "Form bar inactive — get into squat position" if not squat_valid
               else "Form bar inactive — waiting for squat")
        cv2.putText(frame, msg, (bm+4, by2-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1, cv2.LINE_AA)

    mlabel = f"MAX {max_pred}"
    mt  = cv2.getTextSize(mlabel, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    mx  = w - mt[0] - bm
    cv2.putText(frame, mlabel, (mx+2, by2-8),  cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0),     4, cv2.LINE_AA)
    cv2.putText(frame, mlabel, (mx,   by2-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,200,0), 2, cv2.LINE_AA)

    if not person_detected:
        _draw_banner(frame, "NO ONE DETECTED", (30,30,30), (0,200,255))
    elif not squat_valid:
        _draw_banner(frame, "NO SQUAT DETECTED — STEP INTO POSITION", (20,20,60), (100,180,255))


# ─────────────────────────────────────────────
#  LIVE SESSION CLASS
# ─────────────────────────────────────────────
class LiveSquatSession:
    """
    Manages a single live webcam squat analysis session.

    Usage
    -----
        session = LiveSquatSession(camera_index=0)
        session.start()
        # in MJPEG generator:
        jpeg = session.get_frame()
        # poll stats:
        stats = session.get_stats()
        # end session:
        summary = session.stop()
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._running     = False
        self._thread      = None
        self._lock        = threading.Lock()

        # Latest JPEG frame bytes
        self._latest_frame: Optional[bytes] = None

        # Live state (read by get_stats)
        self._state = {
            "total_reps":      0,
            "good_reps":       0,
            "bad_reps":        0,
            "cur_prob":        0.5,
            "phase":           "WAIT",
            "squat_valid":     False,
            "person_detected": False,
            "max_pred":        "—",
            "live_faults":     [],
            "smooth_k":        170.0,
            "phase_steps":     0,
        }

        # History (used in stop() summary)
        self._rep_history: list  = []
        self._fault_log:   dict  = {}
        self._start_time: float  = 0.0

        # Load model once at construction time
        if not GB_MODEL.exists():
            raise FileNotFoundError(f"GB model not found: {GB_MODEL}")
        with open(GB_MODEL, "rb") as fh:
            self._pipeline = pickle.load(fh)
        self._feature_names = None
        if META_JSON.exists():
            try:
                with open(META_JSON) as fh:
                    self._feature_names = json.load(fh).get("feature_names")
            except Exception:
                pass
        print(f"[LiveSquatSession] Model loaded. camera_index={camera_index}")

    # ── Public ──────────────────────────────────────────────────
    def start(self):
        if self._running:
            raise RuntimeError("Session already running.")
        self._running    = True
        self._start_time = time.time()
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[LiveSquatSession] Background thread started.")

    def stop(self) -> dict:
        print("[LiveSquatSession] Stopping session…")
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        return self._build_summary()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def get_stats(self) -> dict:
        with self._lock:
            s = self._state.copy()
        total = s["total_reps"]
        return {
            "status":              "running",
            "total_reps":          total,
            "good_reps":           s["good_reps"],
            "bad_reps":            s["bad_reps"],
            "form_rate_percent":   round(s["good_reps"]/total*100, 1) if total else 0.0,
            "predicted_max_reps":  s["max_pred"],
            "current_phase":       s["phase"],
            "person_detected":     s["person_detected"],
            "squat_valid":         s["squat_valid"],
            "current_form_score":  round(s["cur_prob"]*100, 1),
            "live_faults":         s["live_faults"],
            "knee_angle":          round(s["smooth_k"], 1),
            "session_duration_sec": round(time.time() - self._start_time, 1),
        }

    # ── Background loop ─────────────────────────────────────────
    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[LiveSquatSession] ERROR: cannot open camera {self.camera_index}")
            self._running = False
            return

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Per-run state
        knee_buf            = deque(maxlen=SMOOTHING_WINDOW)
        smooth_k            = 170.0
        cycle_step          = 0
        frames_in_down_zone = 0
        frames_in_up_zone   = 0

        good_reps      = 0
        bad_reps       = 0
        rep_history    = []
        fault_log      = {}
        cur_rep_confs  = []
        cur_rep_faults = []
        min_knee       = 999.0

        cur_prob        = 0.5
        squat_valid     = False
        person_detected = False
        max_pred        = "—"
        live_faults     = []

        PHASE_LABEL = {
            0: "WAIT",
            1: "1: Stand UP",
            2: "2: Squat DOWN",
            3: "3: Stand UP \u2713",
        }

        print("[LiveSquatSession] Processing loop started.")

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            person_detected = result.pose_landmarks is not None

            if person_detected:
                lm = result.pose_landmarks.landmark

                squat_valid = _is_squat_position(lm)
                raw_k       = _avg_knee_angle(lm)
                knee_buf.append(raw_k)
                smooth_k = float(np.mean(knee_buf))

                try:
                    feats       = _extract_features(lm)
                    cur_prob    = _predict_prob(self._pipeline, feats, self._feature_names)
                    live_faults = _detect_faults(feats) if squat_valid else []
                except Exception:
                    feats       = {}
                    cur_prob    = 0.5
                    live_faults = []

                # Zone counters
                if smooth_k < KNEE_ANGLE_DOWN:
                    frames_in_down_zone += 1;  frames_in_up_zone   = 0
                elif smooth_k > KNEE_ANGLE_UP:
                    frames_in_up_zone   += 1;  frames_in_down_zone = 0
                else:
                    frames_in_down_zone  = 0;  frames_in_up_zone   = 0

                confirmed_down = frames_in_down_zone >= FRAME_CONFIRM
                confirmed_up   = frames_in_up_zone   >= FRAME_CONFIRM

                # 4-step cycle
                if cycle_step == 0:
                    if confirmed_down:
                        cycle_step = 1; frames_in_down_zone = 0

                elif cycle_step == 1:
                    if confirmed_up:
                        cycle_step = 2; frames_in_up_zone = 0

                elif cycle_step == 2:
                    cur_rep_confs  = []
                    cur_rep_faults = []
                    min_knee       = 999.0
                    if confirmed_down:
                        cycle_step = 3; frames_in_down_zone = 0

                elif cycle_step == 3:
                    if squat_valid:
                        cur_rep_confs.append(cur_prob)
                        if smooth_k < min_knee:
                            min_knee = smooth_k
                        for fault in live_faults:
                            if fault not in cur_rep_faults:
                                cur_rep_faults.append(fault)

                    if confirmed_up:
                        cycle_step = 2; frames_in_up_zone = 0

                        avg_conf = float(np.mean(cur_rep_confs)) if cur_rep_confs else 0.5
                        depth_faults = _detect_faults(
                            feats if feats else {}, min_knee, check_depth=True)
                        for df in depth_faults:
                            if df not in cur_rep_faults:
                                cur_rep_faults.append(df)
                        if min_knee > MIN_DEPTH_ANGLE:
                            avg_conf *= 0.75

                        total_reps = good_reps + bad_reps + 1
                        rep_history.append((total_reps, avg_conf))
                        fault_log[total_reps] = list(cur_rep_faults)

                        if avg_conf >= GOOD_FORM_THRESHOLD:
                            good_reps += 1
                        else:
                            bad_reps  += 1

                        max_pred       = _predict_max_reps(rep_history, total_reps)
                        cur_rep_confs  = []
                        cur_rep_faults = []
                        min_knee       = 999.0

                        print(f"  [REP {total_reps}] "
                              f"{'GOOD' if avg_conf>=GOOD_FORM_THRESHOLD else 'BAD '} "
                              f"{avg_conf*100:.1f}%  depth={min_knee:.1f}°")

            else:
                squat_valid         = False
                cur_prob            = 0.5
                live_faults         = []
                frames_in_down_zone = 0
                frames_in_up_zone   = 0
                knee_buf.clear()

            # Build state snapshot
            phase       = PHASE_LABEL[cycle_step]
            phase_steps = cycle_step
            snap = {
                "total_reps":      good_reps + bad_reps,
                "good_reps":       good_reps,
                "bad_reps":        bad_reps,
                "cur_prob":        cur_prob,
                "phase":           phase,
                "squat_valid":     squat_valid,
                "person_detected": person_detected,
                "max_pred":        max_pred,
                "live_faults":     list(live_faults),
                "smooth_k":        smooth_k,
                "phase_steps":     phase_steps,
            }

            # Draw overlay on frame
            _draw_overlay(frame, snap)

            # Encode to JPEG
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            with self._lock:
                self._state        = snap
                self._rep_history  = list(rep_history)
                self._fault_log    = dict(fault_log)
                if ok:
                    self._latest_frame = buf.tobytes()

        cap.release()
        pose.close()
        print("[LiveSquatSession] Processing loop ended.")

    # ── Final summary ───────────────────────────────────────────
    def _build_summary(self) -> dict:
        with self._lock:
            rep_history = list(self._rep_history)
            fault_log   = dict(self._fault_log)
            good_reps   = self._state["good_reps"]
            bad_reps    = self._state["bad_reps"]

        total_reps = good_reps + bad_reps
        form_rate  = round(good_reps / total_reps * 100, 1) if total_reps else 0.0
        duration   = round(time.time() - self._start_time, 1)

        all_faults   = [f for fs in fault_log.values() for f in fs]
        fault_counts = dict(Counter(all_faults).most_common())

        rep_log = []
        for rep_no, conf in rep_history:
            rep_log.append({
                "rep":        rep_no,
                "form_score": round(conf * 100, 1),
                "form_tag":   "GOOD" if conf >= GOOD_FORM_THRESHOLD else "BAD",
                "faults":     fault_log.get(rep_no, []),
            })

        return {
            "status":               "stopped",
            "total_reps":           total_reps,
            "good_reps":            good_reps,
            "bad_reps":             bad_reps,
            "form_rate_percent":    form_rate,
            "predicted_max_reps":   _predict_max_reps(rep_history, total_reps),
            "session_duration_sec": duration,
            "rep_log":              rep_log,
            "fault_summary":        fault_counts,
        }