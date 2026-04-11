"""
====================================================
  SQUAT  —  PER-FRAME MODEL REACTION VISUALIZER
  Shows each model classifying EVERY frame live.
  Right panel = scrolling confidence timeline per model.
  Terminal = full frame-by-frame log at the end.
  Press Q / Esc to quit early.
====================================================

HOW IT WORKS (printed to terminal on startup):
  1. MediaPipe extracts 33 body landmarks from each frame.
  2. extract_features() converts landmarks → ~50 joint angles,
     ratios, symmetry scores, and relative coordinates.
  3. Every frame's feature vector is fed into ALL loaded
     sklearn pipeline models simultaneously.
  4. Each model returns predict_proba() → P(GOOD form).
  5. The live display shows:
       LEFT  panel  : skeleton + squat state + key angles
       RIGHT panel  : per-model confidence bars + timeline graph
  6. The timeline graph scrolls left as the video plays,
     showing how each model's confidence rises/falls through
     the squat movement (down = low confidence, up = high).
  7. Terminal report at end shows every frame's raw numbers.
====================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import time
import pickle
import warnings
import traceback
from pathlib import Path
from collections import deque

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG — edit paths here
# ─────────────────────────────────────────────
INPUT_VIDEO = r"C:\major_project\serverSide\ML_Model\pushup_model\testing_video\squat2.mp4"
MODELS_DIR  = r"C:\major_project\serverSide\ML_Model\squat_model\saved_models"
META_JSON   = r"C:\major_project\serverSide\ML_Model\squat_model\saved_models\best_model_meta.json"

WINDOW_NAME     = "Per-Frame Model Reaction  [Q = quit]"
DISPLAY_W       = 1600   # total window width
DISPLAY_H       = 900    # total window height

# Squat thresholds (same as main script)
KNEE_DOWN_ANGLE  = 110
KNEE_UP_ANGLE    = 155
FRAME_CONFIRM    = 3
SMOOTHING_WINDOW = 5

# Timeline: how many frames to keep in the scrolling graph
TIMELINE_LEN = 200

# Per-model colour palette (BGR) — up to 8 models
MODEL_COLORS = [
    (0,   220, 255),   # cyan
    (0,   180, 0  ),   # green
    (255, 160, 0  ),   # orange
    (180, 0,   255),   # purple
    (0,   100, 255),   # blue
    (255, 60,  60 ),   # red
    (200, 200, 0  ),   # yellow
    (0,   200, 150),   # teal
]

# ─────────────────────────────────────────────
#  HOW IT WORKS — print on startup
# ─────────────────────────────────────────────
def print_how_it_works():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          HOW THE PER-FRAME CLASSIFIER WORKS                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  STEP 1  │ MediaPipe Pose                                    ║
║          │ Detects 33 body landmarks (x, y, visibility)      ║
║          │ from each raw video frame using a neural net.     ║
║                                                              ║
║  STEP 2  │ Feature Extraction  (extract_features)            ║
║          │ Converts landmarks into ~50 interpretable         ║
║          │ features: knee/hip/trunk/ankle angles, spine      ║
║          │ lean, symmetry scores, joint width ratios,        ║
║          │ relative coordinates anchored to mid-hip.         ║
║                                                              ║
║  STEP 3  │ Per-Frame Classification                          ║
║          │ Every frame's feature vector is passed to ALL     ║
║          │ loaded sklearn models via predict_proba().        ║
║          │ Each model outputs P(GOOD form) ∈ [0.0, 1.0].    ║
║                                                              ║
║  STEP 4  │ Live Display                                      ║
║          │ Left  : skeleton overlay + state + angles         ║
║          │ Right : confidence bars (current frame) +         ║
║          │         scrolling timeline graph (past 200 frames)║
║          │ Colours per model stay consistent throughout.     ║
║                                                              ║
║  STEP 5  │ Terminal Log                                      ║
║          │ After video ends, every frame's raw probabilities ║
║          │ are printed in a table so you can see exactly     ║
║          │ when and why each model changed its opinion.      ║
║                                                              ║
║  KEY INSIGHT: Models trained on averaged rep-level features  ║
║  will show GRADUAL confidence curves — dropping as the       ║
║  person bends (angles change) and rising as they stand up.   ║
║  Disagreement between models exposes uncertainty zones.      ║
╚══════════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
#  GEOMETRY
# ─────────────────────────────────────────────
def compute_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def lm_xy(lm, i):
    return [lm[i].x, lm[i].y]

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION  (identical to main script)
# ─────────────────────────────────────────────
def extract_features(lm):
    f = {}
    msx = (lm[11].x + lm[12].x) / 2;  msy = (lm[11].y + lm[12].y) / 2
    mhx = (lm[23].x + lm[24].x) / 2;  mhy = (lm[23].y + lm[24].y) / 2

    f["left_knee_angle"]   = compute_angle(lm_xy(lm,23), lm_xy(lm,25), lm_xy(lm,27))
    f["right_knee_angle"]  = compute_angle(lm_xy(lm,24), lm_xy(lm,26), lm_xy(lm,28))
    f["avg_knee_angle"]    = (f["left_knee_angle"] + f["right_knee_angle"]) / 2
    f["knee_angle_diff"]   = abs(f["left_knee_angle"] - f["right_knee_angle"])

    f["left_hip_angle"]    = compute_angle(lm_xy(lm,11), lm_xy(lm,23), lm_xy(lm,25))
    f["right_hip_angle"]   = compute_angle(lm_xy(lm,12), lm_xy(lm,24), lm_xy(lm,26))
    f["avg_hip_angle"]     = (f["left_hip_angle"] + f["right_hip_angle"]) / 2
    f["hip_angle_diff"]    = abs(f["left_hip_angle"] - f["right_hip_angle"])

    f["left_trunk_angle"]  = compute_angle(lm_xy(lm,11), lm_xy(lm,23), [lm[23].x, lm[23].y+0.1])
    f["right_trunk_angle"] = compute_angle(lm_xy(lm,12), lm_xy(lm,24), [lm[24].x, lm[24].y+0.1])
    f["avg_trunk_angle"]   = (f["left_trunk_angle"] + f["right_trunk_angle"]) / 2

    f["left_ankle_angle"]  = compute_angle(lm_xy(lm,25), lm_xy(lm,27), lm_xy(lm,31))
    f["right_ankle_angle"] = compute_angle(lm_xy(lm,26), lm_xy(lm,28), lm_xy(lm,32))
    f["avg_ankle_angle"]   = (f["left_ankle_angle"] + f["right_ankle_angle"]) / 2

    dx, dy = msx - mhx, msy - mhy
    f["spine_lean_angle"]  = float(np.degrees(np.arctan2(abs(dx), abs(dy)+1e-6)))

    f["left_elbow_angle"]  = compute_angle(lm_xy(lm,11), lm_xy(lm,13), lm_xy(lm,15))
    f["right_elbow_angle"] = compute_angle(lm_xy(lm,12), lm_xy(lm,14), lm_xy(lm,16))

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
#  SQUAT POSITION VALIDATOR
# ─────────────────────────────────────────────
def is_squat_position(lm):
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
            return False, "order wrong"
        if mid_sh_y > mid_hip_y - 0.05:
            return False, "sh not above hip"
        vert = abs(mid_hip_y - mid_sh_y) + 1e-6
        if abs(mid_sh_x - mid_hip_x) / vert > 0.60:
            return False, "too horizontal"
        if hasattr(l_an, 'visibility') and l_an.visibility < 0.3 and r_an.visibility < 0.3:
            return False, "ankles hidden"
        if abs(l_kn.x - r_kn.x) < 0.01:
            return False, "partial body"
        return True, "ok"
    except:
        return False, "error"

# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
def load_models(models_dir):
    print("\n[MODELS] Scanning", models_dir)
    if not Path(models_dir).exists():
        print("[MODELS] ERROR: directory not found")
        return {}
    pkl_files = [f for f in Path(models_dir).glob("*.pkl") if f.stem != "best_model"]
    models = {}
    for p in sorted(pkl_files):
        try:
            with open(p, "rb") as f:
                models[p.stem] = pickle.load(f)
            print(f"  [OK]  {p.stem}  ({p.stat().st_size//1024} KB)")
        except Exception as e:
            print(f"  [!!]  {p.stem}  FAILED: {e}")
    print(f"[MODELS] Loaded {len(models)} models")
    return models

def load_feature_names(meta_path):
    if not Path(meta_path).exists():
        print("[META] Not found — will use dict key order")
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        names = meta.get("feature_names")
        if names:
            print(f"[META] {len(names)} feature names loaded")
        return names
    except:
        return None

# ─────────────────────────────────────────────
#  PER-FRAME PREDICTION  (all models, one frame)
# ─────────────────────────────────────────────
def predict_frame(models, feature_row, feature_names):
    """Returns dict {model_name: prob_good}  for one frame."""
    try:
        if feature_names:
            X = pd.DataFrame([feature_row])[feature_names]
        else:
            X = pd.DataFrame([feature_row])
    except Exception as e:
        return {n: 0.5 for n in models}

    out = {}
    for name, pipeline in models.items():
        try:
            classes   = list(pipeline.classes_)
            proba     = pipeline.predict_proba(X)[0]
            prob_good = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
            out[name] = round(prob_good, 4)
        except:
            out[name] = 0.5
    return out

# ─────────────────────────────────────────────
#  DRAW RIGHT PANEL  (bars + timeline graph)
# ─────────────────────────────────────────────
def draw_right_panel(canvas, model_names, model_colors,
                     cur_probs, timeline_history,
                     frame_idx, squat_valid, squat_state,
                     smooth_knee, avg_hip):
    """
    canvas : pre-allocated numpy array for the right panel
    timeline_history : deque of dicts {model_name: prob}  (last TIMELINE_LEN frames)
    """
    h, w = canvas.shape[:2]
    canvas[:] = (22, 22, 22)   # dark background

    # ── Title ──
    cv2.putText(canvas, "PER-FRAME MODEL REACTION", (10, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 200, 255), 1)
    cv2.putText(canvas, f"Frame {frame_idx}", (w - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    cv2.line(canvas, (10, 32), (w-10, 32), (55, 55, 55), 1)

    # ── Squat state badge ──
    state_col = (0, 255, 255) if squat_state == "DOWN" else (160, 160, 160)
    valid_col = (0, 200, 0)   if squat_valid           else (0, 60, 200)
    cv2.putText(canvas, f"State: {squat_state}", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_col, 1)
    cv2.putText(canvas, f"Knee: {smooth_knee:.0f}°  Hip: {avg_hip:.0f}°",
                (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    valid_txt = "SQUAT VALID" if squat_valid else "NOT SQUAT"
    cv2.putText(canvas, valid_txt, (w-130, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, valid_col, 1)
    cv2.line(canvas, (10, 88), (w-10, 88), (55, 55, 55), 1)

    # ── Current-frame confidence bars ──
    bar_top  = 96
    bar_h    = 22
    bar_gap  = 6
    bar_maxw = w - 140

    for i, mname in enumerate(model_names):
        col   = model_colors[i % len(model_colors)]
        prob  = cur_probs.get(mname, 0.5)
        pct   = int(prob * 100)
        y     = bar_top + i * (bar_h + bar_gap)
        label = mname[:14]

        # Model name
        cv2.putText(canvas, label, (10, y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # Bar background
        bx = 130
        cv2.rectangle(canvas, (bx, y), (bx + bar_maxw, y + bar_h), (45, 45, 45), -1)

        # Bar fill  — colour shifts red<0.5, green≥0.5
        fill_col = col if prob >= 0.5 else tuple(int(c * 0.45) for c in col)
        bw = int(bar_maxw * prob)
        cv2.rectangle(canvas, (bx, y), (bx + bw, y + bar_h), fill_col, -1)

        # 50% marker
        mid_x = bx + bar_maxw // 2
        cv2.line(canvas, (mid_x, y), (mid_x, y + bar_h), (100, 100, 100), 1)

        # Percentage text inside bar
        pct_str = f"{pct}%"
        tx = bx + bw + 4 if bw < bar_maxw - 40 else bx + bw - 36
        cv2.putText(canvas, pct_str, (tx, y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (230, 230, 230) if bw > 20 else (180, 180, 180), 1)

        # GOOD/BAD label
        verdict     = "GOOD" if prob >= 0.5 else "BAD"
        verdict_col = (0, 200, 0) if prob >= 0.5 else (0, 60, 220)
        cv2.putText(canvas, verdict, (10, y + bar_h - 5 + bar_h + bar_gap - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, verdict_col, 1)

    # ── Timeline graph ──
    graph_top = bar_top + len(model_names) * (bar_h + bar_gap) + 18
    graph_h   = h - graph_top - 30
    graph_w   = w - 20

    if graph_h < 60:
        return   # not enough vertical space

    # Graph background
    cv2.rectangle(canvas, (10, graph_top), (10 + graph_w, graph_top + graph_h),
                  (30, 30, 30), -1)
    cv2.rectangle(canvas, (10, graph_top), (10 + graph_w, graph_top + graph_h),
                  (70, 70, 70), 1)

    # Grid lines at 25 / 50 / 75%
    for pct_line in [25, 50, 75]:
        gy = graph_top + graph_h - int(graph_h * pct_line / 100)
        cv2.line(canvas, (10, gy), (10 + graph_w, gy), (55, 55, 55), 1)
        cv2.putText(canvas, f"{pct_line}%", (graph_w - 24, gy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)

    # Title
    cv2.putText(canvas, "CONFIDENCE TIMELINE  (last 200 frames)",
                (10, graph_top - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (120, 120, 120), 1)

    # Plot lines
    hist = list(timeline_history)
    n    = len(hist)
    if n >= 2:
        step_x = graph_w / TIMELINE_LEN
        for i, mname in enumerate(model_names):
            col    = model_colors[i % len(model_colors)]
            pts    = []
            offset = TIMELINE_LEN - n   # pad left if fewer than TIMELINE_LEN frames
            for t, snap in enumerate(hist):
                prob = snap.get(mname, 0.5)
                x    = int(10 + (offset + t) * step_x)
                y    = int(graph_top + graph_h - prob * graph_h)
                y    = max(graph_top + 1, min(graph_top + graph_h - 1, y))
                pts.append((x, y))
            for j in range(1, len(pts)):
                cv2.line(canvas, pts[j-1], pts[j], col, 2)

    # Legend
    leg_y = graph_top + graph_h + 18
    x_leg = 10
    for i, mname in enumerate(model_names):
        col   = model_colors[i % len(model_colors)]
        short = mname[:12]
        cv2.rectangle(canvas, (x_leg, leg_y - 8), (x_leg + 14, leg_y + 2), col, -1)
        cv2.putText(canvas, short, (x_leg + 18, leg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
        x_leg += len(short) * 7 + 30
        if x_leg > w - 80:
            break  # avoid overflow

# ─────────────────────────────────────────────
#  DRAW LEFT PANEL  (skeleton + angles + state)
# ─────────────────────────────────────────────
def draw_left_overlay(frame, state, squat_valid, smooth_knee, avg_hip,
                      spine_lean, frame_idx, elapsed):
    h, w = frame.shape[:2]

    # Dark top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, "SQUAT FRAME CLASSIFIER", (8, 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 200, 255), 1)

    state_col = (0, 255, 255) if state == "DOWN" else (160, 255, 160)
    cv2.putText(frame, f"State: {state}", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_col, 1)

    mins = int(elapsed) // 60; secs = int(elapsed) % 60
    cv2.putText(frame, f"{mins:02d}:{secs:02d}  f{frame_idx}", (w - 120, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

    # Angle readouts at bottom-left
    def angle_label(label, val, y_pos, warn_lo=None, warn_hi=None):
        col = (200, 200, 200)
        if warn_lo and val < warn_lo: col = (0, 80, 255)
        if warn_hi and val > warn_hi: col = (0, 80, 255)
        cv2.putText(frame, f"{label}: {val:.0f}°", (8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)

    by = h - 70
    angle_label("Knee",  smooth_knee, by,        warn_lo=60, warn_hi=175)
    angle_label("Hip",   avg_hip,     by + 22)
    angle_label("Spine", spine_lean,  by + 44,   warn_hi=45)

    if not squat_valid:
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 160), -1)
        cv2.putText(frame, "NOT IN SQUAT POSITION", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

# ─────────────────────────────────────────────
#  TERMINAL FRAME LOG
# ─────────────────────────────────────────────
def print_frame_log(frame_log, model_names):
    if not frame_log:
        print("\n[LOG] No frames recorded.")
        return

    print("\n" + "=" * 80)
    print("  PER-FRAME CLASSIFIER LOG")
    print("=" * 80)

    # Header
    hdr = "  FRAME  TIME    KNEE   HIP   STATE  VALID  "
    for m in model_names:
        hdr += m[:9].ljust(10)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in frame_log:
        valid_str = "YES" if r["squat_valid"] else "NO "
        row = (f"  {r['frame']:5d}  {r['time']:5.2f}s  "
               f"{r['knee']:5.1f}  {r['hip']:5.1f}  "
               f"{r['state']:<6} {valid_str}  ")
        for m in model_names:
            p = r["probs"].get(m, 0.5)
            row += f"{p*100:5.1f}%   "
        print(row)

    print("\n" + "=" * 80)
    print("  MODEL AGREEMENT ANALYSIS")
    print("=" * 80)
    print(f"  Total frames logged : {len(frame_log)}")

    for m in model_names:
        probs   = [r["probs"].get(m, 0.5) for r in frame_log]
        avg_p   = np.mean(probs)
        std_p   = np.std(probs)
        pct_gd  = sum(1 for p in probs if p >= 0.5) / len(probs) * 100
        # Find frames where this model disagreed most with ensemble avg
        ens     = [np.mean(list(r["probs"].values())) for r in frame_log]
        diffs   = [abs(p - e) for p, e in zip(probs, ens)]
        max_diff_f = frame_log[int(np.argmax(diffs))]["frame"]
        print(f"\n  {m}")
        print(f"    Avg confidence  : {avg_p*100:.1f}%")
        print(f"    Std deviation   : {std_p*100:.1f}%  (higher = more volatile)")
        print(f"    Frames GOOD     : {pct_gd:.1f}%")
        print(f"    Max vs ensemble : frame {max_diff_f}  ({max(diffs)*100:.1f}% gap)")

    # Frames of highest disagreement between models
    print("\n  TOP 10 FRAMES WITH HIGHEST MODEL DISAGREEMENT")
    print("  " + "-" * 50)
    disagreement = []
    for r in frame_log:
        probs = list(r["probs"].values())
        disagreement.append((np.std(probs), r))
    disagreement.sort(key=lambda x: -x[0])
    for std_val, r in disagreement[:10]:
        probs_str = "  ".join(f"{m[:6]}:{r['probs'].get(m,0.5)*100:.0f}%"
                               for m in model_names)
        print(f"  Frame {r['frame']:5d} ({r['time']:.2f}s)  "
              f"std={std_val*100:.1f}%  knee={r['knee']:.0f}°  |  {probs_str}")

    print("\n" + "=" * 80 + "\n")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print_how_it_works()

    models = load_models(MODELS_DIR)
    if not models:
        print("ABORT: no models loaded.")
        return

    feature_names = load_feature_names(META_JSON)

    model_names  = list(models.keys())
    model_colors = MODEL_COLORS[:len(model_names)]

    # Print legend once
    print("\n  MODEL → COLOUR LEGEND")
    for i, m in enumerate(model_names):
        col = MODEL_COLORS[i % len(MODEL_COLORS)]
        print(f"  {col}  {m}")

    # Open video
    if not Path(INPUT_VIDEO).exists():
        print("ERROR: video not found:", INPUT_VIDEO)
        return
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("ERROR: cannot open video")
        return

    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS          = cap.get(cv2.CAP_PROP_FPS) or 30
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[VIDEO] {W}x{H}  {FPS:.1f}fps  {TOTAL_FRAMES} frames  "
          f"{TOTAL_FRAMES/FPS:.1f}s")

    # Layout: left = video, right = panel
    # Scale video to fit DISPLAY_H
    vid_h = DISPLAY_H
    vid_w = int(W * vid_h / H)
    panel_w = DISPLAY_W - vid_w
    if panel_w < 300:
        vid_w   = DISPLAY_W * 2 // 3
        panel_w = DISPLAY_W - vid_w
        vid_h   = int(H * vid_w / W)

    print(f"[DISPLAY] video pane {vid_w}x{vid_h}  |  right panel {panel_w}x{DISPLAY_H}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # State
    knee_buf    = deque(maxlen=SMOOTHING_WINDOW)
    state       = "UP"
    down_frames = 0
    up_frames   = 0

    timeline_history = deque(maxlen=TIMELINE_LEN)
    frame_log        = []            # full frame-by-frame record for terminal
    right_panel      = np.zeros((DISPLAY_H, panel_w, 3), dtype=np.uint8)

    cur_probs    = {m: 0.5 for m in model_names}
    smooth_k     = 180.0
    avg_hip_val  = 180.0
    spine_lean   = 0.0
    squat_valid  = False

    frame_idx = 0
    log_every = max(1, TOTAL_FRAMES // 10)

    print("\n[RUN] Starting — press Q or Esc to quit early\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx  += 1
        elapsed_vid = frame_idx / FPS

        if frame_idx % log_every == 0:
            print(f"  frame {frame_idx}/{TOTAL_FRAMES}  "
                  f"{frame_idx/TOTAL_FRAMES*100:.0f}%  state={state}")

        # ── Pose ──
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            mp_draw.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(100, 255, 100), thickness=2)
            )

            squat_valid, _ = is_squat_position(lm)

            # Key angles
            lk   = compute_angle(lm_xy(lm,23), lm_xy(lm,25), lm_xy(lm,27))
            rk   = compute_angle(lm_xy(lm,24), lm_xy(lm,26), lm_xy(lm,28))
            knee_buf.append((lk + rk) / 2)
            smooth_k    = float(np.mean(knee_buf))
            lh  = compute_angle(lm_xy(lm,11), lm_xy(lm,23), lm_xy(lm,25))
            rh  = compute_angle(lm_xy(lm,12), lm_xy(lm,24), lm_xy(lm,26))
            avg_hip_val = (lh + rh) / 2
            msx = (lm[11].x + lm[12].x) / 2
            mhx = (lm[23].x + lm[24].x) / 2
            msy = (lm[11].y + lm[12].y) / 2
            mhy = (lm[23].y + lm[24].y) / 2
            spine_lean  = float(np.degrees(np.arctan2(abs(msx-mhx), abs(msy-mhy)+1e-6)))

            # ── PER-FRAME CLASSIFICATION ──
            try:
                feats     = extract_features(lm)
                cur_probs = predict_frame(models, feats, feature_names)
            except Exception as e:
                cur_probs = {m: 0.5 for m in model_names}

            # State machine (for context display only — not rep-gated here)
            if squat_valid:
                if smooth_k < KNEE_DOWN_ANGLE:
                    down_frames += 1; up_frames = 0
                elif smooth_k > KNEE_UP_ANGLE:
                    up_frames += 1; down_frames = 0
                if state == "UP"   and down_frames >= FRAME_CONFIRM:
                    state = "DOWN"; down_frames = 0
                if state == "DOWN" and up_frames   >= FRAME_CONFIRM:
                    state = "UP";   up_frames = 0
            else:
                down_frames = 0; up_frames = 0

        else:
            squat_valid = False
            cur_probs   = {m: 0.5 for m in model_names}

        # ── Record for timeline + log ──
        timeline_history.append(dict(cur_probs))
        frame_log.append({
            "frame":       frame_idx,
            "time":        round(elapsed_vid, 3),
            "knee":        round(smooth_k, 1),
            "hip":         round(avg_hip_val, 1),
            "spine":       round(spine_lean, 1),
            "state":       state,
            "squat_valid": squat_valid,
            "probs":       dict(cur_probs),
        })

        # ── Draw left panel ──
        draw_left_overlay(frame, state, squat_valid,
                          smooth_k, avg_hip_val, spine_lean,
                          frame_idx, elapsed_vid)

        # ── Draw right panel ──
        draw_right_panel(right_panel, model_names, model_colors,
                         cur_probs, timeline_history,
                         frame_idx, squat_valid, state,
                         smooth_k, avg_hip_val)

        # ── Composite & display ──
        vid_frame   = cv2.resize(frame, (vid_w, vid_h))
        panel_frame = cv2.resize(right_panel, (panel_w, DISPLAY_H))

        # Pad vid_frame to DISPLAY_H if needed
        if vid_h < DISPLAY_H:
            pad = np.zeros((DISPLAY_H - vid_h, vid_w, 3), dtype=np.uint8)
            vid_frame = np.vstack([vid_frame, pad])

        composite = np.hstack([vid_frame, panel_frame])
        cv2.imshow(WINDOW_NAME, composite)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print(f"\n[RUN] Quit at frame {frame_idx}")
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    # ── Terminal report ──
    print_frame_log(frame_log, model_names)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("UNHANDLED EXCEPTION:", e)
        traceback.print_exc()
    print("\nDone.")