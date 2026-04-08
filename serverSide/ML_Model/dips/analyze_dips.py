"""
analyze_dips.py  —  Real-time Dip Analyzer  (Top-5 Models)
===========================================================
Plays your dip video in an OpenCV window showing:

  ┌─────────────────────────────────────┬──────────────────┐
  │  DIP ANALYZER      ██████░ 42%  ... │  SPACE=pause Q=  │  ← TOP BAR
  ├─────────────────────────────────────┤                  │
  │  REPS              │                │  gradient_boost  │
  │  ████ 5            │  video +       │  stacking_ensem  │
  │                    │  skeleton      │  adaboost        │
  │  BOTTOM ↓          │                │  catboost        │
  │  L:78°  R:81°      │                │  bagging         │
  │                    │                │──────────────────│
  │                    │                │  VOTE: CORRECT   │
  ├────────────────────┴────────────────┴──────────────────┤
  │  FORM: CORRECT  conf▓▓▓▓▓▓░ 84%  │  Score 92/100 ...  │  ← BOT BAR
  │  FATIGUE: NONE  ▓░░░░░░ 8/100    │  Max est: 14 reps  │
  └────────────────────────────────────────────────────────┘

Top-5 models (by CV accuracy on your data):
  gradient_boosting  CV=0.7515  Test=0.7500
  stacking_ensemble  CV=0.7333  Test=0.7500
  adaboost           CV=0.6970  Test=0.6667
  catboost           CV=0.6803  Test=0.6667
  bagging            CV=0.6803  Test=0.7500

Controls:
  SPACE  — pause / resume
  Q      — quit
  ESC    — quit

Usage:
  python analyze_dips.py --video path/to/video.mp4
  python analyze_dips.py --video path/to/video.MOV --speed 0.5
  python analyze_dips.py --video path/to/video.mp4 --no-models
"""

import os, sys, math, pickle, glob, argparse, warnings, traceback
from collections import Counter, deque

# ── silence noisy libs before importing them ──────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]     = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]    = "1"

# ── force every print to flush immediately so Windows terminal shows it ───────
import builtins, functools
builtins.print = functools.partial(builtins.print, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS  (each step is logged so you can see exactly where a crash happens)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  DIP ANALYZER  (Top-5 Models)  —  starting")
print("=" * 62)

print("[1/6] numpy ...", end=" ")
import numpy as np
print("OK")

print("[2/6] opencv ...", end=" ")
import cv2
print("OK  version:", cv2.__version__)

print("[3/6] mediapipe ...", end=" ")
import mediapipe as mp
print("OK")

print("[4/6] scipy ...", end=" ")
from scipy.ndimage import uniform_filter1d
from scipy.stats   import skew, kurtosis, linregress
print("OK")

print("[5/6] yaml ...", end=" ")
import yaml
print("OK")

print("[6/6] Ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TOP5 = [
    "gradient_boosting",
    "stacking_ensemble",
    "adaboost",
    "catboost",
    "bagging",
]

CLASSES = ["correct", "elbow_flare", "shallow_depth", "forward_lean"]

# Landmark indices
L_SH,R_SH = 11,12;  L_EL,R_EL = 13,14;  L_WR,R_WR = 15,16
L_HI,R_HI = 23,24;  L_KN,R_KN = 25,26;  NOSE = 0

# BGR colour palette
WHITE  = (255,255,255);  BLACK  = (  0,  0,  0);  GRAY   = (140,140,140)
DARK   = ( 22, 22, 22);  GREEN  = (  0,210, 80);  YELLOW = (  0,200,255)
ORANGE = (  0,155,255);  RED    = ( 50, 50,230);  CYAN   = (220,195,  0)
TEAL   = (180,200,  0);  PANEL  = ( 18, 18, 18)

FORM_COL   = {"correct":GREEN,"elbow_flare":YELLOW,
              "shallow_depth":ORANGE,"forward_lean":RED,
              "unknown":GRAY,"error":GRAY}
FAT_COL    = {"none":GREEN,"early":CYAN,"moderate":YELLOW,"severe":RED}
PHASE_COL  = {"top":GREEN,"ascending":TEAL,"descending":YELLOW,"bottom":ORANGE}
FONT_D = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

# Layout sizes (pixels)
TOP_H = 48      # top status bar
BOT_H = 195     # bottom info bar
PNLW  = 290     # right-side model panel
TGT_H = 560     # video display height (width calculated from aspect ratio)

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def put(img, text, xy, scale=0.55, col=WHITE, thick=1, font=FONT_S):
    """Anti-aliased text with automatic black shadow."""
    x,y = int(xy[0]), int(xy[1])
    cv2.putText(img, str(text), (x+1,y+1), font, scale, BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(img, str(text), (x,  y),   font, scale, col,   thick,   cv2.LINE_AA)

def overlay_rect(img, x1,y1,x2,y2, col=DARK, alpha=0.75, border=None, bthick=1):
    """Semi-transparent filled rectangle."""
    ov = img.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
    if border:
        cv2.rectangle(img,(x1,y1),(x2,y2),border,bthick)

def progress_bar(img, x,y,w,h, val,maxv, fg, bg=(45,45,45)):
    """Horizontal progress bar."""
    cv2.rectangle(img,(x,y),(x+w,y+h),bg,-1)
    filled = int(min(max(val,0)/max(maxv,1),1.0)*w)
    if filled > 0:
        cv2.rectangle(img,(x,y),(x+filled,y+h),fg,-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),GRAY,1)

def deg_str(v):
    return "{:.0f}".format(v) + chr(176)   # e.g. "87°"

def score_col(s):
    return GREEN if s >= 80 else YELLOW if s >= 50 else RED

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def angle3(a, b, c):
    ba = a-b;  bc = c-b
    cos = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
    return math.degrees(math.acos(np.clip(cos,-1.0,1.0)))

def lm2xyz(lm):
    return np.array([lm.x, lm.y, lm.z], np.float32)

def get_elbow_angles(lms):
    try:
        l = angle3(lm2xyz(lms[L_SH]), lm2xyz(lms[L_EL]), lm2xyz(lms[L_WR]))
        r = angle3(lm2xyz(lms[R_SH]), lm2xyz(lms[R_EL]), lm2xyz(lms[R_WR]))
        return l, r
    except Exception:
        return None, None

def lms_to_kp(lms):
    kp = np.zeros(99, np.float32)
    for j,lm in enumerate(lms):
        kp[j*3], kp[j*3+1], kp[j*3+2] = lm.x, lm.y, lm.z
    return kp

def compute_6angles(kp_seq):
    """(T,99) → (T,6) joint angles."""
    T  = len(kp_seq)
    kp = np.array(kp_seq, np.float32)
    A  = np.zeros((T,6), np.float32)
    for t in range(T):
        def v(i): return kp[t, i*3:i*3+3]
        ls,rs = v(L_SH),v(R_SH);  le,re = v(L_EL),v(R_EL)
        lw,rw = v(L_WR),v(R_WR);  lh,rh = v(L_HI),v(R_HI)
        lk,rk = v(L_KN),v(R_KN);  ns    = v(NOSE)
        mh = (lh+rh)/2;  ms = (ls+rs)/2
        A[t,0]=angle3(ls,le,lw);  A[t,1]=angle3(rs,re,rw)
        A[t,2]=angle3(le,ls,lh);  A[t,3]=angle3(re,rs,rh)
        A[t,4]=angle3(ns,ms,mh);  A[t,5]=angle3(ms,mh,(lk+rk)/2)
    return A

def make_features(kp_seq):
    """(T,99) keypoints → 1-D feature vector (~1163 dims)."""
    kp_arr = np.array(kp_seq, np.float32)
    ang    = compute_6angles(kp_arr)
    def stat(s):
        v = np.diff(s, axis=0)
        return np.concatenate([
            s.mean(0), s.std(0), s.min(0), s.max(0),
            s.max(0)-s.min(0), np.median(s,0),
            np.percentile(s,75,0)-np.percentile(s,25,0),
            skew(s,0), kurtosis(s,0), v.mean(0), v.std(0),
        ]).astype(np.float32)
    ea  = (ang[:,0]+ang[:,1]) / 2
    sm  = uniform_filter1d(ea, size=3)
    sym = float(np.mean(np.abs(ang[:,0]-ang[:,1])))
    bio = np.array([
        sm.min(), sm.max(), sm.max()-sm.min(),
        float(np.mean(ang[:,2:4])), float(np.mean(ang[:,4])),
        sym,
        float(np.sum(np.diff(np.sign(np.diff(sm))) < 0)),
        float(np.std(sm)),
    ], np.float32)
    return np.concatenate([stat(kp_arr), stat(ang), bio])

# ─────────────────────────────────────────────────────────────────────────────
# REP / PHASE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class RepDetector:
    def __init__(self):
        self.buf      = deque(maxlen=9)
        self.phase    = "top"
        self.reps     = 0
        self.in_bot   = False
        self.prev     = None
        self.depths   = []    # min elbow angle per bottom
        self.rtimes   = []    # frame index of each bottom

    def update(self, avg_elbow: float, frame_idx: int) -> str:
        self.buf.append(avg_elbow)
        if len(self.buf) < 3:
            return self.phase

        sm = float(np.mean(self.buf))
        BOT = 85    # arms fully bent  → bottom of dip
        TOP = 150   # arms extended    → top of dip

        if sm <= BOT:
            if not self.in_bot:
                self.in_bot = True
                self.reps  += 1
                self.depths.append(sm)
                self.rtimes.append(frame_idx)
            self.phase = "bottom"
        elif sm >= TOP:
            self.in_bot = False
            self.phase  = "top"
        else:
            self.in_bot = False
            if self.prev is not None:
                self.phase = "descending" if sm < self.prev else "ascending"
        self.prev = sm
        return self.phase

# ─────────────────────────────────────────────────────────────────────────────
# FATIGUE
# ─────────────────────────────────────────────────────────────────────────────

def calc_fatigue(depths, rtimes, fps):
    n = len(depths)
    if n < 2:
        return "none", 0.0, max(n * 4, 8)

    signals = []

    # depth degradation
    slope, *_ = linregress(range(n), depths)
    depth_rise = (slope * n / (np.mean(depths) + 1e-6)) * 100
    signals.append(min(max(depth_rise, 0) / 30.0, 1.0) * 40)

    # tempo slowdown
    if len(rtimes) >= 3:
        durs = np.diff(rtimes) / fps
        s2, *_ = linregress(range(len(durs)), durs)
        tempo_inc = (s2 * len(durs) / (np.mean(durs) + 1e-6)) * 100
        signals.append(min(max(tempo_inc, 0) / 40.0, 1.0) * 30)

    score = min(100.0, sum(signals))
    level = ("none"     if score < 10 else
             "early"    if score < 30 else
             "moderate" if score < 55 else "severe")

    if slope > 0 and n >= 2:
        max_est = n + max(0, int((90 - depths[-1]) / (slope + 1e-6)))
    else:
        mult = {"none": 2.5, "early": 1.6, "moderate": 1.2, "severe": 1.0}[level]
        max_est = int(n * mult)

    return level, score, max(max_est, n)

# ─────────────────────────────────────────────────────────────────────────────
# FORM SCORE
# ─────────────────────────────────────────────────────────────────────────────

def calc_form_score(kp_seq, thr):
    ang = compute_6angles(kp_seq)
    iss, ded = {}, 0
    if float(np.mean(ang[:,2:4]))              > thr["elbow_flare_angle"]: iss["Elbow Flare"]=25;  ded+=25
    if float(np.min((ang[:,0]+ang[:,1])/2))    > thr["min_depth_angle"]:  iss["Shallow Depth"]=25; ded+=25
    if float(np.mean(ang[:,4]))                > thr["torso_lean_angle"]: iss["Forward Lean"]=20;  ded+=20
    if float(np.mean(np.abs(ang[:,0]-ang[:,1])))> 15:                     iss["Asymmetry"]=15;     ded+=15
    return max(0, 100 - ded), iss

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING & INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_top5(ckpt_dir):
    """Load only the top-5 .pkl files."""
    models = {}
    print("\nLoading top-5 models from:", ckpt_dir)
    for name in TOP5:
        path = os.path.join(ckpt_dir, name + ".pkl")
        if not os.path.exists(path):
            print("  [MISSING]", name, "— run pipeline.py first")
            continue
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            print("  [OK]", name)
        except Exception as e:
            print("  [ERROR]", name, "->", e)
    print("  Loaded:", len(models), "/ 5 models\n")
    return models

def load_preprocessor(proc_dir):
    path = os.path.join(proc_dir, "preprocessor.pkl")
    if not os.path.exists(path):
        print("[WARN] preprocessor.pkl not found in", proc_dir,
              "— raw features will be used")
        return None
    with open(path, "rb") as f:
        pp = pickle.load(f)
    print("[OK] preprocessor loaded")
    return pp

def run_inference(X, models):
    """X shape: (1, feature_dim)  →  list of result dicts."""
    results = []
    for name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                pidx  = int(np.argmax(probs))
                conf  = float(probs[pidx]) * 100
            else:
                pidx  = int(model.predict(X)[0])
                probs = np.zeros(len(CLASSES)); probs[pidx] = 1.0
                conf  = 100.0
            pred = CLASSES[pidx] if pidx < len(CLASSES) else "unknown"
            results.append({"name": name, "pred": pred,
                             "conf": conf, "probs": probs})
        except Exception as e:
            results.append({"name": name, "pred": "error",
                             "conf": 0.0, "probs": np.zeros(len(CLASSES))})
    return results

def majority_vote(results):
    valid = [r for r in results if r["pred"] not in ("error", "unknown")]
    if not valid:
        return "unknown", 0.0
    counts = Counter(r["pred"] for r in valid)
    top    = counts.most_common(1)[0][0]
    avg_c  = float(np.mean([r["conf"] for r in valid if r["pred"] == top]))
    return top, avg_c

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg():
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    # fallback defaults
    return {
        "preprocessing": {"max_frames": 30},
        "training":      {"checkpoint_dir": "outputs/checkpoints"},
        "data":          {"processed_dir":  "data/processed"},
        "thresholds": {
            "elbow_flare_angle": 45,
            "min_depth_angle":   90,
            "torso_lean_angle":  20,
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# BUILD DISPLAY CANVAS
# ─────────────────────────────────────────────────────────────────────────────

def build_canvas(vid_frame,
                 fidx, total_frames,
                 phase, el, er,
                 reps,
                 form_cls, form_conf, form_score, issues,
                 fat_level, fat_score, max_reps,
                 model_results, paused):

    VH, VW = vid_frame.shape[:2]
    FW = VW + PNLW
    FH = TOP_H + VH + BOT_H

    canvas = np.full((FH, FW, 3), 12, np.uint8)

    # ── paste video frame ─────────────────────────────────────────────────────
    canvas[TOP_H:TOP_H+VH, 0:VW] = vid_frame

    # ═══ TOP BAR ═════════════════════════════════════════════════════════════
    overlay_rect(canvas, 0,0, FW,TOP_H, DARK, 0.9, (0,180,180))
    put(canvas, "DIP ANALYZER", (10,33), scale=0.9, col=CYAN, thick=2, font=FONT_D)

    # progress bar
    prog = fidx / max(total_frames-1, 1)
    bx,by,bw,bh = VW//2-100, 15, 200, 10
    progress_bar(canvas, bx,by,bw,bh, fidx,total_frames, CYAN)
    put(canvas, "{:.0f}%".format(prog*100), (bx+bw+6,by+bh), scale=0.38, col=GRAY)
    put(canvas, "SPACE=pause  Q=quit", (FW-225,32), scale=0.42, col=GRAY)

    # ═══ VIDEO-AREA OVERLAYS (positioned relative to video start = TOP_H) ════
    V0 = TOP_H   # y-offset where video starts

    # -- Rep counter box
    overlay_rect(canvas, 8,V0+6, 128,V0+102, DARK, 0.80, (0,180,180))
    put(canvas, "REPS", (18,V0+27), scale=0.50, col=GRAY)
    rs = str(reps)
    cv2.putText(canvas, rs, (16,V0+90), FONT_D, 2.6, BLACK, 10, cv2.LINE_AA)
    cv2.putText(canvas, rs, (16,V0+90), FONT_D, 2.6, CYAN,   4, cv2.LINE_AA)

    # -- Phase box
    pc = PHASE_COL.get(phase, WHITE)
    overlay_rect(canvas, 8,V0+107, 215,V0+144, DARK, 0.80, pc)
    put(canvas, phase.upper(), (16,V0+134), scale=0.76, col=pc, thick=2, font=FONT_D)

    # -- Elbow angle box
    overlay_rect(canvas, 8,V0+149, 215,V0+205, DARK, 0.75)
    def ac(a): return GREEN if a and a<90 else YELLOW if a and a<130 else RED
    put(canvas, "L: " + (deg_str(el) if el else "--"), (14,V0+169), scale=0.57, col=ac(el))
    put(canvas, "R: " + (deg_str(er) if er else "--"), (14,V0+191), scale=0.57, col=ac(er))

    # ═══ RIGHT PANEL ══════════════════════════════════════════════════════════
    PX = VW
    overlay_rect(canvas, PX,0, FW,FH, (14,14,14), 0.96, (0,160,160))

    put(canvas, "TOP-5 MODELS", (PX+8,TOP_H+22), scale=0.56, col=CYAN)
    put(canvas, str(len(model_results))+" loaded", (PX+8,TOP_H+40),
        scale=0.36, col=GRAY)

    ROW_H  = 46
    avail  = (VH - 55) // ROW_H

    for k, r in enumerate(model_results[:avail]):
        ry  = TOP_H + 54 + k * ROW_H
        mc  = FORM_COL.get(r["pred"], GRAY)
        overlay_rect(canvas, PX+4,ry-2, FW-4,ry+ROW_H-4, (32,32,32), 0.65)

        # model name (abbreviated)
        nm_display = r["name"].replace("_", " ")[:19]
        put(canvas, nm_display, (PX+8,ry+14), scale=0.38, col=WHITE)

        # prediction label
        pred_display = r["pred"].replace("_"," ")
        put(canvas, pred_display, (PX+8,ry+29), scale=0.44, col=mc)

        # confidence mini-bar
        progress_bar(canvas, PX+8,ry+33, PNLW-22,6, r["conf"],100, mc)

        # confidence %
        put(canvas, "{:.0f}%".format(r["conf"]), (FW-38,ry+29),
            scale=0.36, col=GRAY)

    # -- vote strip at bottom of panel
    if model_results:
        maj, mconf = majority_vote(model_results)
        vote_y = TOP_H + VH - 2
        overlay_rect(canvas, PX+4,vote_y-44, FW-4,vote_y,
                     (38,38,38), 0.88, (0,180,180))
        vcol = FORM_COL.get(maj, WHITE)
        put(canvas, "VOTE: " + maj.replace("_"," ").upper(),
            (PX+8,vote_y-24), scale=0.46, col=vcol)
        put(canvas, "{:.0f}% confidence".format(mconf),
            (PX+8,vote_y-7), scale=0.37, col=GRAY)

    # ═══ BOTTOM BAR ═══════════════════════════════════════════════════════════
    BY = TOP_H + VH
    overlay_rect(canvas, 0,BY, FW,FH, DARK, 0.92, (0,160,160))

    # Three columns
    C1 = 12
    C2 = VW // 3 + 8
    C3 = 2 * VW // 3 + 8
    r1 = BY+26;  r2 = BY+54;  r3 = BY+80
    r4 = BY+104; r5 = BY+132; r6 = BY+162

    # ── Column 1: Form classification ──────────────────────────────────────
    put(canvas, "FORM CLASSIFICATION", (C1,r1), scale=0.44, col=GRAY)
    fc = FORM_COL.get(form_cls, WHITE)
    put(canvas, form_cls.replace("_"," ").upper(), (C1,r2),
        scale=0.84, col=fc, thick=2, font=FONT_D)
    put(canvas, "Confidence", (C1,r3), scale=0.40, col=GRAY)
    progress_bar(canvas, C1,r3+5, 175,11, form_conf,100, fc)
    put(canvas, "{:.1f}%".format(form_conf), (C1+180,r3+13),
        scale=0.39, col=WHITE)
    if issues:
        put(canvas, "Issues:", (C1,r4+2), scale=0.42, col=YELLOW)
        iy = r4 + 20
        for iss in list(issues.keys())[:3]:
            put(canvas, "  * " + iss, (C1,iy), scale=0.39, col=ORANGE)
            iy += 17
    else:
        put(canvas, "No issues — clean form!", (C1,r4+2),
            scale=0.43, col=GREEN)

    # ── Column 2: Form score + max reps ────────────────────────────────────
    put(canvas, "FORM SCORE", (C2,r1), scale=0.44, col=GRAY)
    sc = score_col(form_score)
    ss = str(form_score) + "/100"
    cv2.putText(canvas, ss, (C2,r2+8), FONT_D, 1.05, BLACK, 7, cv2.LINE_AA)
    cv2.putText(canvas, ss, (C2,r2+8), FONT_D, 1.05, sc,    2, cv2.LINE_AA)
    progress_bar(canvas, C2,r3+5, 175,11, form_score,100, sc)

    put(canvas, "Est. Max Reps", (C2,r4+4), scale=0.44, col=GRAY)
    ms = str(max_reps)
    cv2.putText(canvas, ms, (C2,r5+8), FONT_D, 1.2, BLACK, 7, cv2.LINE_AA)
    cv2.putText(canvas, ms, (C2,r5+8), FONT_D, 1.2, CYAN,  2, cv2.LINE_AA)
    done_pct = min(reps / max(max_reps,1) * 100, 100)
    progress_bar(canvas, C2,r5+14, 175,10, done_pct,100, CYAN)
    put(canvas, "done", (C2+180,r5+22), scale=0.35, col=GRAY)

    # ── Column 3: Fatigue ───────────────────────────────────────────────────
    put(canvas, "FATIGUE", (C3,r1), scale=0.44, col=GRAY)
    flc = FAT_COL.get(fat_level, GRAY)
    cv2.putText(canvas, fat_level.upper(), (C3,r2+4), FONT_D, 0.95, BLACK, 6, cv2.LINE_AA)
    cv2.putText(canvas, fat_level.upper(), (C3,r2+4), FONT_D, 0.95, flc,   2, cv2.LINE_AA)
    put(canvas, "Score", (C3,r3), scale=0.40, col=GRAY)
    progress_bar(canvas, C3,r3+5, 175,11, fat_score,100, flc)
    put(canvas, "{:.1f}/100".format(fat_score), (C3,r4+2), scale=0.46, col=flc)
    tip = {
        "none":     "Fresh — keep pushing!",
        "early":    "Early signs — maintain form",
        "moderate": "Tiring — 2-3 reps left",
        "severe":   "Near limit — stop safely",
    }[fat_level]
    put(canvas, tip, (C3,r5), scale=0.40, col=GRAY)

    # ── PAUSED banner ─────────────────────────────────────────────────────────
    if paused:
        pw, ph = 160, 40
        px2 = VW//2 - pw//2
        py2 = V0 + VH//2 - ph//2
        overlay_rect(canvas, px2,py2, px2+pw,py2+ph, DARK, 0.85, YELLOW)
        put(canvas, "PAUSED", (px2+28,py2+27), scale=0.9,
            col=YELLOW, thick=2, font=FONT_D)

    return canvas

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(video_path, reps, form_cls, form_conf,
                  form_score, issues, fat_level, fat_score,
                  max_reps, model_results):
    W = 65
    print("\n" + "=" * W)
    print("  FINAL ANALYSIS SUMMARY")
    print("=" * W)
    print("  Video          :", os.path.basename(video_path))
    print("-" * W)
    print("  Reps completed :", reps)
    print("  Estimated max  :", max_reps)
    print("-" * W)
    maj, mc = majority_vote(model_results) if model_results else ("unknown", 0.0)
    print("  Form class     :", maj.upper().replace("_"," "),
          "  ({:.1f}% avg conf)".format(mc))
    print("  Form score     :", form_score, "/ 100")
    if issues:
        print("  Issues:")
        for k in issues: print("    *", k)
    else:
        print("  Issues         : None — great technique!")
    print("-" * W)
    print("  Fatigue level  :", fat_level.upper())
    print("  Fatigue score  :", "{:.1f}".format(fat_score), "/ 100")
    print("-" * W)
    if model_results:
        print("  TOP-5 MODEL PREDICTIONS:")
        print("  {:<26} {:<18} {:>9}".format("Model","Prediction","Confidence"))
        print("  " + "-" * 55)
        for r in sorted(model_results, key=lambda x: -x["conf"]):
            mark = "  ◄" if r["pred"] == maj else ""
            print("  {:<26} {:<18} {:>8.1f}%{}".format(
                r["name"][:25], r["pred"][:17], r["conf"], mark))
    print("=" * W + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# ── DEFAULT VIDEO PATH — change this to your video file ──────────────────────
DEFAULT_VIDEO = r"C:\Users\punav shigwan\OneDrive\Desktop\all_exersice\tricep dips\tricep dips_20.mp4"

def main():
    ap = argparse.ArgumentParser(description="Real-time dip analysis (top-5 models).")
    ap.add_argument("--video",     default=DEFAULT_VIDEO,
                    help="Path to video file (.mp4 / .mov / .avi)  "
                         "[default: hardcoded DEFAULT_VIDEO at top of script]")
    ap.add_argument("--speed",     type=float, default=1.0,
                    help="Playback speed multiplier (default 1.0)")
    ap.add_argument("--no-models", action="store_true",
                    help="Skip ML inference — skeleton + reps only")
    args = ap.parse_args()

    # Use DEFAULT_VIDEO if --video was not passed on the command line
    video_path = args.video if args.video else DEFAULT_VIDEO

    if not os.path.exists(video_path):
        print("ERROR: video not found:", video_path)
        print("  Edit DEFAULT_VIDEO at the top of analyze_dips.py")
        sys.exit(1)
    args.video = video_path

    cfg  = load_cfg()
    thr  = cfg.get("thresholds", {"elbow_flare_angle":45,
                                   "min_depth_angle":90,
                                   "torso_lean_angle":20})

    # ── Load top-5 models ─────────────────────────────────────────────────────
    models = {}
    pp     = None
    if not args.no_models:
        models = load_top5(cfg["training"]["checkpoint_dir"])
        pp     = load_preprocessor(cfg["data"]["processed_dir"])

    # ── Open video ────────────────────────────────────────────────────────────
    print("Opening video:", args.video)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.video)   # retry once
    if not cap.isOpened():
        print("ERROR: OpenCV cannot open this file.")
        print("  If it is a .MOV file try: pip install opencv-contrib-python")
        sys.exit(1)

    vid_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[VIDEO] {w}x{h}  {fps:.1f}fps  {n} frames".format(
        w=raw_w, h=raw_h, fps=vid_fps, n=total_frm))

    # ── Calculate display sizes ───────────────────────────────────────────────
    vid_h = TGT_H
    vid_w = int(raw_w * vid_h / max(raw_h, 1))
    win_w = vid_w + PNLW
    win_h = TOP_H + vid_h + BOT_H
    print("[WINDOW] {ww}x{wh}  (video {vw}x{vh})".format(
        ww=win_w, wh=win_h, vw=vid_w, vh=vid_h))

    # ── MediaPipe Pose ────────────────────────────────────────────────────────
    print("[POSE] Initialising MediaPipe Pose...")
    pose = mp.solutions.pose.Pose(
        model_complexity         = 1,
        smooth_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )
    print("[POSE] Ready.")

    # ── Window ────────────────────────────────────────────────────────────────
    cv2.namedWindow("Dip Analyzer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dip Analyzer", win_w, win_h)
    # Force window to actually render on Windows before the loop starts
    blank = np.full((win_h, win_w, 3), 20, np.uint8)
    put(blank, "Loading... please wait", (win_w//2 - 140, win_h//2),
        scale=0.9, col=CYAN, thick=2, font=FONT_D)
    cv2.imshow("Dip Analyzer", blank)
    cv2.waitKey(1)
    print("[WINDOW] Window opened successfully.")

    # ── State variables ───────────────────────────────────────────────────────
    detector   = RepDetector()
    kp_buf     = deque(maxlen=30)
    mresults   = []
    form_cls   = "unknown"
    form_conf  = 0.0
    form_score = 100
    issues     = {}
    fat_level  = "none"
    fat_score  = 0.0
    max_reps   = 8
    paused     = False
    fidx       = 0
    frame      = None
    el = er    = None

    INFER_INT  = max(1, int(vid_fps * 1.5))  # run inference every 1.5 s
    last_infer = -INFER_INT
    delay_ms   = max(1, int(1000.0 / (vid_fps * max(args.speed, 0.05))))

    # ── Read first frame (verify codec works) ─────────────────────────────────
    print("[READ] Reading first frame...", end=" ")
    ret, raw = cap.read()
    if not ret:
        print("FAILED")
        print("ERROR: Cannot read video frames.")
        print("  Try: pip install opencv-contrib-python")
        cap.release(); sys.exit(1)
    print("OK")
    print("\n[RUNNING] Window should appear now.")
    print("  SPACE = pause/resume     Q / ESC = quit\n")

    # ═════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═════════════════════════════════════════════════════════════════════════
    while True:

        # ── advance frame when not paused ────────────────────────────────────
        if not paused:
            if not ret:
                print("[END] Video finished.")
                break
            fidx += 1

            # resize to display size
            frame = cv2.resize(raw, (vid_w, vid_h))

            # ── MediaPipe ────────────────────────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            el = er = None

            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark

                # draw skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0,200,90), thickness=2, circle_radius=3),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0,130,255), thickness=2),
                )

                # elbow angles + annotate on frame
                el, er = get_elbow_angles(lms)
                for side_el, side_sh, side_wr in [
                    (L_EL, L_SH, L_WR),
                    (R_EL, R_SH, R_WR),
                ]:
                    try:
                        a  = angle3(lm2xyz(lms[side_sh]),
                                    lm2xyz(lms[side_el]),
                                    lm2xyz(lms[side_wr]))
                        ex = int(lms[side_el].x * vid_w)
                        ey = int(lms[side_el].y * vid_h)
                        ac = GREEN if a<90 else YELLOW if a<130 else RED
                        cv2.circle(frame, (ex,ey), 7, ac, -1)
                        put(frame, deg_str(a), (ex+9,ey-9), scale=0.5, col=ac)
                    except Exception:
                        pass

                # rep detection
                if el and er:
                    detector.update((el+er)/2, fidx)

                # accumulate keypoints
                kp_buf.append(lms_to_kp(lms))

                # fatigue update after each rep
                if len(detector.depths) >= 2:
                    fat_level, fat_score, max_reps = calc_fatigue(
                        detector.depths, detector.rtimes, vid_fps)

                # ML inference every INFER_INT frames
                if models and fidx - last_infer >= INFER_INT and len(kp_buf) >= 10:
                    last_infer = fidx
                    try:
                        seq = list(kp_buf)
                        if len(seq) < 30:
                            seq = [seq[0]] * (30 - len(seq)) + seq
                        seq = np.array(seq[-30:], np.float32)

                        feat = make_features(seq)
                        X    = feat[np.newaxis]        # (1, 1163)

                        if pp is not None:
                            try:
                                X = pp.transform(X)
                            except Exception:
                                pass  # use raw if pp fails

                        mresults   = run_inference(X, models)
                        form_cls, form_conf = majority_vote(mresults)
                        form_score, issues  = calc_form_score(seq, thr)
                    except Exception:
                        pass   # keep last good results

            # read next frame
            ret, raw = cap.read()

        # ── build + display canvas ────────────────────────────────────────────
        if frame is not None:
            canvas = build_canvas(
                vid_frame  = frame,
                fidx       = fidx,
                total_frames = total_frm,
                phase      = detector.phase,
                el         = el,
                er         = er,
                reps       = detector.reps,
                form_cls   = form_cls,
                form_conf  = form_conf,
                form_score = form_score,
                issues     = issues,
                fat_level  = fat_level,
                fat_score  = fat_score,
                max_reps   = max_reps,
                model_results = mresults,
                paused     = paused,
            )
            cv2.imshow("Dip Analyzer", canvas)

        # ── keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("[QUIT] Exiting.")
            break
        elif key == 32:    # SPACE
            paused = not paused
            print("[PAUSE]" if paused else "[RESUME]")

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    # ── terminal summary ──────────────────────────────────────────────────────
    print_summary(
        video_path = args.video,
        reps       = detector.reps,
        form_cls   = form_cls,
        form_conf  = form_conf,
        form_score = form_score,
        issues     = issues,
        fat_level  = fat_level,
        fat_score  = fat_score,
        max_reps   = max_reps,
        model_results = mresults,
    )


if __name__ == "__main__":
    main()