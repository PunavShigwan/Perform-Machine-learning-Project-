"""
dip_service.py
==============
Tricep-dip video analysis using MediaPipe Pose only — no ML model required.

Every metric is derived from joint angles and landmark positions:

Per-rep
-------
  - rep_number
  - start / bottom / end timestamps (seconds)
  - duration (seconds)
  - min_elbow_angle          — depth of the dip (lower = deeper)
  - elbow_symmetry           — |left_angle - right_angle| at bottom
  - torso_lean_angle         — forward lean at the bottom
  - form_score  0-100        — geometry deductions
  - issues                   — list of detected problems
  - advice                   — actionable coaching cue
  - fatigue_at_rep           — none / early / moderate / severe

Overall
-------
  - total_reps / estimated_max_reps
  - overall form score (avg across reps)
  - recurring issues
  - fatigue level + score
  - avg / min / max rep duration
  - depth consistency (std-dev of min elbow angles)
  - one-paragraph plain-English summary
"""

import os
import math
import subprocess
import warnings
from collections import deque
from dataclasses  import dataclass, field
from typing       import List, Optional

import numpy as np
import cv2
import mediapipe as mp

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

try:
    from scipy.stats import linregress
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    subprocess.run(["ffmpeg", "-version"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    _HAS_FFMPEG = True
except Exception:
    _HAS_FFMPEG = False

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# ─────────────────────────────────────────────────────────────────────────────
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HI, R_HI = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28
NOSE       = 0

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (all in degrees unless noted)
# ─────────────────────────────────────────────────────────────────────────────
THR = {
    "elbow_flare_angle":    50,   # shoulder-abduction angle at bottom
    "min_depth_angle":      95,   # min elbow angle still counts as "shallow"
    "torso_lean_angle":     30,   # forward lean  (nose–mid_shoulder–mid_hip)
    "asymmetry_angle":      15,   # |L_elbow - R_elbow| at bottom
    "wrist_flare_angle":    40,   # wrist deviation from neutral
    "rep_bot":              90,   # elbow angle → bottom of dip
    "rep_top":             155,   # elbow angle → top of dip (rep complete)
}

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255);  BLACK  = (  0,   0,   0)
GREEN  = (  0, 210,  80);  YELLOW = (  0, 200, 255)
ORANGE = (  0, 155, 255);  RED    = ( 50,  50, 230)
CYAN   = (220, 195,   0);  GRAY   = (140, 140, 140)
DARK   = ( 22,  22,  22);  TEAL   = (180, 200,   0)

FORM_COL = {
    "correct":       GREEN,
    "elbow_flare":   YELLOW,
    "shallow_depth": ORANGE,
    "forward_lean":  RED,
    "asymmetric":    TEAL,
    "unknown":       GRAY,
}
FAT_COL = {"none": GREEN, "early": CYAN, "moderate": YELLOW, "severe": RED}
FONT_D  = cv2.FONT_HERSHEY_DUPLEX
FONT_S  = cv2.FONT_HERSHEY_SIMPLEX

_FAT_TIP = {
    "none":     "You look fresh — great endurance.",
    "early":    "Early fatigue signs — focus on maintaining form.",
    "moderate": "Moderate fatigue — about 2–3 reps left.",
    "severe":   "Near your limit — stop safely.",
}
_ISSUE_ADVICE = {
    "Elbow Flare":   "Tuck your elbows — keep them at ~45° from your torso.",
    "Shallow Depth": "Lower until upper arms are parallel to the floor.",
    "Forward Lean":  "Stay upright — excessive lean shifts load to your chest.",
    "Asymmetry":     "Both arms should bend evenly — check grip width.",
    "Wrist Deviation": "Keep wrists neutral — avoid bending them back.",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    for p in ["config.yaml", "app/config.yaml"]:
        if _HAS_YAML and os.path.exists(p):
            with open(p) as f:
                cfg = yaml.safe_load(f)
                if "thresholds" in cfg:
                    THR.update(cfg["thresholds"])
                return cfg
    return {}

_load_cfg()

# ─────────────────────────────────────────────────────────────────────────────
# PER-REP DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RepRecord:
    rep_number:          int
    start_frame:         int   = 0
    bottom_frame:        int   = 0
    end_frame:           int   = 0
    start_time_sec:      float = 0.0
    bottom_time_sec:     float = 0.0
    end_time_sec:        float = 0.0
    duration_sec:        float = 0.0

    # depth
    min_elbow_angle:     float = 999.0  # lowest reached (smaller = deeper)
    avg_elbow_at_bottom: float = 0.0

    # geometry at bottom
    elbow_symmetry:      float = 0.0   # |L - R| angle at deepest point
    torso_lean_angle:    float = 0.0   # forward lean at deepest point
    shoulder_abduction:  float = 0.0   # avg shoulder-abduction angle at bottom
    wrist_deviation:     float = 0.0   # avg wrist deviation at bottom

    # tempo
    descent_sec:         float = 0.0   # start → bottom
    ascent_sec:          float = 0.0   # bottom → end

    # scoring
    form_score:          int         = 100
    issues:              List[str]   = field(default_factory=list)
    advice:              str         = ""
    fatigue_at_rep:      str         = "none"

    # raw per-frame measurements collected during this rep
    _elbow_angles:       List[float] = field(default_factory=list, repr=False)
    _torso_angles:       List[float] = field(default_factory=list, repr=False)
    _shoulder_abds:      List[float] = field(default_factory=list, repr=False)
    _wrist_devs:         List[float] = field(default_factory=list, repr=False)
    _sym_angles:         List[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "rep_number":          self.rep_number,
            "start_time_sec":      round(self.start_time_sec,  2),
            "bottom_time_sec":     round(self.bottom_time_sec, 2),
            "end_time_sec":        round(self.end_time_sec,    2),
            "duration_sec":        round(self.duration_sec,    2),
            "descent_sec":         round(self.descent_sec,     2),
            "ascent_sec":          round(self.ascent_sec,      2),
            "min_elbow_angle":     round(self.min_elbow_angle, 1),
            "elbow_symmetry":      round(self.elbow_symmetry,  1),
            "torso_lean_angle":    round(self.torso_lean_angle,1),
            "shoulder_abduction":  round(self.shoulder_abduction, 1),
            "wrist_deviation":     round(self.wrist_deviation, 1),
            "form_score":          self.form_score,
            "issues":              self.issues,
            "advice":              self.advice,
            "fatigue_at_rep":      self.fatigue_at_rep,
        }

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _v(lms, idx) -> np.ndarray:
    lm = lms[idx]
    return np.array([lm.x, lm.y, lm.z], np.float32)

def _angle3(a, b, c) -> float:
    ba  = a - b;  bc = c - b
    n   = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos = np.dot(ba, bc) / n
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))

def _elbow_angle(lms, side: str) -> float:
    """Elbow flexion angle for 'left' or 'right'."""
    if side == "left":
        return _angle3(_v(lms, L_SH), _v(lms, L_EL), _v(lms, L_WR))
    return _angle3(_v(lms, R_SH), _v(lms, R_EL), _v(lms, R_WR))

def _shoulder_abduction(lms, side: str) -> float:
    """Angle between upper arm and torso (shoulder abduction)."""
    if side == "left":
        return _angle3(_v(lms, L_EL), _v(lms, L_SH), _v(lms, L_HI))
    return _angle3(_v(lms, R_EL), _v(lms, R_SH), _v(lms, R_HI))

def _torso_lean(lms) -> float:
    """Forward lean: angle at mid-shoulder between nose and mid-hip."""
    ms = (_v(lms, L_SH) + _v(lms, R_SH)) / 2
    mh = (_v(lms, L_HI) + _v(lms, R_HI)) / 2
    return _angle3(_v(lms, NOSE), ms, mh)

def _wrist_deviation(lms, side: str) -> float:
    """Angle at wrist between elbow–wrist and wrist–finger direction (approx)."""
    # Use elbow → wrist → hip as a proxy for wrist bend
    if side == "left":
        return _angle3(_v(lms, L_EL), _v(lms, L_WR), _v(lms, L_HI))
    return _angle3(_v(lms, R_EL), _v(lms, R_WR), _v(lms, R_HI))

def _avg_elbow(lms) -> float:
    return (_elbow_angle(lms, "left") + _elbow_angle(lms, "right")) / 2

def _deg(v: float) -> str:
    return f"{v:.0f}\u00b0"

# ─────────────────────────────────────────────────────────────────────────────
# FORM SCORING  (pure geometry)
# ─────────────────────────────────────────────────────────────────────────────

def _score_rep(rec: RepRecord) -> None:
    """Fill rec.form_score, rec.issues, rec.advice in-place."""
    issues = {}
    ded    = 0

    # 1. Depth  (did they go low enough?)
    if rec.min_elbow_angle > THR["min_depth_angle"]:
        issues["Shallow Depth"] = 25;  ded += 25

    # 2. Elbow flare  (shoulder abduction at bottom)
    if rec.shoulder_abduction > THR["elbow_flare_angle"]:
        issues["Elbow Flare"] = 20;    ded += 20

    # 3. Forward lean  (torso angle at bottom)
    if rec.torso_lean_angle > THR["torso_lean_angle"]:
        issues["Forward Lean"] = 20;   ded += 20

    # 4. Asymmetry  (|L elbow - R elbow| at deepest point)
    if rec.elbow_symmetry > THR["asymmetry_angle"]:
        issues["Asymmetry"] = 15;      ded += 15

    # 5. Wrist deviation
    if rec.wrist_deviation > THR["wrist_flare_angle"]:
        issues["Wrist Deviation"] = 10; ded += 10

    rec.form_score = max(0, 100 - ded)
    rec.issues     = list(issues.keys())

    if not rec.issues:
        rec.advice = "Clean rep — great technique!"
    else:
        # Lead with the most costly issue
        worst = max(issues, key=issues.get)
        rec.advice = _ISSUE_ADVICE.get(worst, "")

# ─────────────────────────────────────────────────────────────────────────────
# FATIGUE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def _calc_fatigue(records: list, fps: float):
    """
    Estimate fatigue from:
      - depth degradation (min elbow angle rising over reps)
      - tempo slowdown   (rep duration increasing over reps)
    Returns (level, score 0-100, estimated_max_reps).
    """
    n = len(records)
    if n < 2:
        return "none", 0.0, max(n * 4, 8)

    depths    = [r.min_elbow_angle for r in records]
    durations = [r.duration_sec    for r in records]
    signals   = []

    if _HAS_SCIPY:
        # depth: angle rising means getting shallower (fatigue)
        slope_d, *_ = linregress(range(n), depths)
        depth_change = (slope_d * n / (np.mean(depths) + 1e-6)) * 100
        signals.append(min(max(depth_change, 0) / 30.0, 1.0) * 40)

        # tempo: duration increasing means slowing down (fatigue)
        slope_t, *_ = linregress(range(n), durations)
        tempo_change = (slope_t * n / (np.mean(durations) + 1e-6)) * 100
        signals.append(min(max(tempo_change, 0) / 40.0, 1.0) * 30)
    else:
        # simple fallback: compare first half vs second half
        h = n // 2
        if depths[h:] and depths[:h]:
            d_diff = np.mean(depths[h:]) - np.mean(depths[:h])
            signals.append(min(max(d_diff, 0) / 15.0, 1.0) * 40)

    score = min(100.0, sum(signals))
    level = ("none"     if score < 10 else
             "early"    if score < 30 else
             "moderate" if score < 55 else "severe")

    # estimate remaining reps
    if _HAS_SCIPY and slope_d > 0:
        max_est = n + max(0, int((THR["rep_bot"] - depths[-1]) / (slope_d + 1e-6)))
    else:
        mult    = {"none": 2.5, "early": 1.6, "moderate": 1.2, "severe": 1.0}[level]
        max_est = int(n * mult)

    return level, score, max(max_est, n)

# ─────────────────────────────────────────────────────────────────────────────
# REP DETECTOR  (state machine)
# ─────────────────────────────────────────────────────────────────────────────

class _RepDetector:
    def __init__(self, fps: float):
        self.fps         = fps
        self.buf         = deque(maxlen=7)   # smoothing buffer
        self.phase       = "top"
        self.reps        = 0
        self.in_bot      = False
        self.prev_sm     = None
        self.records:    List[RepRecord] = []
        self._current:   Optional[RepRecord] = None

    def _smooth(self) -> float:
        return float(np.mean(self.buf)) if self.buf else 0.0

    def update(self, lms, frame_idx: int) -> str:
        """
        Feed one frame's landmarks.
        Returns current phase string.
        """
        try:
            le = _elbow_angle(lms, "left")
            re = _elbow_angle(lms, "right")
        except Exception:
            return self.phase

        avg = (le + re) / 2
        self.buf.append(avg)
        if len(self.buf) < 3:
            return self.phase

        sm  = self._smooth()
        t   = frame_idx / self.fps

        # per-frame measurements for open rep
        if self._current is not None:
            self._current._elbow_angles.append(avg)
            try:
                self._current._sym_angles.append(abs(le - re))
                self._current._torso_angles.append(_torso_lean(lms))
                self._current._shoulder_abds.append(
                    (_shoulder_abduction(lms,"left") +
                     _shoulder_abduction(lms,"right")) / 2)
                self._current._wrist_devs.append(
                    (_wrist_deviation(lms,"left") +
                     _wrist_deviation(lms,"right")) / 2)
            except Exception:
                pass

            # track minimum elbow (deepest point)
            if avg < self._current.min_elbow_angle:
                self._current.min_elbow_angle     = avg
                self._current.avg_elbow_at_bottom = avg
                self._current.bottom_frame        = frame_idx
                self._current.bottom_time_sec     = t
                # snapshot geometry at the deepest point
                try:
                    self._current.elbow_symmetry     = abs(le - re)
                    self._current.torso_lean_angle   = _torso_lean(lms)
                    self._current.shoulder_abduction = (
                        _shoulder_abduction(lms,"left") +
                        _shoulder_abduction(lms,"right")) / 2
                    self._current.wrist_deviation = (
                        _wrist_deviation(lms,"left") +
                        _wrist_deviation(lms,"right")) / 2
                except Exception:
                    pass

        # ── state transitions ─────────────────────────────────────────────────
        if sm <= THR["rep_bot"]:
            if not self.in_bot:
                self.in_bot  = True
                self.reps   += 1
                rec = RepRecord(rep_number=self.reps)
                rec.start_frame    = frame_idx
                rec.start_time_sec = t
                rec.min_elbow_angle = avg
                self._current = rec
            self.phase = "bottom"

        elif sm >= THR["rep_top"]:
            if self.in_bot and self._current is not None:
                rec = self._current
                rec.end_frame    = frame_idx
                rec.end_time_sec = t
                rec.duration_sec = t - rec.start_time_sec
                rec.descent_sec  = rec.bottom_time_sec - rec.start_time_sec
                rec.ascent_sec   = rec.end_time_sec   - rec.bottom_time_sec
                self.records.append(rec)
                self._current = None
            self.in_bot = False
            self.phase  = "top"

        else:
            if self.prev_sm is not None:
                self.phase = "descending" if sm < self.prev_sm else "ascending"
            self.in_bot = False

        self.prev_sm = sm
        return self.phase

    def close(self, frame_idx: int) -> None:
        """Call at end-of-video to close any rep that never returned to top."""
        if self._current is not None:
            rec = self._current
            t   = frame_idx / self.fps
            rec.end_frame    = frame_idx
            rec.end_time_sec = t
            rec.duration_sec = t - rec.start_time_sec
            rec.descent_sec  = rec.bottom_time_sec - rec.start_time_sec
            rec.ascent_sec   = 0.0
            self.records.append(rec)
            self._current = None

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_writer(path: str, fps: float, w: int, h: int):
    for codec in ["avc1", "mp4v"]:
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if wr.isOpened():
            print(f"[DIP SERVICE] VideoWriter → codec={codec}")
            return wr, None
        wr.release()
    avi = path.replace(".mp4", "_tmp.avi")
    wr  = cv2.VideoWriter(avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not wr.isOpened():
        raise RuntimeError("VideoWriter failed to open with any codec.")
    print(f"[DIP SERVICE] VideoWriter → XVID fallback")
    return wr, avi

def _finalize_video(out_path: str, tmp_avi) -> str:
    if tmp_avi is None:
        print(f"[DIP SERVICE] Saved → {out_path} ({os.path.getsize(out_path)//1024} KB)")
        return out_path
    if _HAS_FFMPEG:
        cmd = ["ffmpeg", "-y", "-i", tmp_avi,
               "-vcodec","libx264","-crf","23","-preset","fast",
               "-pix_fmt","yuv420p", out_path]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0:
            os.remove(tmp_avi)
            print(f"[DIP SERVICE] Re-encoded → {out_path} ({os.path.getsize(out_path)//1024} KB)")
            return out_path
        print(f"[DIP SERVICE] ffmpeg error: {r.stderr.decode()[:150]}")
    avi_out = out_path.replace(".mp4", ".avi")
    os.rename(tmp_avi, avi_out)
    print(f"[DIP SERVICE] Saved as AVI → {avi_out}")
    return avi_out

# ─────────────────────────────────────────────────────────────────────────────
# FRAME ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

def _put(img, text, xy, scale=0.52, col=WHITE, thick=1, font=FONT_S):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, str(text), (x+1,y+1), font, scale, BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(img, str(text), (x,  y),   font, scale, col,   thick,   cv2.LINE_AA)

def _bar(img, x, y, w, h, val, maxv, fg, bg=(40,40,40)):
    cv2.rectangle(img,(x,y),(x+w,y+h),bg,-1)
    f = int(min(max(val,0)/max(maxv,1),1.0)*w)
    if f > 0: cv2.rectangle(img,(x,y),(x+f,y+h),fg,-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),GRAY,1)

def _panel(img, x1,y1,x2,y2, col=DARK, alpha=0.78, border=None):
    ov = img.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
    if border: cv2.rectangle(img,(x1,y1),(x2,y2),border,1)

def _annotate(frame, phase, lms,
              reps, form_score, fat_level, fat_score, max_reps,
              records: list):
    H, W = frame.shape[:2]

    # ── top-left: rep counter ─────────────────────────────────────────────────
    _panel(frame, 6, 6, 130, 102, DARK, 0.80, CYAN)
    _put(frame, "REPS", (16, 26), scale=0.44, col=GRAY)
    cv2.putText(frame, str(reps), (12,90), FONT_D, 2.5, BLACK, 9, cv2.LINE_AA)
    cv2.putText(frame, str(reps), (12,90), FONT_D, 2.5, CYAN,  3, cv2.LINE_AA)

    # ── phase tag ─────────────────────────────────────────────────────────────
    pc = {"top":GREEN,"ascending":TEAL,"descending":YELLOW,"bottom":ORANGE}.get(phase,WHITE)
    _panel(frame, 6, 106, 210, 138, DARK, 0.78, pc)
    _put(frame, phase.upper(), (14,130), scale=0.68, col=pc, thick=2, font=FONT_D)

    # ── live joint angles ─────────────────────────────────────────────────────
    if lms:
        try:
            le = _elbow_angle(lms,"left");   re = _elbow_angle(lms,"right")
            lt = _torso_lean(lms)
            la = _shoulder_abduction(lms,"left")
            ra = _shoulder_abduction(lms,"right")
            _panel(frame, 6, 142, 210, 260, DARK, 0.72)
            def ac(a): return GREEN if a<90 else YELLOW if a<130 else RED
            _put(frame, f"L elbow  {_deg(le)}", (12,160), col=ac(le))
            _put(frame, f"R elbow  {_deg(re)}", (12,178), col=ac(re))
            tc = GREEN if lt<20 else YELLOW if lt<30 else RED
            _put(frame, f"Torso    {_deg(lt)}", (12,196), col=tc)
            ac2 = GREEN if la<35 else YELLOW if la<50 else RED
            _put(frame, f"L abduct {_deg(la)}", (12,214), col=ac2)
            ac3 = GREEN if ra<35 else YELLOW if ra<50 else RED
            _put(frame, f"R abduct {_deg(ra)}", (12,232), col=ac3)
            # elbow dots on skeleton
            for idx, ang in [(L_EL,le),(R_EL,re)]:
                ex = int(lms[idx].x * W);  ey = int(lms[idx].y * H)
                ac4 = GREEN if ang<90 else YELLOW if ang<130 else RED
                cv2.circle(frame,(ex,ey),8,ac4,-1)
                _put(frame,_deg(ang),(ex+10,ey-10),scale=0.48,col=ac4)
        except Exception:
            pass

    # ── last rep badge (top-right) ────────────────────────────────────────────
    if records:
        last = records[-1]
        rx   = W - 240
        _panel(frame, rx, 6, W-6, 108, DARK, 0.82, CYAN)
        _put(frame, f"REP {last.rep_number} RESULT", (rx+8,24), scale=0.42, col=GRAY)
        sc_col = GREEN if last.form_score>=80 else YELLOW if last.form_score>=50 else RED
        _put(frame, f"Score  {last.form_score}/100",  (rx+8,44), scale=0.50, col=sc_col)
        _put(frame, f"Depth  {last.min_elbow_angle:.0f}\u00b0", (rx+8,62), scale=0.50, col=CYAN)
        _put(frame, f"Dur    {last.duration_sec:.1f}s", (rx+8,80), scale=0.50, col=WHITE)
        if last.issues:
            _put(frame, last.issues[0], (rx+8,98), scale=0.40, col=YELLOW)

    # ── bottom HUD ────────────────────────────────────────────────────────────
    bY = H - 78
    _panel(frame, 0, bY, W, H, DARK, 0.84)

    # form score
    sc_col = GREEN if form_score>=80 else YELLOW if form_score>=50 else RED
    _put(frame, "FORM SCORE", (10, bY+20), scale=0.42, col=GRAY)
    cv2.putText(frame, f"{form_score}/100", (10,bY+52),
                FONT_D, 0.95, BLACK, 6, cv2.LINE_AA)
    cv2.putText(frame, f"{form_score}/100", (10,bY+52),
                FONT_D, 0.95, sc_col, 2, cv2.LINE_AA)
    _bar(frame, 10, bY+58, 150, 8, form_score, 100, sc_col)

    # fatigue
    flc  = FAT_COL.get(fat_level, GRAY)
    mid  = W // 3
    _put(frame, "FATIGUE", (mid, bY+20), scale=0.42, col=GRAY)
    cv2.putText(frame, fat_level.upper(), (mid,bY+52),
                FONT_D, 0.95, BLACK, 6, cv2.LINE_AA)
    cv2.putText(frame, fat_level.upper(), (mid,bY+52),
                FONT_D, 0.95, flc,   2, cv2.LINE_AA)
    _bar(frame, mid, bY+58, 150, 8, fat_score, 100, flc)

    # est max
    mid2 = (2*W)//3
    _put(frame, "EST MAX REPS", (mid2, bY+20), scale=0.42, col=GRAY)
    cv2.putText(frame, str(max_reps), (mid2,bY+52),
                FONT_D, 0.95, BLACK, 6, cv2.LINE_AA)
    cv2.putText(frame, str(max_reps), (mid2,bY+52),
                FONT_D, 0.95, CYAN,  2, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# OVERALL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _build_overall(records: list, fat_level: str, fat_score: float,
                   max_reps: int) -> dict:
    if not records:
        return {
            "total_reps":         0,
            "estimated_max_reps": max_reps,
            "avg_form_score":     0,
            "min_form_score":     0,
            "max_form_score":     0,
            "avg_depth_angle":    0.0,
            "best_depth_angle":   0.0,
            "depth_consistency":  0.0,
            "avg_duration_sec":   0.0,
            "min_duration_sec":   0.0,
            "max_duration_sec":   0.0,
            "avg_descent_sec":    0.0,
            "avg_ascent_sec":     0.0,
            "avg_torso_lean":     0.0,
            "avg_elbow_symmetry": 0.0,
            "recurring_issues":   [],
            "overall_advice":     "No reps detected.",
            "fatigue_level":      fat_level,
            "fatigue_score":      round(fat_score, 2),
            "fatigue_tip":        _FAT_TIP[fat_level],
        }

    scores    = [r.form_score      for r in records]
    depths    = [r.min_elbow_angle for r in records]
    durs      = [r.duration_sec    for r in records]
    desc      = [r.descent_sec     for r in records]
    asc       = [r.ascent_sec      for r in records]
    leans     = [r.torso_lean_angle   for r in records]
    syms      = [r.elbow_symmetry     for r in records]

    # recurring issues: appear in ≥ 50% of reps
    from collections import Counter
    all_issues = [iss for r in records for iss in r.issues]
    counts     = Counter(all_issues)
    recurring  = [iss for iss, cnt in counts.most_common() if cnt >= len(records)//2 + 1]

    # overall advice: worst recurring issue
    overall_advice = ("All reps looked clean — great work!"
                      if not recurring
                      else _ISSUE_ADVICE.get(recurring[0], ""))

    return {
        "total_reps":         len(records),
        "estimated_max_reps": max_reps,
        "avg_form_score":     int(np.mean(scores)),
        "min_form_score":     int(np.min(scores)),
        "max_form_score":     int(np.max(scores)),
        "avg_depth_angle":    round(float(np.mean(depths)), 1),
        "best_depth_angle":   round(float(np.min(depths)),  1),   # lower = deeper
        "depth_consistency":  round(float(np.std(depths)),  1),   # lower = more consistent
        "avg_duration_sec":   round(float(np.mean(durs)),   2),
        "min_duration_sec":   round(float(np.min(durs)),    2),
        "max_duration_sec":   round(float(np.max(durs)),    2),
        "avg_descent_sec":    round(float(np.mean(desc)),   2),
        "avg_ascent_sec":     round(float(np.mean(asc)),    2),
        "avg_torso_lean":     round(float(np.mean(leans)),  1),
        "avg_elbow_symmetry": round(float(np.mean(syms)),   1),
        "recurring_issues":   recurring,
        "overall_advice":     overall_advice,
        "fatigue_level":      fat_level,
        "fatigue_score":      round(fat_score, 2),
        "fatigue_tip":        _FAT_TIP[fat_level],
    }

def _build_summary(records: list, overall: dict) -> str:
    n = len(records)
    if n == 0:
        return "No reps were detected. Ensure your full body is visible and dip below 90° elbow angle."

    parts = [
        f"You completed {n} rep(s) (estimated max: {overall['estimated_max_reps']}).",
        f"Average form score: {overall['avg_form_score']}/100 "
        f"(best rep: {overall['max_form_score']}, worst: {overall['min_form_score']}).",
        f"Average depth: {overall['avg_depth_angle']}\u00b0 elbow angle "
        f"(best: {overall['best_depth_angle']}\u00b0).",
        f"Average rep duration: {overall['avg_duration_sec']}s "
        f"(descent {overall['avg_descent_sec']}s / ascent {overall['avg_ascent_sec']}s).",
    ]
    if overall["recurring_issues"]:
        parts.append(f"Recurring issues: {', '.join(overall['recurring_issues'])}. "
                     + overall["overall_advice"])
    else:
        parts.append("No recurring form issues — excellent technique!")
    parts.append(
        f"Fatigue: {overall['fatigue_level']} ({overall['fatigue_score']:.0f}/100). "
        + overall["fatigue_tip"]
    )
    return "  ".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dip_video(input_path: str, output_path: str) -> dict:
    """
    Analyse a tricep-dip video using MediaPipe Pose (no ML model needed).

    Parameters
    ----------
    input_path  : uploaded raw video path
    output_path : path for the annotated output video

    Returns
    -------
    dict matching DipAnalysisResponse
    (_saved_path injected for the API layer to build the video URL)
    """
    print(f"\n[DIP SERVICE] Starting  input={input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[DIP SERVICE] {raw_w}x{raw_h}  {fps:.1f}fps  {total_frm} frames")

    writer, tmp_avi = _make_writer(output_path, fps, raw_w, raw_h)

    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector    = _RepDetector(fps)
    fat_level   = "none"
    fat_score   = 0.0
    max_reps    = 8
    form_score  = 100        # live overlay value (avg of completed reps)
    phase       = "top"
    lms_live    = None
    fidx        = 0

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        fidx += 1
        frame = raw.copy()

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lms_live = None

        if res.pose_landmarks:
            lms_live = res.pose_landmarks.landmark

            # skeleton overlay
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0,200,90), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0,130,255), thickness=2),
            )

            prev_count = detector.reps
            phase = detector.update(lms_live, fidx)

            # rep just completed → score it + update fatigue
            if detector.reps > prev_count and detector.records:
                completed = detector.records[-1]
                _score_rep(completed)

                fat_level, fat_score, max_reps = _calc_fatigue(
                    detector.records, fps)
                completed.fatigue_at_rep = fat_level

                # update live form score overlay
                form_score = int(np.mean([r.form_score for r in detector.records]))
                print(f"[DIP SERVICE] Rep {completed.rep_number}  "
                      f"score={completed.form_score}  "
                      f"depth={completed.min_elbow_angle:.0f}\u00b0  "
                      f"dur={completed.duration_sec:.1f}s  "
                      f"issues={completed.issues}")

        _annotate(frame, phase, lms_live,
                  detector.reps, form_score, fat_level, fat_score, max_reps,
                  detector.records)
        writer.write(frame)

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    pose.close()
    detector.close(fidx)    # finalise any incomplete rep

    saved_path = _finalize_video(output_path, tmp_avi)

    # final fatigue with all records
    if detector.records:
        fat_level, fat_score, max_reps = _calc_fatigue(detector.records, fps)
        for rec in detector.records:
            if not rec.issues:   # score any unseen reps
                _score_rep(rec)

    overall = _build_overall(detector.records, fat_level, fat_score, max_reps)
    summary = _build_summary(detector.records, overall)

    print(f"[DIP SERVICE] Done  reps={overall['total_reps']}  "
          f"avg_score={overall['avg_form_score']}  fatigue={fat_level}")

    return {
        "_saved_path":      saved_path,
        "overall_analysis": overall,
        "per_rep_analysis": [r.to_dict() for r in detector.records],
        "summary":          summary,
    }