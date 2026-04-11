import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
<<<<<<< HEAD
from collections import deque
=======
from collections import deque, Counter
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69

print("📦 Loading pushup_service...")

# =====================================================
# MODEL PATH
# =====================================================
<<<<<<< HEAD
# =====================================================
# MODEL PATH (DYNAMIC & SAFE)
# =====================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)

MODEL_PATH = r"C:\Users\rauls\Desktop\Perform-Machine-learning-Project-\serverSide\ML_Model\pushup_model\saved_models\GradientBoosting.pkl"

# =====================================================
# CONFIG
# =====================================================
ELBOW_DOWN = 90
ELBOW_UP = 155
SMOOTHING_WINDOW = 5
FRAME_CONFIRM = 2
=======
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
MODEL_PATH = r"C:\major_project\serverSide\ML_Model\pushup_model\saved_models\GradientBoosting.pkl"

# =====================================================
# SERVER CONFIG — localhost video URL base
# =====================================================
VIDEO_SERVER_BASE = "http://localhost:5000/outputs"   # ← change port if needed

# =====================================================
# REP DETECTION CONFIG
# =====================================================
ELBOW_DOWN       = 90
ELBOW_UP         = 155
SMOOTHING_WINDOW = 5
FRAME_CONFIRM    = 2

# =====================================================
# STRICT FORM THRESHOLDS
# =====================================================
BODY_ANGLE_MIN      = 165
BODY_ANGLE_MAX      = 195
HIP_SAG_THRESHOLD   = 0.08
HEAD_NECK_TOLERANCE = 25

# =====================================================
# WRONG-EXERCISE DETECTION THRESHOLDS
# =====================================================
# Pushup: body horizontal (body_angle ~180), wrists below shoulders
# Squat:  body vertical   (hip_knee < 120°), person standing/bending
# Other:  neither
PUSHUP_BODY_ANGLE_MIN   = 140   # Body roughly horizontal
PUSHUP_BODY_ANGLE_MAX   = 210
PUSHUP_WRIST_Y_BELOW    = 0.0   # Wrist.y > shoulder.y  (image coords)
WRONG_EXERCISE_FRAMES   = 15    # consecutive non-pushup frames before warning
WRONG_EXERCISE_COOLDOWN = 45    # frames between repeated warnings
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69

# =====================================================
# LAZY MODEL LOAD
# =====================================================
_model = None

def get_model():
    global _model
    if _model is None:
<<<<<<< HEAD
        print("📦 Loading ML model...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print("✅ Pushup ML model loaded")
=======
        print("  📦  Loading ML model from disk...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"  ❌  Model not found: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print("  ✅  Pushup ML model loaded successfully")
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
    return _model

# =====================================================
# MEDIAPIPE
# =====================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# =====================================================
<<<<<<< HEAD
# HELPER FUNCTIONS
=======
# ANGLE / GEOMETRY HELPERS
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
# =====================================================
def angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(np.degrees(radians))
    return 360 - ang if ang > 180 else ang

<<<<<<< HEAD

def elbow_angle(lm):
    return angle(lm[11], lm[13], lm[15])


def body_angle(lm):
    return angle(lm[11], lm[23], lm[27])


def is_pushup_position(lm):
    body_ang = body_angle(lm)
    shoulder, hip, wrist = lm[11], lm[23], lm[15]
    return (
        160 <= body_ang <= 200 and
=======
def elbow_angle(lm):
    return angle(lm[11], lm[13], lm[15])

def body_angle(lm):
    return angle(lm[11], lm[23], lm[27])

def head_angle(lm):
    return angle(lm[0], lm[11], lm[23])

# =====================================================
# WRONG-EXERCISE DETECTOR
# =====================================================
def classify_exercise(lm):
    """
    Classify what exercise the person appears to be doing.
    Returns: ('pushup' | 'squat' | 'standing' | 'unknown', reason_str)
    """
    b_ang      = body_angle(lm)
    shoulder   = lm[11]
    hip        = lm[23]
    wrist      = lm[15]
    knee       = lm[25]
    ankle      = lm[27]

    # Body horizontal + wrists below shoulders → pushup
    is_horizontal = PUSHUP_BODY_ANGLE_MIN <= b_ang <= PUSHUP_BODY_ANGLE_MAX
    wrist_below   = wrist.y > shoulder.y - 0.05

    if is_horizontal and wrist_below:
        return "pushup", f"body_angle={b_ang:.1f}°"

    # Body upright + knee bending → squat
    hip_knee_diff = abs(hip.y - knee.y)
    if b_ang > 130 and hip_knee_diff < 0.25:
        # Hip near knee height = squatting
        if hip.y > knee.y - 0.05:
            return "squat", f"body_angle={b_ang:.1f}°, hip≈knee"
        # Body upright, hip well above knee = standing
        return "standing", f"body_angle={b_ang:.1f}°"

    return "unknown", f"body_angle={b_ang:.1f}°"


def wrong_exercise_message(detected):
    msgs = {
        "squat":    "⚠️  This looks like a SQUAT, not a pushup! Please get into pushup position.",
        "standing": "⚠️  Person appears to be STANDING. Please get into pushup position.",
        "unknown":  "⚠️  Unrecognised exercise. Expected PUSHUP position (body horizontal).",
    }
    return msgs.get(detected, "⚠️  Wrong exercise detected. Please perform a pushup.")


# =====================================================
# PUSHUP POSITION CHECK
# =====================================================
def is_pushup_position(lm):
    b_ang = body_angle(lm)
    shoulder, hip, wrist = lm[11], lm[23], lm[15]
    return (
        160 <= b_ang <= 200 and
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
        wrist.y > shoulder.y and
        abs(hip.y - shoulder.y) < 0.15
    )

<<<<<<< HEAD

def form_score(lm):
    body_ang = body_angle(lm)
    return int(np.clip(100 - abs(180 - body_ang) * 1.8, 0, 100))


def rep_rating(score):
    if score >= 85:
        return "Excellent", 2
    elif score >= 70:
        return "Good", 1
    elif score >= 50:
        return "Poor", 0
    else:
        return "Bad", -2


def fatigue_index(log):
    if len(log) < 2:
        return 0

    form_drop = max(0, log[0]["form"] - log[-1]["form"])
    time_increase = max(0, log[-1]["time"] - log[0]["time"]) * 10
    bad_ratio = sum(1 for r in log if r["form"] < 50) / len(log) * 100

    fatigue = (
        0.5 * form_drop +
        0.3 * time_increase +
        0.2 * bad_ratio
    )
    return int(np.clip(fatigue, 0, 100))


def fatigue_level(v):
    if v < 30:
        return "LOW"
    elif v < 60:
        return "MODERATE"
    else:
        return "HIGH"
=======
# =====================================================
# STRICT FORM ANALYSIS
# =====================================================
def analyze_form_strict(lm):
    issues    = []
    breakdown = {}

    # 1. Body alignment (40 pts)
    b_ang    = body_angle(lm)
    body_dev = abs(180 - b_ang)
    breakdown["body_alignment"] = int(np.clip(40 - body_dev * 2.5, 0, 40))
    if body_dev > 15:
        issues.append("Hips too high (piking)" if b_ang < 180 else "Hips sagging — keep core tight")

    # 2. Hip position (20 pts)
    shoulder, hip, ankle = lm[11], lm[23], lm[27]
    expected_hip_y = shoulder.y + (ankle.y - shoulder.y) * (
        (hip.x - shoulder.x) / max(abs(ankle.x - shoulder.x), 0.001)
    )
    hip_offset = abs(hip.y - expected_hip_y)
    breakdown["hip_position"] = int(np.clip(20 - hip_offset * 200, 0, 20))
    if hip_offset > HIP_SAG_THRESHOLD:
        issues.append("Hips dropping — engage your core" if hip.y > expected_hip_y else "Hips raised — lower your hips")

    # 3. Head/neck (15 pts)
    h_ang    = head_angle(lm)
    head_dev = abs(180 - h_ang)
    breakdown["head_alignment"] = int(np.clip(15 - head_dev * 0.8, 0, 15))
    if head_dev > HEAD_NECK_TOLERANCE:
        issues.append("Head too high — look down" if lm[0].y < lm[11].y else "Head drooping — keep neck neutral")

    # 4. Arm width (15 pts)
    arm_w    = abs(lm[15].x - lm[16].x)
    shldr_w  = abs(lm[11].x - lm[12].x)
    ratio    = arm_w / max(shldr_w, 0.001)
    w_dev    = abs(ratio - 1.25)
    breakdown["arm_width"] = int(np.clip(15 - w_dev * 20, 0, 15))
    if ratio < 0.8:
        issues.append("Hands too close together")
    elif ratio > 1.8:
        issues.append("Hands too wide apart")

    # 5. Range of motion (10 pts)
    e_ang = elbow_angle(lm)
    breakdown["range_of_motion"] = int(np.clip(10 - max(0, e_ang - ELBOW_DOWN) * 0.3, 0, 10))
    if e_ang > ELBOW_DOWN + 15:
        issues.append("Not going low enough — full range of motion")

    return sum(breakdown.values()), issues, breakdown


def form_grade(score):
    if score >= 88: return "EXCELLENT", (0, 220, 0)
    if score >= 72: return "GOOD",      (0, 200, 120)
    if score >= 55: return "FAIR",      (0, 165, 255)
    if score >= 38: return "POOR",      (0, 80, 255)
    return "BAD", (0, 0, 220)

def rep_rating(score):
    if score >= 85: return "Excellent", 2
    if score >= 70: return "Good",      1
    if score >= 50: return "Poor",      0
    return "Bad", -2

def fatigue_index(log):
    if len(log) < 2: return 0
    form_drop     = max(0, log[0]["form"] - log[-1]["form"])
    time_increase = max(0, log[-1]["time"] - log[0]["time"]) * 10
    bad_ratio     = sum(1 for r in log if r["form"] < 50) / len(log) * 100
    return int(np.clip(0.5*form_drop + 0.3*time_increase + 0.2*bad_ratio, 0, 100))

def fatigue_level(v):
    if v < 30: return "LOW"
    if v < 60: return "MODERATE"
    return "HIGH"

# =====================================================
# DRAWING HELPERS
# =====================================================
def draw_form_panel(frame, score, grade, grade_color, issues, breakdown, state, pushups):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "PUSHUP ANALYZER", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)
    cv2.line(frame, (10, 35), (290, 35), (80, 80, 80), 1)

    cv2.putText(frame, f"Reps: {pushups}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    state_color = (0, 255, 255) if state == "DOWN" else (200, 200, 200)
    cv2.putText(frame, f"State: {state}", (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1)
    cv2.line(frame, (10, 100), (290, 100), (80, 80, 80), 1)

    cv2.putText(frame, "FORM SCORE", (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    bar_w = int(270 * score / 100)
    cv2.rectangle(frame, (10, 128), (280, 148), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 128), (10 + bar_w, 148), grade_color, -1)
    cv2.putText(frame, f"{score}%  {grade}", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.65, grade_color, 2)
    cv2.line(frame, (10, 173), (290, 173), (80, 80, 80), 1)

    labels = {"body_alignment":"Body Line","hip_position":"Hips",
              "head_alignment":"Head/Neck","arm_width":"Arm Width","range_of_motion":"ROM"}
    maxes  = {"body_alignment":40,"hip_position":20,"head_alignment":15,"arm_width":15,"range_of_motion":10}
    y = 185
    for key, label in labels.items():
        val   = breakdown.get(key, 0)
        mx    = maxes[key]
        ratio = val / mx
        color = (0, int(200 * ratio), int(200 * (1 - ratio)))
        bw    = int(130 * ratio)
        cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        cv2.rectangle(frame, (120, y-11), (250, y+2), (50, 50, 50), -1)
        cv2.rectangle(frame, (120, y-11), (120+bw, y+2), color, -1)
        cv2.putText(frame, f"{val}/{mx}", (255, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        y += 22

    cv2.line(frame, (10, y), (290, y), (80, 80, 80), 1)
    y += 14

    if issues:
        cv2.putText(frame, "FORM ISSUES:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1)
        y += 18
        for issue in issues[:4]:
            words, line = issue.split(), ""
            for word in words:
                if len(line + word) < 30:
                    line += word + " "
                else:
                    cv2.putText(frame, f"  {line.strip()}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 140, 255), 1)
                    y += 16; line = word + " "
            if line:
                cv2.putText(frame, f"  {line.strip()}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 140, 255), 1)
                y += 16
    else:
        cv2.putText(frame, "Great form!", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)


def draw_alert(frame, issues):
    if not issues: return
    h, w   = frame.shape[:2]
    alert  = issues[0]
    font   = cv2.FONT_HERSHEY_DUPLEX
    scale  = 0.75
    thick  = 2
    (tw, th), _ = cv2.getTextSize(alert, font, scale, thick)
    x = (w - tw) // 2
    y = h - 60
    cv2.rectangle(frame, (x-10, y-th-8), (x+tw+10, y+8), (0, 0, 180), -1)
    cv2.rectangle(frame, (x-10, y-th-8), (x+tw+10, y+8), (0, 0, 255), 2)
    cv2.putText(frame, alert, (x, y), font, scale, (255, 255, 255), thick)


def draw_wrong_exercise_banner(frame, exercise_type):
    """Full-width red banner at top warning about wrong exercise."""
    h, w = frame.shape[:2]
    msg   = f"WRONG EXERCISE DETECTED: {exercise_type.upper()} — Do PUSHUPS!"
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.72
    thick = 2
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)

    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 255), 3)
    x = max(10, (w - tw) // 2)
    cv2.putText(frame, msg, (x, 34), font, scale, (255, 255, 255), thick)

>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69

# =====================================================
# MAIN SERVICE FUNCTION
# =====================================================
<<<<<<< HEAD
def analyze_pushup_video(input_path, output_path):

    get_model()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open input video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25

    print("🎬 Video opened successfully")
    print("🎞 Resolution:", w, "x", h, "FPS:", fps)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open")
=======
def analyze_pushup_video(input_path, output_path, video_filename=None):
    """
    Analyze a pushup video.

    Args:
        input_path      : full path to input video
        output_path     : full path to save annotated video
        video_filename  : filename used to build localhost URL
                          (defaults to basename of output_path)

    Returns dict with all stats + localhost video URL + wrong-exercise warnings.
    """

    print("\n" + "=" * 60)
    print("  PUSHUP VIDEO ANALYSIS  —  STARTING")
    print("=" * 60)
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")

    # ── STEP 1 — Load model ────────────────────────
    print("\n  [1/6]  Loading ML model...")
    get_model()

    # ── STEP 2 — Open video ────────────────────────
    print("  [2/6]  Opening input video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"  ❌  Cannot open video: {input_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"       Resolution : {w}×{h}")
    print(f"       FPS        : {fps}")
    print(f"       Frames     : {total_frames}")
    print(f"       Duration   : ~{total_frames//fps}s")

    # ── STEP 3 — Init writer & pose ────────────────
    print("  [3/6]  Initialising VideoWriter & MediaPipe Pose...")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("  ❌  VideoWriter failed to open (codec issue?)")
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
<<<<<<< HEAD

    state = "UP"
    pushups = 0
    predicted_max = 5

    elbow_buf = deque(maxlen=SMOOTHING_WINDOW)
    down_frames = 0
    up_frames = 0
    down_scores = []
    valid_down = False

    # ✅ NEW: per-rep detailed log
    rep_log = []
    last_rep_time = None

=======
    print("       MediaPipe Pose ready")

    # ── State variables ────────────────────────────
    state        = "UP"
    pushups      = 0
    predicted_max = 5

    elbow_buf        = deque(maxlen=SMOOTHING_WINDOW)
    down_frames      = 0
    up_frames        = 0
    down_scores      = []
    down_breakdowns  = []
    down_issues_all  = []
    valid_down       = False

    rep_log       = []
    last_rep_time = None

    # Wrong-exercise tracking
    wrong_ex_frames    = 0          # consecutive non-pushup frames
    wrong_ex_cooldown  = 0          # frames until next warning allowed
    wrong_ex_events    = []         # list of {frame, exercise, message}
    current_wrong_type = None       # shown on screen while active

    # Smooth display state
    current_score      = 100
    current_grade      = "EXCELLENT"
    current_grade_color = (0, 220, 0)
    current_issues     = []
    current_breakdown  = {
        "body_alignment": 40, "hip_position": 20,
        "head_alignment": 15, "arm_width": 15, "range_of_motion": 10,
    }

    frame_idx      = 0
    no_pose_frames = 0
    start_time     = time.time()
    log_interval   = max(1, total_frames // 20)  # log every 5%

    print(f"\n  [4/6]  Processing frames...")
    print(f"         (progress logged every ~5% = {log_interval} frames)\n")

    # ── STEP 4 — Main processing loop ─────────────
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

<<<<<<< HEAD
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            elbow_buf.append(elbow_angle(lm))
            smooth_elbow = np.mean(elbow_buf)
            score = form_score(lm)

            if smooth_elbow < ELBOW_DOWN:
                down_frames += 1
                up_frames = 0
            elif smooth_elbow > ELBOW_UP:
                up_frames += 1
                down_frames = 0

            if state == "UP" and down_frames >= FRAME_CONFIRM:
                state = "DOWN"
                down_scores = []
                valid_down = is_pushup_position(lm)

            if state == "DOWN":
                down_scores.append(score)

            if state == "DOWN" and up_frames >= FRAME_CONFIRM:
                if valid_down:
                    pushups += 1

                    now = time.time()
                    rep_time = now - last_rep_time if last_rep_time else 1.0
                    last_rep_time = now

                    rep_form = int((np.mean(down_scores) + score) / 2)
                    rating, delta = rep_rating(rep_form)
                    predicted_max = max(5, predicted_max + delta)

                    # ✅ STORE FORM % PER REP
                    rep_log.append({
                        "rep": pushups,
                        "form": rep_form,
                        "rating": rating,
                        "time": round(rep_time, 2)
                    })

                state = "UP"

            fatigue = fatigue_index(rep_log)

            cv2.putText(frame, f"Pushups: {pushups}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Fatigue: {fatigue}% ({fatigue_level(fatigue)})",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        out.write(frame)

=======
        frame_idx += 1

        # Periodic progress log
        if frame_idx % log_interval == 0 or frame_idx == 1:
            elapsed = time.time() - start_time
            pct     = int(frame_idx / max(total_frames, 1) * 100)
            rate    = frame_idx / max(elapsed, 0.001)
            eta     = (total_frames - frame_idx) / max(rate, 0.001)
            print(f"         Frame {frame_idx:5d}/{total_frames}  "
                  f"({pct:3d}%)  |  reps={pushups}  |  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.0f}s")

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not res.pose_landmarks:
            no_pose_frames += 1
            out.write(frame)
            continue

        lm = res.pose_landmarks.landmark

        # ── Wrong-exercise detection ───────────────
        exercise_type, ex_reason = classify_exercise(lm)

        if exercise_type != "pushup":
            wrong_ex_frames += 1
            if wrong_ex_frames >= WRONG_EXERCISE_FRAMES and wrong_ex_cooldown <= 0:
                msg = wrong_exercise_message(exercise_type)
                wrong_ex_events.append({
                    "frame":    frame_idx,
                    "exercise": exercise_type,
                    "message":  msg,
                    "reason":   ex_reason,
                })
                wrong_ex_cooldown  = WRONG_EXERCISE_COOLDOWN
                current_wrong_type = exercise_type
                print(f"  ⚠️   WRONG EXERCISE @ frame {frame_idx} — "
                      f"detected '{exercise_type}' ({ex_reason})")
                print(f"         MSG: {msg}")
        else:
            wrong_ex_frames    = max(0, wrong_ex_frames - 1)
            current_wrong_type = None

        if wrong_ex_cooldown > 0:
            wrong_ex_cooldown -= 1

        # ── Elbow smoothing ────────────────────────
        elbow_buf.append(elbow_angle(lm))
        smooth_elbow = np.mean(elbow_buf)

        # ── Form analysis ──────────────────────────
        score, issues, breakdown = analyze_form_strict(lm)
        current_score       = int(0.7 * current_score + 0.3 * score)
        current_grade, current_grade_color = form_grade(current_score)
        current_issues      = issues
        current_breakdown   = breakdown

        # ── Rep state machine ──────────────────────
        if smooth_elbow < ELBOW_DOWN:
            down_frames += 1; up_frames = 0
        elif smooth_elbow > ELBOW_UP:
            up_frames += 1; down_frames = 0

        if state == "UP" and down_frames >= FRAME_CONFIRM:
            state          = "DOWN"
            down_scores    = []
            down_breakdowns = []
            down_issues_all = []
            valid_down     = is_pushup_position(lm)
            print(f"         ↓ Rep starting (frame {frame_idx})  "
                  f"valid_position={valid_down}")

        if state == "DOWN":
            down_scores.append(score)
            down_breakdowns.append(breakdown)
            down_issues_all.extend(issues)

        if state == "DOWN" and up_frames >= FRAME_CONFIRM:
            if valid_down:
                pushups += 1
                now       = time.time()
                rep_time  = round(now - last_rep_time, 2) if last_rep_time else 1.0
                last_rep_time = now

                rep_form = int((np.mean(down_scores) + score) / 2)
                rating, delta = rep_rating(rep_form)
                predicted_max = max(5, predicted_max + delta)

                avg_breakdown = {}
                if down_breakdowns:
                    for k in down_breakdowns[0]:
                        avg_breakdown[k] = int(np.mean([d[k] for d in down_breakdowns]))

                unique_issues = list(dict.fromkeys(down_issues_all))

                rep_log.append({
                    "rep":            pushups,
                    "form":           rep_form,
                    "form_grade":     rating,
                    "rating":         rating,
                    "time":           rep_time,
                    "form_breakdown": avg_breakdown,
                    "form_issues":    unique_issues,
                    "is_good_form":   rep_form >= 70,
                })
                print(f"         ✅ Rep {pushups} complete — "
                      f"form={rep_form}% ({rating})  time={rep_time}s  "
                      f"issues={unique_issues or 'none'}")
            else:
                print(f"         ⚠️  Movement detected but invalid pushup position "
                      f"(frame {frame_idx}) — rep NOT counted")
            state = "UP"

        fatigue = fatigue_index(rep_log)

        # ── Draw skeleton ──────────────────────────
        mp_draw.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(200,200,200), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(100,255,100), thickness=2)
        )

        # ── Draw form panel ────────────────────────
        draw_form_panel(frame, current_score, current_grade,
                        current_grade_color, current_issues,
                        current_breakdown, state, pushups)

        # ── Draw form alert ────────────────────────
        critical = [i for i in current_issues if any(
            kw in i.lower() for kw in ["sag","hips","core","range"])]
        draw_alert(frame, critical)

        # ── Draw wrong-exercise banner ─────────────
        if current_wrong_type and wrong_ex_cooldown > WRONG_EXERCISE_COOLDOWN - 60:
            draw_wrong_exercise_banner(frame, current_wrong_type)

        # ── Fatigue badge ──────────────────────────
        cv2.putText(frame, f"Fatigue: {fatigue}% ({fatigue_level(fatigue)})",
                    (w - 340, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out.write(frame)

    # ── STEP 5 — Cleanup ──────────────────────────
    print(f"\n  [5/6]  Finalising video...")
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
    cap.release()
    out.release()
    pose.close()

<<<<<<< HEAD
    final_fatigue = fatigue_index(rep_log)

    return {
        "pushup_count": pushups,
        "fatigue": final_fatigue,
        "fatigue_level": fatigue_level(final_fatigue),
        "estimated_range": f"{predicted_max-2} - {predicted_max+2}",
        "reps": rep_log,                 # ✅ NEW FIELD
        "output_video_path": os.path.abspath(output_path)

    }
=======
    elapsed_total = time.time() - start_time
    print(f"         Processed {frame_idx} frames in {elapsed_total:.1f}s "
          f"({frame_idx/max(elapsed_total,0.001):.1f} fps)")
    print(f"         No-pose frames : {no_pose_frames} "
          f"({100*no_pose_frames//max(frame_idx,1)}%)")
    print(f"         Video saved    : {output_path}")

    # ── STEP 6 — Build response packet ────────────
    print("  [6/6]  Building response packet...")

    final_fatigue = fatigue_index(rep_log)

    all_issues_flat = []
    for r in rep_log:
        all_issues_flat.extend(r.get("form_issues", []))
    issue_freq  = Counter(all_issues_flat)
    top_issues  = [{"issue": k, "occurrences": v}
                   for k, v in issue_freq.most_common(5)]

    bad_rep_count = sum(1 for r in rep_log if not r["is_good_form"])
    overall_form  = int(np.mean([r["form"] for r in rep_log])) if rep_log else 0
    overall_grade, _ = form_grade(overall_form)

    # Build localhost video URL
    fname      = video_filename or os.path.basename(output_path)
    video_url  = f"{VIDEO_SERVER_BASE}/{fname}"

    # Wrong-exercise summary
    wrong_ex_summary = {
        "detected":       len(wrong_ex_events) > 0,
        "total_warnings": len(wrong_ex_events),
        "events":         wrong_ex_events,
        "warning_message": (
            wrong_ex_events[-1]["message"] if wrong_ex_events else None
        ),
    }

    packet = {
        # ── Core stats
        "pushup_count":       pushups,
        "fatigue":            final_fatigue,
        "fatigue_level":      fatigue_level(final_fatigue),
        "estimated_range":    f"{predicted_max-2} - {predicted_max+2}",

        # ── Form summary
        "overall_form_score": overall_form,
        "overall_form_grade": overall_grade,
        "bad_form_reps":      bad_rep_count,
        "top_form_issues":    top_issues,

        # ── Per-rep detail
        "reps":               rep_log,

        # ── Wrong exercise
        "wrong_exercise":     wrong_ex_summary,

        # ── Video paths / URL
        "output_video_path":  os.path.abspath(output_path),
        "video_url":          video_url,         # ← localhost link
    }

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Pushups          : {pushups}")
    print(f"  Overall form     : {overall_form}% ({overall_grade})")
    print(f"  Fatigue          : {final_fatigue}% ({fatigue_level(final_fatigue)})")
    print(f"  Bad-form reps    : {bad_rep_count}/{len(rep_log)}")
    print(f"  Wrong-ex events  : {len(wrong_ex_events)}")
    print(f"  Video URL        : {video_url}")
    if wrong_ex_events:
        print(f"\n  ⚠️  Wrong-exercise warnings:")
        for ev in wrong_ex_events:
            print(f"     Frame {ev['frame']:5d} — {ev['exercise']}  |  {ev['message']}")
    if top_issues:
        print(f"\n  Top form issues:")
        for ti in top_issues:
            print(f"     {ti['issue']:45s}  ×{ti['occurrences']}")
    print("=" * 60 + "\n")

    return packet
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
