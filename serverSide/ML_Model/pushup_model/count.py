import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt

# ===============================
# PATHS
# ===============================
MODEL_PATH = r"C:\major_project\serverSide\ML_Model\pushup_model\saved_models\GradientBoosting.pkl"
VIDEO_PATH = r"C:\major_project\serverSide\ML_Model\pushup_model\testing_video\saily.mp4"

# ===============================
# CONFIG
# ===============================
ELBOW_DOWN = 90
ELBOW_UP = 155
SMOOTHING_WINDOW = 5
FRAME_CONFIRM = 2

# ===============================
# LOAD MODEL
# ===============================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===============================
# MEDIAPIPE
# ===============================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ===============================
# HELPER FUNCTIONS
# ===============================
def angle(a, b, c):
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    ang = abs(np.degrees(
        np.arctan2(c[1]-b[1], c[0]-b[0]) -
        np.arctan2(a[1]-b[1], a[0]-b[0])
    ))
    return 360 - ang if ang > 180 else ang


def elbow_angle(lm):
    return angle(lm[11], lm[13], lm[15])


def body_angle(lm):
    return angle(lm[11], lm[23], lm[27])


def is_pushup_position(lm):
    body_ang = body_angle(lm)
    shoulder, hip, wrist = lm[11], lm[23], lm[15]

    horizontal = 160 <= body_ang <= 200
    wrist_below_shoulder = wrist.y > shoulder.y
    hip_near_shoulder = abs(hip.y - shoulder.y) < 0.15

    return horizontal and wrist_below_shoulder and hip_near_shoulder


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


# ===============================
# FATIGUE FUNCTIONS
# ===============================
def fatigue_index(rep_log):
    if len(rep_log) < 2:
        return 0

    form_start = rep_log[0]["final"]
    form_end = rep_log[-1]["final"]
    form_drop = max(0, form_start - form_end)

    time_start = rep_log[0]["time"]
    time_end = rep_log[-1]["time"]
    time_increase = max(0, time_end - time_start) * 10

    bad_reps = sum(1 for r in rep_log if r["final"] < 50)
    bad_ratio = (bad_reps / len(rep_log)) * 100

    fatigue = (
        0.5 * form_drop +
        0.3 * time_increase +
        0.2 * bad_ratio
    )

    return int(np.clip(fatigue, 0, 100))


def fatigue_level(value):
    if value < 30:
        return "LOW"
    elif value < 60:
        return "MODERATE"
    else:
        return "HIGH"


# ===============================
# VIDEO
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)

# ===============================
# STATE VARIABLES
# ===============================
state = "UP"
pushup_count = 0
predicted_max = 5

elbow_buf = deque(maxlen=SMOOTHING_WINDOW)
down_frames = 0
up_frames = 0

down_scores = []
valid_down = False

session_log = []
last_rep_time = None

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        elbow_buf.append(elbow_angle(lm))
        smooth_elbow = np.mean(elbow_buf)
        score = form_score(lm)

        # FSM
        if smooth_elbow < ELBOW_DOWN:
            down_frames += 1
            up_frames = 0
        elif smooth_elbow > ELBOW_UP:
            up_frames += 1
            down_frames = 0
        else:
            down_frames = up_frames = 0

        # ENTER DOWN
        if state == "UP" and down_frames >= FRAME_CONFIRM:
            state = "DOWN"
            down_scores = []
            valid_down = is_pushup_position(lm)

        # COLLECT DOWN SCORES
        if state == "DOWN":
            down_scores.append(score)

        # COMPLETE REP
        if state == "DOWN" and up_frames >= FRAME_CONFIRM:
            if valid_down:
                pushup_count += 1

                now = time.time()
                rep_time = now - last_rep_time if last_rep_time else 1.0
                last_rep_time = now

                avg_down = int(np.mean(down_scores))
                avg_up = score
                rep_form = int((avg_down + avg_up) / 2)

                rating, delta = rep_rating(rep_form)
                predicted_max = max(5, predicted_max + delta)

                session_log.append({
                    "rep": pushup_count,
                    "down": avg_down,
                    "up": avg_up,
                    "final": rep_form,
                    "rating": rating,
                    "time": rep_time
                })

                fatigue = fatigue_index(session_log)

                print(
                    f"Rep {pushup_count} | "
                    f"Form {rep_form}% ({rating}) | "
                    f"Time {rep_time:.2f}s | "
                    f"Fatigue {fatigue}% ({fatigue_level(fatigue)})"
                )

            state = "UP"

        # ===============================
        # UI
        # ===============================
        fatigue = fatigue_index(session_log)

        cv2.putText(frame, f"Pushups: {pushup_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Fatigue: {fatigue}% ({fatigue_level(fatigue)})", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        mp_draw.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Push-up Fatigue Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ===============================
# SESSION REPORT
# ===============================
print("\n📊 SESSION REPORT")
for r in session_log:
    print(
        f"Rep {r['rep']} | "
        f"Down {r['down']}% | Up {r['up']}% | "
        f"Final {r['final']}% ({r['rating']}) | "
        f"Time {r['time']:.2f}s"
    )

final_fatigue = fatigue_index(session_log)
print("\n🧠 FINAL FATIGUE:", final_fatigue, fatigue_level(final_fatigue))
print("\n✅ FINAL ESTIMATED RANGE:", predicted_max - 2, "-", predicted_max + 2)

# ===============================
# GRAPHS
# ===============================
# Rep Quality
plt.figure()
plt.plot(
    [r["rep"] for r in session_log],
    [r["final"] for r in session_log],
    marker='o'
)
plt.axhline(70, linestyle='--')
plt.axhline(85, linestyle='--')
plt.xlabel("Rep Number")
plt.ylabel("Form Score")
plt.title("Push-up Rep Quality")
plt.show()

# Fatigue Progression
plt.figure()
plt.plot(
    [r["rep"] for r in session_log],
    [fatigue_index(session_log[:i+1]) for i in range(len(session_log))],
    marker='o'
)
plt.xlabel("Rep Number")
plt.ylabel("Fatigue Index")
plt.title("Fatigue Progression")
plt.show()
