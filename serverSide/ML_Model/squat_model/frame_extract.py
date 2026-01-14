import cv2
import numpy as np
import mediapipe as mp
import joblib

# =====================================================
# CONFIG
# =====================================================
VIDEO_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\sample_videos\input1.mp4"
MODEL_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models\best_model.pkl"
LABEL_MAP_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models\label_map.pkl"

FRONT_VIEW_Z_THRESHOLD = 0.12

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(MODEL_PATH)
LABEL_MAP = joblib.load(LABEL_MAP_PATH)

# =====================================================
# MEDIAPIPE INIT
# =====================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# UTILS
# =====================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def draw_2d_line(frame, p1, p2, color):
    h, w = frame.shape[:2]
    cv2.line(frame,
             (int(p1[0]*w), int(p1[1]*h)),
             (int(p2[0]*w), int(p2[1]*h)),
             color, 4)


def draw_joint(frame, p, color):
    h, w = frame.shape[:2]
    cv2.circle(frame, (int(p[0]*w), int(p[1]*h)), 6, color, -1)


def project_3d(p, z, scale=0.6):
    return [p[0] + z * scale, p[1] - z * scale]


def evaluate_form(phase, knee, hip, shoulder, elbow):
    issues = []
    if phase in ["P1_first_pull", "P3_second_pull"]:
        if knee < 130 or knee > 170:
            issues.append("KNEE")
        if hip < 140:
            issues.append("HIP")
    if phase in ["P4_catch", "P6_dip"]:
        if shoulder < 150:
            issues.append("SHOULDER")
        if elbow < 160:
            issues.append("ELBOW")
    return issues


def calculate_score(issues):
    score = 100
    for j in ["KNEE", "HIP", "SHOULDER", "ELBOW"]:
        if j in issues:
            score -= 25
    return max(score, 0)

# =====================================================
# VIDEO STREAM
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)
prev_knee_angle, prev_hip_y = None, None
smooth_score = 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # ===============================
        # LANDMARKS (LEFT SIDE)
        # ===============================
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        front_view = abs(ls.z - rs.z) < FRONT_VIEW_Z_THRESHOLD

        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        shoulder = [ls.x, ls.y]
        elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # ===============================
        # 3D PROJECTION IF FRONT VIEW
        # ===============================
        if front_view:
            shoulder = project_3d(shoulder, ls.z)
            hip = project_3d(hip, lm[mp_pose.PoseLandmark.LEFT_HIP.value].z)
            knee = project_3d(knee, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].z)
            ankle = project_3d(ankle, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].z)
            elbow = project_3d(elbow, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].z)
            wrist = project_3d(wrist, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].z)

        # ===============================
        # ANGLES & MODEL
        # ===============================
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        shoulder_angle = calculate_angle(elbow, shoulder, hip)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        knee_vel = 0 if prev_knee_angle is None else knee_angle - prev_knee_angle
        hip_vel = 0 if prev_hip_y is None else hip[1] - prev_hip_y
        prev_knee_angle, prev_hip_y = knee_angle, hip[1]

        phase = LABEL_MAP[model.predict([[knee_angle, hip_vel, knee_vel]])[0]]

        issues = evaluate_form(phase, knee_angle, hip_angle, shoulder_angle, elbow_angle)
        score = calculate_score(issues)
        smooth_score = int(0.85*smooth_score + 0.15*score)

        color = (0,255,0) if smooth_score >= 75 else (0,0,255)

        # ===============================
        # DRAW SKELETON
        # ===============================
        for p1, p2 in [(shoulder, hip), (hip, knee), (knee, ankle),
                       (shoulder, elbow), (elbow, wrist)]:
            draw_2d_line(frame, p1, p2, color)

        for p in [shoulder, hip, knee, ankle, elbow, wrist]:
            draw_joint(frame, p, color)

        # ===============================
        # UI
        # ===============================
        cv2.rectangle(frame, (10,10), (420,140), (0,0,0), -1)
        cv2.putText(frame, f"View: {'3D FRONT' if front_view else '2D SIDE'}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Phase: {phase}",
                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Score: {smooth_score}/100",
                    (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.imshow("Clean & Jerk – Adaptive 2D / 3D Coach", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Finished")
