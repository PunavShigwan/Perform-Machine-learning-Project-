"""
pushup_live_service.py
LivePushupSession — manages webcam capture, pose analysis,
rep counting, form scoring and wrong-exercise detection
in a background thread.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque, Counter

print("📦 pushup_live_service.py loaded")

# =====================================================
# CONFIG
# =====================================================
ELBOW_DOWN              = 90
ELBOW_UP                = 155
SMOOTHING_WINDOW        = 5
FRAME_CONFIRM           = 2
HIP_SAG_THRESHOLD       = 0.08
HEAD_NECK_TOLERANCE     = 25
WRONG_EXERCISE_FRAMES   = 15
WRONG_EXERCISE_COOLDOWN = 45

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# =====================================================
# GEOMETRY
# =====================================================
def _angle(a, b, c):
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    r = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(np.degrees(r))
    return 360 - ang if ang > 180 else ang

def _elbow_angle(lm): return _angle(lm[11], lm[13], lm[15])
def _body_angle(lm):  return _angle(lm[11], lm[23], lm[27])
def _head_angle(lm):  return _angle(lm[0],  lm[11], lm[23])

# =====================================================
# EXERCISE CLASSIFIER
# =====================================================
def _classify_exercise(lm):
    b = _body_angle(lm)
    if 140 <= b <= 210 and lm[15].y > lm[11].y - 0.05:
        return "pushup", f"body={b:.1f}"
    hip_knee = abs(lm[23].y - lm[25].y)
    if b > 130 and hip_knee < 0.25:
        return ("squat" if lm[23].y > lm[25].y - 0.05 else "standing"), f"body={b:.1f}"
    return "unknown", f"body={b:.1f}"

def _wrong_msg(ex):
    return {
        "squat":    "Wrong exercise! Looks like a SQUAT — get into PUSHUP position.",
        "standing": "Wrong exercise! You are STANDING — get into PUSHUP position.",
        "unknown":  "Wrong exercise! Expected PUSHUP position (body horizontal).",
    }.get(ex, "Wrong exercise detected. Please perform a PUSHUP.")

# =====================================================
# PUSHUP CHECK
# =====================================================
def _is_pushup_pos(lm):
    b = _body_angle(lm)
    return (160 <= b <= 200 and lm[15].y > lm[11].y
            and abs(lm[23].y - lm[11].y) < 0.15)

# =====================================================
# FORM ANALYSIS
# =====================================================
def _analyze_form(lm):
    issues = {}; bd = {}
    b = _body_angle(lm); dev = abs(180 - b)
    bd["body_alignment"] = int(np.clip(40 - dev * 2.5, 0, 40))
    if dev > 15:
        issues["body"] = "Hips too high (piking)" if b < 180 else "Hips sagging — keep core tight"

    sh, hp, ak = lm[11], lm[23], lm[27]
    exp = sh.y + (ak.y - sh.y) * ((hp.x - sh.x) / max(abs(ak.x - sh.x), 0.001))
    off = abs(hp.y - exp)
    bd["hip_position"] = int(np.clip(20 - off * 200, 0, 20))
    if off > HIP_SAG_THRESHOLD:
        issues["hip"] = "Hips dropping — engage your core" if hp.y > exp else "Hips raised — lower your hips"

    hd = abs(180 - _head_angle(lm))
    bd["head_alignment"] = int(np.clip(15 - hd * 0.8, 0, 15))
    if hd > HEAD_NECK_TOLERANCE:
        issues["head"] = "Head too high — look down" if lm[0].y < lm[11].y else "Head drooping — neutral neck"

    ratio = abs(lm[15].x - lm[16].x) / max(abs(lm[11].x - lm[12].x), 0.001)
    bd["arm_width"] = int(np.clip(15 - abs(ratio - 1.25) * 20, 0, 15))
    if ratio < 0.8:   issues["arm"] = "Hands too close together"
    elif ratio > 1.8: issues["arm"] = "Hands too wide apart"

    ea = _elbow_angle(lm)
    bd["range_of_motion"] = int(np.clip(10 - max(0, ea - ELBOW_DOWN) * 0.3, 0, 10))
    if ea > ELBOW_DOWN + 15:
        issues["rom"] = "Not going low enough — full range of motion"

    return sum(bd.values()), list(issues.values()), bd

def _form_grade(s):
    if s >= 88: return "EXCELLENT", (0, 220, 0)
    if s >= 72: return "GOOD",      (0, 200, 120)
    if s >= 55: return "FAIR",      (0, 165, 255)
    if s >= 38: return "POOR",      (0, 80, 255)
    return "BAD", (0, 0, 220)

def _rep_rating(s):
    if s >= 85: return "Excellent", 2
    if s >= 70: return "Good",      1
    if s >= 50: return "Poor",      0
    return "Bad", -2

def _fatigue_index(log):
    if len(log) < 2: return 0
    fd = max(0, log[0]["form"] - log[-1]["form"])
    ti = max(0, log[-1]["time"] - log[0]["time"]) * 10
    br = sum(1 for r in log if r["form"] < 50) / len(log) * 100
    return int(np.clip(0.5*fd + 0.3*ti + 0.2*br, 0, 100))

def _fatigue_level(v):
    return "LOW" if v < 30 else ("MODERATE" if v < 60 else "HIGH")

# =====================================================
# DRAWING
# =====================================================
def _draw_panel(frame, score, grade, gc, issues, bd, state, reps):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (300,h), (20,20,20), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "PUSHUP LIVE", (10,28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,200,255), 1)
    cv2.line(frame, (10,35), (290,35), (80,80,80), 1)
    cv2.putText(frame, f"Reps: {reps}", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    sc = (0,255,255) if state=="DOWN" else (200,200,200)
    cv2.putText(frame, f"State: {state}", (10,92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 1)
    cv2.line(frame, (10,100), (290,100), (80,80,80), 1)
    bw = int(270*score/100)
    cv2.rectangle(frame, (10,128), (280,148), (50,50,50), -1)
    cv2.rectangle(frame, (10,128), (10+bw,148), gc, -1)
    cv2.putText(frame, f"{score}%  {grade}", (10,165), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gc, 2)
    cv2.line(frame, (10,173), (290,173), (80,80,80), 1)
    lbs = {"body_alignment":"Body","hip_position":"Hips","head_alignment":"Head",
           "arm_width":"Arms","range_of_motion":"ROM"}
    mx  = {"body_alignment":40,"hip_position":20,"head_alignment":15,"arm_width":15,"range_of_motion":10}
    y = 185
    for k,lb in lbs.items():
        v = bd.get(k,0); m = mx[k]; r = v/m
        col = (0,int(200*r),int(200*(1-r))); bww = int(130*r)
        cv2.putText(frame, lb, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
        cv2.rectangle(frame, (100,y-11), (230,y+2), (50,50,50), -1)
        cv2.rectangle(frame, (100,y-11), (100+bww,y+2), col, -1)
        cv2.putText(frame, f"{v}/{m}", (235,y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
        y += 22
    cv2.line(frame, (10,y), (290,y), (80,80,80), 1); y += 14
    if issues:
        cv2.putText(frame, "FORM ISSUES:", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,80,255), 1); y+=18
        for iss in issues[:3]:
            cv2.putText(frame, f"  {iss[:32]}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,140,255), 1); y+=16
    else:
        cv2.putText(frame, "Great form!", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)

def _draw_alert(frame, issues):
    if not issues: return
    h, w = frame.shape[:2]
    txt  = issues[0]; font = cv2.FONT_HERSHEY_DUPLEX
    (tw,th),_ = cv2.getTextSize(txt, font, 0.72, 2)
    x=(w-tw)//2; y=h-60
    cv2.rectangle(frame,(x-10,y-th-8),(x+tw+10,y+8),(0,0,180),-1)
    cv2.rectangle(frame,(x-10,y-th-8),(x+tw+10,y+8),(0,0,255),2)
    cv2.putText(frame, txt, (x,y), font, 0.72, (255,255,255), 2)

def _draw_wrong_banner(frame, ex_type):
    h, w = frame.shape[:2]
    msg  = f"WRONG EXERCISE: {ex_type.upper()} — Please do PUSHUPS!"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw,th),_ = cv2.getTextSize(msg, font, 0.68, 2)
    cv2.rectangle(frame,(0,0),(w,50),(0,0,180),-1)
    cv2.rectangle(frame,(0,0),(w,50),(0,0,255),3)
    cv2.putText(frame, msg, (max(10,(w-tw)//2),34), font, 0.68, (255,255,255), 2)

def _draw_timer(frame, sec):
    h,w = frame.shape[:2]
    m,s = int(sec)//60, int(sec)%60
    cv2.putText(frame, f"Time: {m:02d}:{s:02d}", (w-165,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)

# =====================================================
# LIVE SESSION
# =====================================================
class LivePushupSession:
    def __init__(self, camera_index: int = 0):
        self.camera_index   = camera_index
        self._lock          = threading.Lock()
        self._running       = False
        self._thread        = None
        self._latest_frame  = None   # JPEG bytes

        # Public stats
        self.pushups              = 0
        self.rep_log              = []
        self.wrong_events         = []
        self.current_state        = "UP"
        self.current_form         = 100
        self.current_grade        = "EXCELLENT"
        self.current_issues       = []
        self.current_fatigue      = 0
        self.current_fatigue_lvl  = "LOW"
        self.wrong_warning_active = False
        self.wrong_warning_msg    = None
        self.session_start_time   = None

        print(f"  ℹ️   LivePushupSession created (camera={camera_index})")

    # --------------------------------------------------
    def start(self):
        if self._running:
            print("  ⚠️   Session already running")
            return {"status": "already_running"}
        print(f"\n  ▶   Starting live session (camera {self.camera_index})...")
        self._running           = True
        self.session_start_time = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("  ✅  Background thread started")
        return {"status": "started"}

    # --------------------------------------------------
    def stop(self):
        if not self._running:
            return {"status": "not_running"}
        print("\n  ⏹   Stopping session...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("  ✅  Session stopped")
        return self.get_stats(final=True)

    # --------------------------------------------------
    def get_frame(self):
        with self._lock:
            return self._latest_frame

    # --------------------------------------------------
    def get_stats(self, final=False):
        elapsed = time.time() - self.session_start_time if self.session_start_time else 0
        issue_freq = Counter(i for r in self.rep_log for i in r.get("form_issues",[]))
        top_issues = [{"issue":k,"occurrences":v} for k,v in issue_freq.most_common(5)]
        bad_reps   = sum(1 for r in self.rep_log if not r["is_good_form"])
        o_form     = int(np.mean([r["form"] for r in self.rep_log])) if self.rep_log else 0
        o_grade,_  = _form_grade(o_form)

        return {
            "status":               "final" if final else "live",
            "pushup_count":         self.pushups,
            "session_duration_sec": round(elapsed, 1),
            "current_state":        self.current_state,
            "current_form_score":   self.current_form,
            "current_form_grade":   self.current_grade,
            "current_form_issues":  self.current_issues,
            "overall_form_score":   o_form,
            "overall_form_grade":   o_grade,
            "bad_form_reps":        bad_reps,
            "top_form_issues":      top_issues,
            "fatigue":              self.current_fatigue,
            "fatigue_level":        self.current_fatigue_lvl,
            "wrong_exercise": {
                "active":          self.wrong_warning_active,
                "warning_message": self.wrong_warning_msg,
                "total_warnings":  len(self.wrong_events),
                "events":          self.wrong_events,
            },
            "reps":       self.rep_log,
            "stream_url": "http://localhost:8000/pushup/live/stream",
        }

    # --------------------------------------------------
    def _run_loop(self):
        print(f"  🎥  Opening camera {self.camera_index}...")
        # On Windows, CAP_DSHOW often works better for external USB cameras
        # It prevents long initialization delays and fixes index issues
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"  ⚠️   CAP_DSHOW failed, trying default backend for camera {self.camera_index}...")
            cap = cv2.VideoCapture(self.camera_index)
            
        if not cap.isOpened():
            print(f"  ❌  CRITICAL: Cannot open camera {self.camera_index} with any backend")
            self._running = False
            return
            
        # Optional: Set resolution to ensure camera is active
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"  ✅  Camera {self.camera_index} initialized successfully — {w}x{h} @ {fps:.0f}fps")

        pose = mp_pose.Pose(static_image_mode=False,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

        elbow_buf        = deque(maxlen=SMOOTHING_WINDOW)
        state            = "UP"
        predicted_max    = 5
        down_frames      = 0;  up_frames = 0
        down_scores      = [];  down_bds = [];  down_issues = []
        valid_down       = False
        last_rep_time    = None
        wrong_frames     = 0;  wrong_cd = 0
        cur_score        = 100
        cur_grade        = "EXCELLENT"
        cur_gc           = (0,220,0)
        cur_issues       = []
        cur_bd           = {"body_alignment":40,"hip_position":20,
                            "head_alignment":15,"arm_width":15,"range_of_motion":10}
        frame_count      = 0
        last_wrong_type  = "unknown"

        print("  🔄  Processing loop running...")

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("  ⚠️   Camera read failed — retrying...")
                time.sleep(0.05); continue

            frame_count += 1
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                # Wrong-exercise detection
                ex_type, ex_reason = _classify_exercise(lm)
                if ex_type != "pushup":
                    wrong_frames += 1
                    last_wrong_type = ex_type
                    if wrong_frames >= WRONG_EXERCISE_FRAMES and wrong_cd <= 0:
                        msg = _wrong_msg(ex_type)
                        self.wrong_events.append({
                            "frame":    frame_count,
                            "exercise": ex_type,
                            "message":  msg,
                            "reason":   ex_reason,
                            "time_sec": round(time.time() - self.session_start_time, 1),
                        })
                        wrong_cd                   = WRONG_EXERCISE_COOLDOWN
                        self.wrong_warning_active  = True
                        self.wrong_warning_msg     = msg
                        print(f"  ⚠️   WRONG EXERCISE @ frame {frame_count} — {ex_type} ({ex_reason})")
                        print(f"       MSG: {msg}")
                else:
                    wrong_frames = max(0, wrong_frames - 1)
                    if wrong_frames == 0:
                        self.wrong_warning_active = False
                if wrong_cd > 0: wrong_cd -= 1

                # Elbow + form
                elbow_buf.append(_elbow_angle(lm))
                smooth_elbow = np.mean(elbow_buf)
                score, issues, bd = _analyze_form(lm)
                cur_score     = int(0.7 * cur_score + 0.3 * score)
                cur_grade, cur_gc = _form_grade(cur_score)
                cur_issues    = issues
                cur_bd        = bd

                # Rep state machine
                if smooth_elbow < ELBOW_DOWN:   down_frames += 1; up_frames = 0
                elif smooth_elbow > ELBOW_UP:   up_frames += 1;   down_frames = 0

                if state == "UP" and down_frames >= FRAME_CONFIRM:
                    state = "DOWN"; down_scores=[]; down_bds=[]; down_issues=[]
                    valid_down = _is_pushup_pos(lm)
                    print(f"  ↓  Rep starting @ frame {frame_count}  valid={valid_down}")

                if state == "DOWN":
                    down_scores.append(score); down_bds.append(bd); down_issues.extend(issues)

                if state == "DOWN" and up_frames >= FRAME_CONFIRM:
                    if valid_down:
                        self.pushups += 1
                        now = time.time()
                        rep_time = round(now - last_rep_time, 2) if last_rep_time else 1.0
                        last_rep_time = now
                        rep_form = int((np.mean(down_scores) + score) / 2)
                        rating, delta = _rep_rating(rep_form)
                        predicted_max = max(5, predicted_max + delta)
                        avg_bd = {k: int(np.mean([d[k] for d in down_bds])) for k in down_bds[0]} if down_bds else {}
                        uniq_issues = list(dict.fromkeys(down_issues))
                        self.rep_log.append({
                            "rep": self.pushups, "form": rep_form,
                            "form_grade": rating, "rating": rating, "time": rep_time,
                            "form_breakdown": avg_bd, "form_issues": uniq_issues,
                            "is_good_form": rep_form >= 70,
                        })
                        print(f"  ✅  Rep {self.pushups} — form={rep_form}% ({rating})  "
                              f"issues={uniq_issues or 'none'}")
                    else:
                        print(f"  ⚠️   Movement detected but invalid pushup pos — NOT counted")
                    state = "UP"

                fatigue = _fatigue_index(self.rep_log)
                self.current_state       = state
                self.current_form        = cur_score
                self.current_grade       = cur_grade
                self.current_issues      = cur_issues
                self.current_fatigue     = fatigue
                self.current_fatigue_lvl = _fatigue_level(fatigue)

                # Draw
                mp_draw.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(200,200,200), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(100,255,100), thickness=2)
                )
                _draw_panel(frame, cur_score, cur_grade, cur_gc, cur_issues, cur_bd, state, self.pushups)
                crit = [i for i in cur_issues if any(kw in i.lower() for kw in ["sag","hips","core","range"])]
                _draw_alert(frame, crit)
                if self.wrong_warning_active and wrong_cd > WRONG_EXERCISE_COOLDOWN - 60:
                    _draw_wrong_banner(frame, last_wrong_type)
                _draw_timer(frame, time.time() - self.session_start_time)
                cv2.putText(frame, f"Fatigue: {fatigue}% ({_fatigue_level(fatigue)})",
                            (w-340,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._latest_frame = jpeg.tobytes()

        cap.release()
        pose.close()
        print(f"  ✅  Camera released | frames={frame_count} | "
              f"pushups={self.pushups} | wrong_events={len(self.wrong_events)}")