"""
pipeline.py
===========
Full ML pipeline for DIP FORM ANALYSIS.
Replaces all deep-learning models with classical Machine Learning estimators.

Stages
──────
  1. Video extraction (frames + MediaPipe keypoints)
  2. Feature engineering  (statistical aggregates over the time axis)
  3. Preprocessing        (StandardScaler + optional MinMaxScaler / PCA)
  4. Train / Val / Test split (stratified)
  5. Train 14 ML classifiers with cross-validation
  6. Evaluate on test set — accuracy, classification report, confusion matrix
  7. Persist every trained model as a .pkl file
  8. Ensemble soft-vote on test set

Usage:
    python pipeline.py                      # full run
    python pipeline.py --skip-preprocess    # reuse .npy feature files
    python pipeline.py --model random_forest
"""

print("=" * 60)
print("  DIP FORM ANALYZER (ML Edition) — starting up ...")
print("  Loading libraries, please wait.")
print("=" * 60)

import os, sys, argparse, yaml, warnings, math

print("[IMPORT] os, sys, argparse, yaml, math ... OK")

try:
    import numpy as np
    print("[IMPORT] numpy ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}  →  pip install numpy"); sys.exit(1)

try:
    import cv2
    print(f"[IMPORT] opencv  {cv2.__version__} ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}  →  pip install opencv-python"); sys.exit(1)

try:
    import mediapipe as mp
    print("[IMPORT] mediapipe ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}  →  pip install mediapipe"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("[IMPORT] matplotlib, seaborn ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}  →  pip install matplotlib seaborn"); sys.exit(1)

try:
    import pandas as pd
    print("[IMPORT] pandas ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}  →  pip install pandas"); sys.exit(1)

try:
    from tqdm import tqdm
    print("[IMPORT] tqdm ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}  →  pip install tqdm"); sys.exit(1)

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, f1_score
    )
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    print("[IMPORT] scikit-learn ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}  →  pip install scikit-learn"); sys.exit(1)

try:
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    from scipy.stats import skew, kurtosis
    print("[IMPORT] scipy ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}  →  pip install scipy"); sys.exit(1)

try:
    import pickle
    print("[IMPORT] pickle (stdlib) ... OK")
except ImportError as e:
    print(f"[IMPORT] ERROR: {e}"); sys.exit(1)

print("[IMPORT] Loading models.py ...")
try:
    from models import (
        BUILDERS, ALL_MODELS, ENSEMBLE_MEMBERS,
        save_model, load_model, ensemble_predict
    )
    print(f"[IMPORT] models.py ... OK  ({len(ALL_MODELS)} models: {ALL_MODELS})")
except Exception as e:
    print(f"[IMPORT] ERROR loading models.py — {e}")
    import traceback; traceback.print_exc(); sys.exit(1)

warnings.filterwarnings("ignore")
import logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)
print("[IMPORT] All libraries loaded.\n")


# ══════════════════════════════════════════════════════════════════════════════
# LANDMARK INDICES  (MediaPipe Pose — 33 landmarks, 99 floats)
# ══════════════════════════════════════════════════════════════════════════════
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
NOSE                   = 0


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def load_cfg():
    print("[CONFIG] Loading config.yaml ...")
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"[CONFIG] Project       : {cfg['project']}")
    print(f"[CONFIG] Classes       : {cfg['classes']}")
    print(f"[CONFIG] Num classes   : {cfg['num_classes']}")
    print(f"[CONFIG] Max frames    : {cfg['preprocessing']['max_frames']}")
    return cfg


def make_dirs(cfg):
    print("\n[DIRS] Creating output directories ...")
    for d in [cfg["data"]["processed_dir"],
              cfg["training"]["checkpoint_dir"],
              cfg["training"]["results_dir"],
              cfg["training"]["log_dir"]]:
        os.makedirs(d, exist_ok=True)
        print(f"[DIRS]   ✔ {d}")


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))

def kp_xyz(kp_flat, idx):
    return kp_flat[idx * 3: idx * 3 + 3]

def compute_angles(kp_seq: np.ndarray) -> np.ndarray:
    """
    kp_seq : (T, 99)  →  angles : (T, 6)
    Angles: L-elbow, R-elbow, L-shoulder, R-shoulder, torso-lean, hip-knee
    """
    T = kp_seq.shape[0]
    angles = np.zeros((T, 6), dtype=np.float32)
    for t in range(T):
        kp = kp_seq[t]
        ls, rs  = kp_xyz(kp, L_SHOULDER), kp_xyz(kp, R_SHOULDER)
        le, re  = kp_xyz(kp, L_ELBOW),    kp_xyz(kp, R_ELBOW)
        lw, rw  = kp_xyz(kp, L_WRIST),    kp_xyz(kp, R_WRIST)
        lh, rh  = kp_xyz(kp, L_HIP),      kp_xyz(kp, R_HIP)
        lk, rk  = kp_xyz(kp, L_KNEE),     kp_xyz(kp, R_KNEE)
        nose    = kp_xyz(kp, NOSE)
        mid_hip = (lh + rh) / 2
        mid_sho = (ls + rs) / 2
        angles[t, 0] = angle_3pts(ls, le, lw)
        angles[t, 1] = angle_3pts(rs, re, rw)
        angles[t, 2] = angle_3pts(le, ls, lh)
        angles[t, 3] = angle_3pts(re, rs, rh)
        angles[t, 4] = angle_3pts(nose, mid_sho, mid_hip)
        angles[t, 5] = angle_3pts(mid_sho, mid_hip, (lk + rk) / 2)
    return angles


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Collapses (T, D) time-series → fixed-length feature vector per sample.
# Statistics: mean, std, min, max, range, median, IQR, skewness, kurtosis,
#             velocity-mean, velocity-std  (per channel)
# ══════════════════════════════════════════════════════════════════════════════

def _stat_features(seq: np.ndarray) -> np.ndarray:
    """seq : (T, D)  →  flat feature vector of length D * 11"""
    vel = np.diff(seq, axis=0)                       # (T-1, D)
    feats = np.concatenate([
        seq.mean(axis=0),
        seq.std(axis=0),
        seq.min(axis=0),
        seq.max(axis=0),
        seq.max(axis=0) - seq.min(axis=0),           # range
        np.median(seq, axis=0),
        np.percentile(seq, 75, axis=0) - np.percentile(seq, 25, axis=0),  # IQR
        skew(seq, axis=0),
        kurtosis(seq, axis=0),
        vel.mean(axis=0),
        vel.std(axis=0),
    ])
    return feats.astype(np.float32)


def extract_features(kp_seq: np.ndarray) -> np.ndarray:
    """
    kp_seq : (T, 99)  keypoints
    Returns : 1-D feature vector:
       - Statistical features from keypoints   (99 * 11 = 1 089 dims)
       - Statistical features from 6 angles    ( 6 * 11 =   66 dims)
       - Biomechanical summary scalars         (          8  dims)
    Total ≈ 1 163 dims
    """
    angles = compute_angles(kp_seq)            # (T, 6)

    kp_feats  = _stat_features(kp_seq)         # 99 * 11 = 1 089
    ang_feats = _stat_features(angles)         #  6 * 11 =    66

    # ── Biomechanical summary
    elbow_avg = (angles[:, 0] + angles[:, 1]) / 2.0
    smooth    = uniform_filter1d(elbow_avg, size=3)
    sym_diff  = np.mean(np.abs(angles[:, 0] - angles[:, 1]))
    bio = np.array([
        smooth.min(),                                  # deepest elbow angle
        smooth.max(),                                  # most extended
        smooth.max() - smooth.min(),                   # ROM
        np.mean(angles[:, 2:4]),                       # avg shoulder angle
        np.mean(angles[:, 4]),                         # avg torso lean
        sym_diff,                                      # L/R asymmetry
        float(np.sum(np.diff(np.sign(np.diff(smooth))) < 0)),  # rep proxy
        float(np.std(smooth)),                         # elbow variability
    ], dtype=np.float32)

    return np.concatenate([kp_feats, ang_feats, bio])


# ══════════════════════════════════════════════════════════════════════════════
# REP COUNTING  (unchanged — pure numpy, no DL dependency)
# ══════════════════════════════════════════════════════════════════════════════

def count_reps_and_timing(kp_seq: np.ndarray, fps: float = 30.0):
    print("[REP COUNT] Computing elbow angles for rep detection ...")
    angles    = compute_angles(kp_seq)
    elbow_avg = (angles[:, 0] + angles[:, 1]) / 2.0
    smooth    = uniform_filter1d(elbow_avg, size=3)
    print(f"[REP COUNT] Smoothed elbow angle — min:{smooth.min():.1f}°  max:{smooth.max():.1f}°")

    bottoms, _ = find_peaks(-smooth, prominence=10, distance=5)
    tops,    _ = find_peaks( smooth, prominence=10, distance=5)
    rep_count  = len(bottoms)
    print(f"[REP COUNT] Bottom frames : {bottoms.tolist()}")
    print(f"[REP COUNT] Top frames    : {tops.tolist()}")
    print(f"[REP COUNT] Total reps    : {rep_count}")

    durations = []
    if len(bottoms) >= 2:
        for i in range(1, len(bottoms)):
            d = (bottoms[i] - bottoms[i - 1]) / fps
            durations.append(d)
            print(f"[REP COUNT]   Rep {i}: {d:.2f}s")
    avg_dur = float(np.mean(durations)) if durations else 0.0
    print(f"[REP COUNT] Avg rep duration: {avg_dur:.2f}s")

    thr_bot = np.percentile(smooth, 25)
    thr_top = np.percentile(smooth, 75)
    phases  = []
    for i, v in enumerate(smooth):
        if v <= thr_bot:
            phases.append("bottom")
        elif v >= thr_top:
            phases.append("top")
        elif i > 0 and smooth[i] < smooth[i - 1]:
            phases.append("descending")
        else:
            phases.append("ascending")
    phase_counts = {p: phases.count(p) for p in ["top", "descending", "bottom", "ascending"]}
    print(f"[REP COUNT] Phase distribution: {phase_counts}")
    return rep_count, avg_dur, phases


# ══════════════════════════════════════════════════════════════════════════════
# FORM SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_form_score(kp_seq: np.ndarray, cfg: dict):
    print("[FORM SCORE] Computing joint angles ...")
    thr    = cfg["thresholds"]
    angles = compute_angles(kp_seq)
    issues, deductions = {}, 0

    avg_shoulder = np.mean(angles[:, 2:4])
    print(f"[FORM SCORE] Avg shoulder angle   : {avg_shoulder:.1f}°  (limit: {thr['elbow_flare_angle']}°)")
    if avg_shoulder > thr["elbow_flare_angle"]:
        issues["elbow_flare"] = f"Shoulder angle {avg_shoulder:.1f}° > {thr['elbow_flare_angle']}°"
        deductions += 25
        print("[FORM SCORE]   ⚠️  ELBOW FLARE    → -25 pts")

    min_elbow = np.min((angles[:, 0] + angles[:, 1]) / 2)
    print(f"[FORM SCORE] Min elbow angle (depth): {min_elbow:.1f}°  (limit: {thr['min_depth_angle']}°)")
    if min_elbow > thr["min_depth_angle"]:
        issues["shallow_depth"] = f"Min elbow {min_elbow:.1f}° > {thr['min_depth_angle']}° (too shallow)"
        deductions += 25
        print("[FORM SCORE]   ⚠️  SHALLOW DEPTH  → -25 pts")

    avg_torso = np.mean(angles[:, 4])
    print(f"[FORM SCORE] Avg torso lean        : {avg_torso:.1f}°  (limit: {thr['torso_lean_angle']}°)")
    if avg_torso > thr["torso_lean_angle"]:
        issues["forward_lean"] = f"Torso lean {avg_torso:.1f}° > {thr['torso_lean_angle']}°"
        deductions += 20
        print("[FORM SCORE]   ⚠️  FORWARD LEAN   → -20 pts")

    sym_diff = np.mean(np.abs(angles[:, 0] - angles[:, 1]))
    print(f"[FORM SCORE] L/R elbow symmetry   : {sym_diff:.1f}°  (limit: 15°)")
    if sym_diff > 15:
        issues["asymmetry"] = f"L/R diff {sym_diff:.1f}° (uneven push)"
        deductions += 15
        print("[FORM SCORE]   ⚠️  ASYMMETRY      → -15 pts")

    score = max(0, 100 - deductions)
    print(f"[FORM SCORE] Final score: {score}/100  (deductions: {deductions})")
    return score, issues


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC INCORRECT FORM
# ══════════════════════════════════════════════════════════════════════════════

def synth_elbow_flare(kp_seq):
    print("[SYNTH] elbow_flare — widening elbows by 0.08 units ...")
    s = kp_seq.copy()
    for idx in [L_ELBOW, R_ELBOW]:
        s[:, idx * 3] += 0.08 * (1 if idx == R_ELBOW else -1)
    return s

def synth_shallow_depth(kp_seq):
    T = kp_seq.shape[0]
    print(f"[SYNTH] shallow_depth — freezing frames {T//3}:{2*T//3} to top pose ...")
    s = kp_seq.copy()
    s[T // 3: 2 * T // 3] = s[0]
    return s

def synth_forward_lean(kp_seq):
    print("[SYNTH] forward_lean — shifting shoulder/hip z by +0.15 ...")
    s = kp_seq.copy()
    for idx in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]:
        s[:, idx * 3 + 2] += 0.15
    return s


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def is_valid(path):
    cap = cv2.VideoCapture(path)
    ok  = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 5
    cap.release()
    if not ok:
        print(f"[VALIDATE] ✘ Invalid/short: {os.path.basename(path)}")
    return ok

def extract_keypoints(path: str, T: int) -> np.ndarray:
    print(f"[KEYPOINTS] MediaPipe Pose on {T} frames: {os.path.basename(path)}")
    pose    = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
    cap     = cv2.VideoCapture(path)
    total   = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, total - 1, T, dtype=int)
    seq, detected, missed = [], 0, 0
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        kp = np.zeros(99, dtype=np.float32)
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                for j, lm in enumerate(res.pose_landmarks.landmark):
                    kp[j * 3], kp[j * 3 + 1], kp[j * 3 + 2] = lm.x, lm.y, lm.z
                detected += 1
            else:
                missed += 1
        seq.append(kp)
    cap.release(); pose.close()
    print(f"[KEYPOINTS]   detected: {detected}/{T}  missed: {missed}")
    return np.array(seq, dtype=np.float32)

def flip_kp(kp: np.ndarray) -> np.ndarray:
    k = kp.copy()
    k[:, 0::3] = 1.0 - k[:, 0::3]
    return k


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (fit on train, apply to val/test)
# ══════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """
    Chains:  StandardScaler → (optional MinMaxScaler) → (optional PCA)
    The fitted object is pickled alongside the models so predict.py can reuse it.
    """
    def __init__(self, use_minmax: bool = True, n_pca_components=None):
        self.scaler     = StandardScaler()
        self.minmax     = MinMaxScaler(feature_range=(0, 1)) if use_minmax else None
        self.pca        = PCA(n_components=n_pca_components, random_state=42) if n_pca_components else None
        self._fitted    = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        print(f"[PREPROC] StandardScaler fit+transform  input shape: {X.shape}")
        X = self.scaler.fit_transform(X)
        if self.minmax is not None:
            print(f"[PREPROC] MinMaxScaler fit+transform")
            X = self.minmax.fit_transform(X)
        if self.pca is not None:
            print(f"[PREPROC] PCA fit+transform  n_components={self.pca.n_components}")
            X = self.pca.fit_transform(X)
            var = np.sum(self.pca.explained_variance_ratio_) * 100
            print(f"[PREPROC] PCA explained variance: {var:.1f}%  shape→{X.shape}")
        self._fitted = True
        return X.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted yet — call fit_transform first.")
        X = self.scaler.transform(X)
        if self.minmax is not None:
            X = self.minmax.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X.astype(np.float32)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[PREPROC] Preprocessor saved → {path}")

    @staticmethod
    def load(path: str) -> "Preprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[PREPROC] Preprocessor loaded ← {path}")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# CLASS MAP
# ══════════════════════════════════════════════════════════════════════════════

CLASS_MAP = {"correct": 0, "elbow_flare": 1, "shallow_depth": 2, "forward_lean": 3}


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESS — extract keypoints, engineer features, save .npy
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(cfg: dict):
    raw_dir = cfg["data"]["raw_dir"]
    proc    = cfg["data"]["processed_dir"]
    T       = cfg["preprocessing"]["max_frames"]

    if not os.path.exists(raw_dir):
        print(f"[PREPROCESS] ✘ raw_dir not found → {raw_dir}"); sys.exit(1)

    videos = [f for f in os.listdir(raw_dir) if f.lower().endswith(".mp4")]
    print(f"\n{'═'*60}")
    print(f"  STEP 1/4 — DATA EXTRACTION & FEATURE ENGINEERING")
    print(f"{'═'*60}")
    print(f"[PREPROCESS] Raw dir       : {raw_dir}")
    print(f"[PREPROCESS] Videos found  : {len(videos)}")
    print(f"[PREPROCESS] Samples/video : 5  (2 correct + 3 synthetic)")

    X_list, y_list = [], []
    skipped = 0

    for v_idx, vid in enumerate(videos):
        path = os.path.join(raw_dir, vid)
        print(f"\n[PREPROCESS] ── Video {v_idx+1}/{len(videos)}: {vid} ──")
        if not is_valid(path):
            skipped += 1; continue

        try:
            kp = extract_keypoints(path, T)
        except Exception as e:
            print(f"[PREPROCESS]   ✘ SKIP {vid}: {e}"); skipped += 1; continue

        def add(k, label):
            feat = extract_features(k)
            X_list.append(feat)
            y_list.append(label)

        print(f"[PREPROCESS]   [1/5] CORRECT (original)...")
        add(kp, CLASS_MAP["correct"])
        print(f"[PREPROCESS]   [2/5] CORRECT (flipped)...")
        add(flip_kp(kp), CLASS_MAP["correct"])
        print(f"[PREPROCESS]   [3/5] ELBOW_FLARE...")
        add(synth_elbow_flare(kp), CLASS_MAP["elbow_flare"])
        print(f"[PREPROCESS]   [4/5] SHALLOW_DEPTH...")
        add(synth_shallow_depth(kp), CLASS_MAP["shallow_depth"])
        print(f"[PREPROCESS]   [5/5] FORWARD_LEAN...")
        add(synth_forward_lean(kp), CLASS_MAP["forward_lean"])

        print(f"[PREPROCESS]   Running total: {len(y_list)} samples")

    print(f"\n[PREPROCESS] Extraction complete.")
    print(f"[PREPROCESS] Total samples : {len(y_list)}  |  Skipped: {skipped}")
    print(f"[PREPROCESS] Class counts  : { {cls:y_list.count(i) for cls,i in CLASS_MAP.items()} }")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)
    print(f"[PREPROCESS] Feature matrix shape: {X.shape}")

    # ── Train / Val / Test split ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 2/4 — TRAIN / VAL / TEST SPLIT")
    print(f"{'═'*60}")
    tr        = cfg["training"]
    test_size = tr["val_split"] + tr["test_split"]
    print(f"[SPLIT] 70% train / 15% val / 15% test  (stratified, seed=42)")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    test_ratio = tr["test_split"] / test_size
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=test_ratio, stratify=y_tmp, random_state=42
    )
    print(f"[SPLIT] Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}")

    # ── Preprocessing (fit on train only) ────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 3/4 — NORMALIZATION & STANDARDIZATION")
    print(f"{'═'*60}")
    pp_cfg = cfg.get("preprocessing_steps", {})
    preprocessor = Preprocessor(
        use_minmax       = pp_cfg.get("use_minmax", True),
        n_pca_components = pp_cfg.get("pca_components", None),
    )
    X_tr_sc  = preprocessor.fit_transform(X_tr)
    X_val_sc = preprocessor.transform(X_val)
    X_te_sc  = preprocessor.transform(X_te)
    print(f"[PREPROC] Train scaled: {X_tr_sc.shape}")
    print(f"[PREPROC] Val scaled  : {X_val_sc.shape}")
    print(f"[PREPROC] Test scaled : {X_te_sc.shape}")

    # ── Save arrays & preprocessor ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  STEP 4/4 — SAVING PROCESSED ARRAYS")
    print(f"{'═'*60}")
    splits = [
        ("train", X_tr_sc,  y_tr),
        ("val",   X_val_sc, y_val),
        ("test",  X_te_sc,  y_te),
    ]
    for split, X_s, y_s in splits:
        np.save(f"{proc}/{split}_X.npy", X_s)
        np.save(f"{proc}/{split}_y.npy", y_s)
        print(f"[SAVE]   ✔ {split}_X.npy  {X_s.shape}")
        print(f"[SAVE]   ✔ {split}_y.npy  {y_s.shape}")

    pp_path = os.path.join(proc, "preprocessor.pkl")
    preprocessor.save(pp_path)

    return X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te


def load_processed(cfg: dict):
    p = cfg["data"]["processed_dir"]
    print(f"[LOAD] Reading .npy files from: {p}/")
    arrays = []
    for s in ["train", "val", "test"]:
        X = np.load(f"{p}/{s}_X.npy")
        y = np.load(f"{p}/{s}_y.npy")
        print(f"[LOAD]   ✔ {s}_X {X.shape}  {s}_y {y.shape}")
        arrays += [X, y]
    return tuple(arrays)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_one(name: str, cfg: dict,
              X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
    print(f"\n{'═'*60}")
    print(f"  TRAINING: {name.upper()}")
    print(f"{'═'*60}")
    ckpt = cfg["training"]["checkpoint_dir"]
    logs = cfg["training"]["log_dir"]

    model = BUILDERS[name](cfg)
    if model is None:
        print(f"[TRAIN] ✘ {name} skipped (optional library not installed).")
        return None

    print(f"[TRAIN] Fitting on {len(y_tr)} samples ...")

    # ── Cross-validation on training set (5-fold, stratified) ────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"[TRAIN] 5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  "
          f"(per-fold: {[f'{s:.3f}' for s in cv_scores]})")
    cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"[TRAIN] 5-fold CV F1-macro: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── Overfit / underfit diagnosis ─────────────────────────────────────────
    if cv_scores.mean() < 0.50:
        print(f"[TRAIN] ⚠️  LOW CV SCORE ({cv_scores.mean():.2f}) — "
              f"possible underfitting. Consider more features or richer model.")
    elif cv_scores.std() > 0.08:
        print(f"[TRAIN] ⚠️  HIGH CV STD ({cv_scores.std():.2f}) — "
              f"possible overfitting. Consider regularisation or more data.")

    # ── Full fit on combined train+val for final checkpoint ──────────────────
    X_full = np.concatenate([X_tr, X_val], axis=0)
    y_full = np.concatenate([y_tr, y_val], axis=0)
    model.fit(X_full, y_full)
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc   = accuracy_score(y_val, model.predict(X_val))
    gap       = train_acc - val_acc
    print(f"[TRAIN] Train accuracy  : {train_acc:.4f}")
    print(f"[TRAIN] Val   accuracy  : {val_acc:.4f}  (gap: {gap:.4f})")
    if gap > 0.15:
        print(f"[TRAIN] ⚠️  Overfitting detected (gap={gap:.2f}). "
              f"Consider stronger regularization.")
    elif val_acc < 0.55:
        print(f"[TRAIN] ⚠️  Underfitting detected (val_acc={val_acc:.2f}). "
              f"Consider more estimators or better features.")

    save_model(model, name, ckpt)

    # ── Log to CSV ────────────────────────────────────────────────────────────
    log_path = os.path.join(logs, f"{name}_cv.csv")
    pd.DataFrame({
        "fold": range(1, 6),
        "accuracy": cv_scores,
        "f1_macro": cv_f1,
    }).to_csv(log_path, index=False)
    print(f"[TRAIN] CV log saved → {log_path}")

    return {
        "model": name,
        "cv_acc_mean": round(cv_scores.mean(), 4),
        "cv_acc_std":  round(cv_scores.std(), 4),
        "cv_f1_mean":  round(cv_f1.mean(), 4),
        "train_acc":   round(train_acc, 4),
        "val_acc":     round(val_acc, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE ONE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_one(name: str, cfg: dict, X_te: np.ndarray, y_te: np.ndarray):
    print(f"\n[EVAL] ── Evaluating: {name} ──")
    ckpt    = cfg["training"]["checkpoint_dir"]
    res     = cfg["training"]["results_dir"]
    classes = cfg["classes"]

    try:
        model = load_model(name, ckpt)
    except FileNotFoundError as e:
        print(f"[EVAL] ✘ {e}  → SKIPPING"); return None

    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred, average="macro")

    print(f"[EVAL]   Test accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"[EVAL]   Test F1-macro  : {f1:.4f}")
    print(f"\n[EVAL]   Classification Report:")
    print(classification_report(y_te, y_pred, target_names=classes))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_title(f"{name} — Confusion Matrix\nAccuracy: {acc*100:.1f}%  F1: {f1:.3f}",
                 fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(res, f"{name}_cm.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"[EVAL]   Confusion matrix saved → {cm_path}")

    return {
        "model":    name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(name: str, model, res_dir: str, top_n: int = 20):
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
    if importance is None:
        return

    idx = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(9, 5))
    plt.barh(range(top_n), importance[idx], color="steelblue")
    plt.yticks(range(top_n), [f"feat_{i}" for i in idx], fontsize=8)
    plt.title(f"{name} — Top {top_n} Feature Importances")
    plt.tight_layout()
    path = os.path.join(res_dir, f"{name}_feat_importance.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[EVAL]   Feature importance plot → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Load existing .npy instead of re-extracting")
    parser.add_argument("--model", type=str, default=None,
                        help="Train/eval a single model by name")
    args = parser.parse_args()

    print(f"\n{'█'*60}")
    print(f"  DIP FORM ANALYZER — FULL ML PIPELINE")
    print(f"{'█'*60}")

    cfg = load_cfg()
    make_dirs(cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.skip_preprocess:
        print(f"\n[MAIN] --skip-preprocess → loading existing arrays")
        X_tr, y_tr, X_val, y_val, X_te, y_te = load_processed(cfg)
    else:
        print(f"\n[MAIN] Starting full preprocessing + feature engineering ...")
        X_tr, y_tr, X_val, y_val, X_te, y_te = preprocess(cfg)

    print(f"\n[MAIN] Feature dimensionality: {X_tr.shape[1]}")

    targets = [args.model] if args.model else ALL_MODELS
    print(f"[MAIN] Models queued: {targets}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{'█'*60}")
    print(f"  TRAINING {len(targets)} MODEL(S)")
    print(f"{'█'*60}")
    train_records = []
    for idx, name in enumerate(targets):
        print(f"\n[MAIN] ── Model {idx+1}/{len(targets)}: {name} ──")
        try:
            rec = train_one(name, cfg, X_tr, y_tr, X_val, y_val)
            if rec:
                train_records.append(rec)
        except Exception as e:
            print(f"[MAIN]   ✘ ERROR training {name}: {e}")
            import traceback; traceback.print_exc()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n{'█'*60}")
    print(f"  EVALUATION ON TEST SET")
    print(f"{'█'*60}")
    eval_records = []
    for name in targets:
        rec = evaluate_one(name, cfg, X_te, y_te)
        if rec:
            eval_records.append(rec)
            # Feature importance where applicable
            try:
                m = load_model(name, cfg["training"]["checkpoint_dir"])
                plot_feature_importance(name, m, cfg["training"]["results_dir"])
            except Exception:
                pass

    # ── Ensemble ──────────────────────────────────────────────────────────────
    if not args.model:
        print(f"\n[EVAL] ── Running ENSEMBLE soft-vote prediction ──")
        try:
            avg_probs  = ensemble_predict(X_te, cfg["training"]["checkpoint_dir"])
            y_pred_ens = np.argmax(avg_probs, axis=1)
            acc_ens    = accuracy_score(y_te, y_pred_ens)
            f1_ens     = f1_score(y_te, y_pred_ens, average="macro")
            print(f"[EVAL]   Ensemble accuracy : {acc_ens:.4f}  ({acc_ens*100:.1f}%)")
            print(f"[EVAL]   Ensemble F1-macro : {f1_ens:.4f}")
            print(classification_report(y_te, y_pred_ens, target_names=cfg["classes"]))
            eval_records.append({
                "model": "ensemble", "accuracy": round(acc_ens, 4), "f1_macro": round(f1_ens, 4)
            })
        except Exception as e:
            print(f"[EVAL]   ✘ Ensemble error: {e}")

    # ── Summary tables ────────────────────────────────────────────────────────
    res_dir = cfg["training"]["results_dir"]
    if train_records:
        df_tr = pd.DataFrame(train_records).sort_values("cv_acc_mean", ascending=False)
        out_tr = os.path.join(res_dir, "train_summary.csv")
        df_tr.to_csv(out_tr, index=False)
        print(f"\n{'█'*60}")
        print(f"  TRAINING SUMMARY (ranked by CV accuracy)")
        print(f"{'█'*60}")
        print(df_tr.to_string(index=False))
        print(f"\n[MAIN] ✔ Train summary → {out_tr}")

    if eval_records:
        df_ev = pd.DataFrame(eval_records).sort_values("accuracy", ascending=False)
        out_ev = os.path.join(res_dir, "test_summary.csv")
        df_ev.to_csv(out_ev, index=False)
        print(f"\n{'█'*60}")
        print(f"  TEST RESULTS (ranked by accuracy)")
        print(f"{'█'*60}")
        print(df_ev.to_string(index=False))
        print(f"\n[MAIN] ✔ Test summary → {out_ev}")

    print(f"\n[MAIN] ✔ Pickled models  : {cfg['training']['checkpoint_dir']}/")
    print(f"[MAIN] ✔ Plots & CSVs   : {cfg['training']['results_dir']}/")
    print(f"\n{'█'*60}")
    print(f"  PIPELINE COMPLETE ✔")
    print(f"{'█'*60}\n")


if __name__ == "__main__":
    main()