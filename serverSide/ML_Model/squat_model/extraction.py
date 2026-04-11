"""
====================================================
  SQUAT FEATURE EXTRACTOR  v2  -  LEAKAGE-FREE
====================================================

KEY FIXES vs v1:
  1. VIDEO-GROUP AWARE  — frames are tagged by their
     source video/folder so the train/test split
     happens AT THE VIDEO LEVEL, not the frame level.
     Identical frames from the same clip can never
     appear in both train and test.

  2. TEMPORAL SUBSAMPLING  — only every Nth frame is
     kept from each clip (FRAME_STRIDE=5 by default).
     This drastically reduces near-duplicate frames
     and speeds up extraction.

  3. PERCENTILE LABELING still used (top 40% -> GOOD)
     but now also stored a raw form_score so the
     trainer can re-threshold without re-extracting.

  4. GROUP COLUMN  — 'video_group' is written to the
     CSV so the trainer can do GroupKFold / group
     split instead of random split.

Folder convention (two options work):
  Option A  — flat folder, filenames carry the clip id:
      frames/
        clip01_frame001.jpg
        clip01_frame002.jpg
        clip02_frame001.jpg   <-- different group
  Option B  — sub-folders per video (recommended):
      frames/
        clip01/frame001.jpg
        clip01/frame002.jpg
        clip02/frame001.jpg

Run:
    python extract_squat_features_v2.py
====================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json
import re
import time
from pathlib import Path
from collections import defaultdict

# =====================================================
# CONFIG  — adjust these paths for your machine
# =====================================================
FRAMES_DIR  = r"C:\major_project\serverSide\ML_Model\squat_model\frames"
OUTPUT_CSV  = r"C:\major_project\serverSide\ML_Model\squat_model\squat_features_v2.csv"
OUTPUT_JSON = r"C:\major_project\serverSide\ML_Model\squat_model\squat_features_v2_meta.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Take 1 frame every FRAME_STRIDE frames PER VIDEO GROUP
# Set to 1 to keep every frame (more data but more duplicates)
FRAME_STRIDE = 5

# Top GOOD_PERCENTILE % by form score become GOOD (1)
GOOD_PERCENTILE = 40

# Minimum keypoint visibility to accept a frame
MIN_VISIBILITY = 0.35


# =====================================================
# PRINT HELPERS
# =====================================================
def section(title):
    print("\n" + "=" * 65)
    print("  " + title)
    print("=" * 65)

def step(msg):  print("\n  >  " + msg)
def ok(msg):    print("     OK   " + msg)
def warn(msg):  print("     WARN " + msg)
def info(msg):  print("     INFO " + msg)


# =====================================================
# MEDIAPIPE SETUP
# =====================================================
mp_pose = mp.solutions.pose


def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def lm_xy(lm, i):   return [lm[i].x, lm[i].y]
def lm_vis(lm, i):  return lm[i].visibility


# =====================================================
# INFER VIDEO GROUP FROM FILE PATH
# =====================================================
def infer_video_group(img_path: Path) -> str:
    """
    Strategy (tried in order):
    1. If the image lives in a sub-folder of FRAMES_DIR, use the sub-folder name.
       e.g. frames/clip01/frame005.jpg  →  'clip01'
    2. Otherwise extract a prefix from the filename before the first digit run.
       e.g. clip01_frame005.jpg  →  'clip01'
       e.g. video_2_frame_042.jpg  →  'video_2'
    3. Fallback: whole filename stem (worst case — each frame is its own group,
       which defeats the purpose but won't crash).
    """
    frames_root = Path(FRAMES_DIR)
    try:
        relative = img_path.relative_to(frames_root)
        parts = relative.parts
        if len(parts) > 1:          # sub-folder exists
            return parts[0]
    except ValueError:
        pass

    # Filename heuristic: everything up to the last long digit run
    stem = img_path.stem
    # Remove trailing frame index (e.g. _0042, _frame042, frame042)
    cleaned = re.sub(r'[_\-]?(?:frame|frm|f)?[_\-]?\d{2,}$', '', stem, flags=re.IGNORECASE)
    return cleaned if cleaned else stem


# =====================================================
# FORM SCORE  (0-100, higher = better squat)
# =====================================================
def compute_form_score(lm):
    score = 0.0
    bd    = {}

    # 1. Knee depth  (25 pts) — lower angle = deeper squat
    left_knee  = compute_angle(lm_xy(lm,23), lm_xy(lm,25), lm_xy(lm,27))
    right_knee = compute_angle(lm_xy(lm,24), lm_xy(lm,26), lm_xy(lm,28))
    avg_knee   = (left_knee + right_knee) / 2
    ks = float(np.clip(25 * (180 - avg_knee) / 90, 0, 25))
    score += ks;  bd["knee_depth"] = round(ks, 2)

    # 2. Knee symmetry  (15 pts)
    ss = float(np.clip(15 * (1 - abs(left_knee - right_knee) / 40), 0, 15))
    score += ss;  bd["knee_symmetry"] = round(ss, 2)

    # 3. Knee valgus / cave  (20 pts) — knee width >= ankle width is good
    knee_w  = abs(lm[25].x - lm[26].x)
    ankle_w = abs(lm[27].x - lm[28].x) + 1e-6
    vs = float(np.clip(20 * (knee_w / ankle_w - 0.5) / 0.7, 0, 20))
    score += vs;  bd["knee_valgus"] = round(vs, 2)

    # 4. Spine upright  (20 pts)
    msx = (lm[11].x + lm[12].x) / 2;  msy = (lm[11].y + lm[12].y) / 2
    mhx = (lm[23].x + lm[24].x) / 2;  mhy = (lm[23].y + lm[24].y) / 2
    spine_lean = float(np.degrees(np.arctan2(abs(msx - mhx), abs(msy - mhy) + 1e-6)))
    sp = float(np.clip(20 * (1 - spine_lean / 60), 0, 20))
    score += sp;  bd["spine_upright"] = round(sp, 2)

    # 5. Hip hinge  (10 pts)
    left_hip  = compute_angle(lm_xy(lm,11), lm_xy(lm,23), lm_xy(lm,25))
    right_hip = compute_angle(lm_xy(lm,12), lm_xy(lm,24), lm_xy(lm,26))
    hs = float(np.clip(10 * (180 - (left_hip + right_hip) / 2) / 100, 0, 10))
    score += hs;  bd["hip_hinge"] = round(hs, 2)

    # 6. Body symmetry  (10 pts)
    sym_err = abs(lm[23].y - lm[24].y) + abs(lm[11].y - lm[12].y)
    bsym = float(np.clip(10 * (1 - sym_err / 0.1), 0, 10))
    score += bsym;  bd["body_symmetry"] = round(bsym, 2)

    return float(np.clip(score, 0, 100)), bd


# =====================================================
# FULL FEATURE EXTRACTION
# =====================================================
def extract_features(lm):
    f = {}
    msx = (lm[11].x + lm[12].x) / 2;  msy = (lm[11].y + lm[12].y) / 2
    mhx = (lm[23].x + lm[24].x) / 2;  mhy = (lm[23].y + lm[24].y) / 2

    # Joint angles
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

    f["spine_lean_angle"]  = float(np.degrees(np.arctan2(
        abs(msx - mhx), abs(msy - mhy) + 1e-6)))

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
        ("nose",0), ("l_shoulder",11), ("r_shoulder",12),
        ("l_hip",23), ("r_hip",24), ("l_knee",25), ("r_knee",26),
        ("l_ankle",27), ("r_ankle",28), ("l_foot",31), ("r_foot",32),
    ]:
        f[name + "_rel_x"] = lm[idx].x - mhx
        f[name + "_rel_y"] = lm[idx].y - mhy

    ys = [lm[i].y for i in [0,11,12,23,24,25,26,27,28]]
    f["body_height_range"] = max(ys) - min(ys)

    # Visibility (kept for filtering only, dropped before training)
    for name, idx in [
        ("vis_l_hip",23), ("vis_r_hip",24), ("vis_l_knee",25),
        ("vis_r_knee",26), ("vis_l_ankle",27), ("vis_r_ankle",28)
    ]:
        f[name] = lm_vis(lm, idx)

    return f


# =====================================================
# MAIN
# =====================================================
def main():
    total_start = time.time()

    section("SQUAT FEATURE EXTRACTION  v2  —  LEAKAGE-FREE")
    print("  Started at      : " + time.strftime("%H:%M:%S"))
    print("  Frames dir      : " + FRAMES_DIR)
    print("  Output CSV      : " + OUTPUT_CSV)
    print("  Frame stride    : every " + str(FRAME_STRIDE) + " frames per video group")
    print("  Label strategy  : Top " + str(GOOD_PERCENTILE) + "% by form score -> GOOD")
    print("")
    print("  KEY FIX: Frames are grouped by video source.")
    print("  The CSV includes a 'video_group' column so the")
    print("  trainer can split at the VIDEO level, preventing")
    print("  near-duplicate frames from leaking between train/test.")

    # --------------------------------------------------
    # STEP 1 — Scan and GROUP by video
    # --------------------------------------------------
    section("STEP 1 / 6  —  Scanning and grouping frames")

    frames_path = Path(FRAMES_DIR)
    if not frames_path.exists():
        print("  ERROR: Folder not found -> " + FRAMES_DIR)
        return

    all_imgs = []
    for ext in IMG_EXTS:
        all_imgs.extend(frames_path.rglob("*" + ext))

    if not all_imgs:
        print("  ERROR: No images found.")
        return

    ok("Total image files found: " + str(len(all_imgs)))

    # Group files by video source
    groups: dict = defaultdict(list)
    for p in sorted(all_imgs):
        groups[infer_video_group(p)].append(p)

    ok("Distinct video groups   : " + str(len(groups)))
    info("Groups detected:")
    for g, files in sorted(groups.items()):
        info("  " + str(g) + "  ->  " + str(len(files)) + " frames")

    # Apply stride subsampling per group
    subsampled = []
    for g, files in sorted(groups.items()):
        kept = files[::FRAME_STRIDE]
        subsampled.extend([(p, g) for p in kept])

    ok("After stride=" + str(FRAME_STRIDE) + " subsampling : " + str(len(subsampled)) + " frames")

    if len(groups) < 2:
        warn("Only 1 video group detected. The trainer will warn about")
        warn("this too. Organise frames into sub-folders per video for")
        warn("proper group-based splitting.")

    # --------------------------------------------------
    # STEP 2 — Load MediaPipe
    # --------------------------------------------------
    section("STEP 2 / 6  —  Loading MediaPipe Pose model")
    t0 = time.time()
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.4
    )
    ok("MediaPipe loaded in " + str(round(time.time() - t0, 2)) + "s")

    # --------------------------------------------------
    # STEP 3 — Extract features
    # --------------------------------------------------
    section("STEP 3 / 6  —  Extracting features and form scores")
    step("Processing " + str(len(subsampled)) + " frames...\n")

    rows       = []
    skip_count = 0
    n_total    = len(subsampled)
    batch_size = max(1, n_total // 10)

    for i, (img_path, group) in enumerate(subsampled, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            warn("Cannot read: " + img_path.name)
            skip_count += 1
            continue

        result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            warn("No pose : " + img_path.name)
            skip_count += 1
            continue

        try:
            lm              = result.pose_landmarks.landmark
            feats           = extract_features(lm)
            form_score, sbd = compute_form_score(lm)
        except Exception as e:
            warn("Error   : " + img_path.name + " -> " + str(e))
            skip_count += 1
            continue

        feats["form_score"]  = round(form_score, 3)
        feats["source_file"] = img_path.name
        feats["video_group"] = group   # <-- KEY addition
        for k, v in sbd.items():
            feats["score_" + k] = v

        rows.append(feats)

        if i % batch_size == 0 or i == n_total:
            elapsed = time.time() - total_start
            rate    = i / max(elapsed, 0.001)
            eta     = (n_total - i) / max(rate, 0.001)
            pct     = int(i / n_total * 100)
            bar     = "#" * int(pct/5) + "." * (20 - int(pct/5))
            print("  [" + bar + "] " + str(pct) + "%  (" +
                  str(i) + "/" + str(n_total) + ")  " +
                  "skip=" + str(skip_count) + "  " +
                  "rate=" + str(round(rate,1)) + "fps  " +
                  "ETA=" + str(int(eta)) + "s")

    pose.close()
    ok("Done. Valid: " + str(len(rows)) + "  Skipped: " + str(skip_count))

    if not rows:
        print("  ERROR: No features extracted.")
        return

    # --------------------------------------------------
    # STEP 4 — Build DataFrame + drop low-visibility
    # --------------------------------------------------
    section("STEP 4 / 6  —  Building DataFrame and cleaning")

    df = pd.DataFrame(rows)
    ok("Raw shape: " + str(df.shape[0]) + " x " + str(df.shape[1]))

    vis_cols = [c for c in df.columns if c.startswith("vis_")]
    before   = len(df)
    df       = df[(df[vis_cols] > MIN_VISIBILITY).all(axis=1)]
    dropped  = before - len(df)
    ok("Dropped low-vis rows: " + str(dropped) + "  Remaining: " + str(len(df)))

    # --------------------------------------------------
    # STEP 5 — Assign labels using percentile
    # --------------------------------------------------
    section("STEP 5 / 6  —  Assigning labels (percentile strategy)")

    scores    = df["form_score"]
    threshold = float(np.percentile(scores, 100 - GOOD_PERCENTILE))

    info("Score stats:")
    info("  MIN=" + str(round(float(scores.min()),2)) +
         "  MAX=" + str(round(float(scores.max()),2)) +
         "  MEAN=" + str(round(float(scores.mean()),2)) +
         "  STD=" + str(round(float(scores.std()),2)))
    ok("Label threshold (top " + str(GOOD_PERCENTILE) + "%) : " + str(round(threshold,2)))

    df["label"]      = (df["form_score"] >= threshold).astype(int)
    df["label_name"] = df["label"].map({1: "good_form", 0: "bad_form"})

    final_good = int((df["label"] == 1).sum())
    final_bad  = int((df["label"] == 0).sum())
    ok("GOOD (1): " + str(final_good) + "   BAD (0): " + str(final_bad))

    # Per-group label breakdown (for audit)
    info("Label distribution per video group:")
    grp_summary = df.groupby("video_group")["label"].value_counts().unstack(fill_value=0)
    for g, row in grp_summary.iterrows():
        bad_n  = row.get(0, 0)
        good_n = row.get(1, 0)
        info("  " + str(g) + " -> GOOD=" + str(good_n) + "  BAD=" + str(bad_n))

    # --------------------------------------------------
    # STEP 6 — Save CSV and JSON
    # --------------------------------------------------
    section("STEP 6 / 6  —  Saving output files")

    meta_cols  = ["label", "label_name", "form_score", "source_file", "video_group"]
    score_cols = [c for c in df.columns if c.startswith("score_")]
    vis_drop   = [c for c in df.columns if c.startswith("vis_")]
    feat_cols  = [c for c in df.columns
                  if c not in meta_cols and c not in score_cols and c not in vis_drop]

    ordered = (["label", "label_name", "video_group"] + feat_cols +
               score_cols + ["form_score", "source_file"])
    df = df[[c for c in ordered if c in df.columns]]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    ok("CSV saved: " + str(len(df)) + " rows  (" +
       str(round(os.path.getsize(OUTPUT_CSV)/1024, 1)) + " KB)")

    meta = {
        "labeling_strategy": "percentile_top_" + str(GOOD_PERCENTILE),
        "label_threshold":   round(threshold, 4),
        "frame_stride":      FRAME_STRIDE,
        "total_samples":     len(df),
        "good_form_count":   final_good,
        "bad_form_count":    final_bad,
        "video_groups":      sorted(df["video_group"].unique().tolist()),
        "feature_count":     len(feat_cols),
        "feature_names":     feat_cols,
        "label_map":         {"0": "bad_form", "1": "good_form"},
        "form_score_stats": {
            "min":    round(float(scores.min()),    4),
            "max":    round(float(scores.max()),    4),
            "mean":   round(float(scores.mean()),   4),
            "median": round(float(scores.median()), 4),
            "std":    round(float(scores.std()),    4),
        },
        "skipped_frames":  skip_count,
        "dropped_low_vis": dropped,
        "all_columns":     list(df.columns),
        "generated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "leakage_fix_note": (
            "video_group column written to CSV. "
            "Trainer uses GroupShuffleSplit + GroupKFold to ensure "
            "all frames from a video stay in one split."
        ),
    }
    with open(OUTPUT_JSON, "w") as jf:
        json.dump(meta, jf, indent=2)
    ok("Meta JSON saved -> " + OUTPUT_JSON)

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------
    section("EXTRACTION COMPLETE")
    print("  Total time     : " + str(round(time.time() - total_start, 1)) + "s")
    print("  Total samples  : " + str(len(df)))
    print("  GOOD (label=1) : " + str(final_good))
    print("  BAD  (label=0) : " + str(final_bad))
    print("  Video groups   : " + str(df["video_group"].nunique()))
    print("  Skipped        : " + str(skip_count))
    print("  Dropped (vis)  : " + str(dropped))
    print("")
    print("  CSV   -> " + OUTPUT_CSV)
    print("  META  -> " + OUTPUT_JSON)
    print("")
    print("  NEXT STEP: run train_squat_models_v2.py")
    print("  The trainer will split by video_group so no two frames")
    print("  from the same clip appear in both train and test.")
    print("=" * 65)


if __name__ == "__main__":
    main()