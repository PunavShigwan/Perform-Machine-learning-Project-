"""
predict.py
----------
Analyze a single dip video using trained ML classifiers (.pkl files).

Outputs:
  - Form classification (correct / elbow_flare / shallow_depth / forward_lean)
  - Form score (0–100)
  - Detected issues with explanations
  - Rep count + average rep duration
  - Per-frame phase labels
  - Class probability breakdown

Usage:
    python predict.py --video myvideo.mp4
    python predict.py --video myvideo.mp4 --model random_forest
    python predict.py --video myvideo.mp4 --ensemble
"""

import argparse, yaml, os
import numpy as np

from pipeline import (
    extract_keypoints,
    extract_features,
    compute_form_score,
    count_reps_and_timing,
    Preprocessor,
)
from models import load_model, ensemble_predict


def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def analyze(video_path: str, model_name: str, cfg: dict, use_ensemble: bool = False):
    T        = cfg["preprocessing"]["max_frames"]
    classes  = cfg["classes"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    proc_dir = cfg["data"]["processed_dir"]

    print(f"\n  Extracting keypoints from: {video_path}")
    kp   = extract_keypoints(video_path, T)              # (T, 99)
    feat = extract_features(kp)                          # (feature_dim,)
    X    = feat[np.newaxis]                              # (1, feature_dim)

    # ── Load and apply the same preprocessor used during training ────────────
    pp_path = os.path.join(proc_dir, "preprocessor.pkl")
    if os.path.exists(pp_path):
        preprocessor = Preprocessor.load(pp_path)
        X = preprocessor.transform(X)
        print(f"  Preprocessor applied. Feature shape: {X.shape}")
    else:
        print(f"  ⚠️  Preprocessor not found at {pp_path} — using raw features.")

    # ── Form classification ───────────────────────────────────────────────────
    if use_ensemble:
        probs       = ensemble_predict(X, ckpt_dir)[0]
        model_label = "ensemble"
    else:
        model       = load_model(model_name, ckpt_dir)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
        else:
            pred_idx = model.predict(X)[0]
            probs    = np.eye(len(classes))[pred_idx]
        model_label = model_name

    pred_idx   = int(np.argmax(probs))
    pred_cls   = classes[pred_idx]
    confidence = probs[pred_idx] * 100

    # ── Form score ────────────────────────────────────────────────────────────
    score, issues = compute_form_score(kp, cfg)

    # ── Rep counting ─────────────────────────────────────────────────────────
    rep_count, avg_dur, phases = count_reps_and_timing(kp, fps=30)

    # ── Print report ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  DIP FORM ANALYSIS REPORT")
    print(f"{'═'*60}")
    print(f"  Model        : {model_label}")
    print(f"  Prediction   : {pred_cls.upper()}  ({confidence:.1f}% confidence)")
    emoji = "✅" if score >= 80 else "⚠️" if score >= 50 else "❌"
    print(f"  Form Score   : {score}/100  {emoji}")
    print(f"  Reps counted : {rep_count}")
    if avg_dur > 0:
        print(f"  Avg rep time : {avg_dur:.2f}s")

    print(f"\n  ── Class Probabilities ──")
    for cls, p in zip(classes, probs):
        bar  = "█" * int(p * 40)
        mark = " ◄" if cls == pred_cls else ""
        print(f"  {cls:<18} {p*100:5.1f}%  {bar}{mark}")

    if issues:
        print(f"\n  ── Issues Detected ──")
        for issue, detail in issues.items():
            print(f"  ⚠️  {issue.replace('_', ' ').title()}: {detail}")
    else:
        print(f"\n  ✅ No form issues detected — great technique!")

    print(f"\n  ── Per-frame Phases (sample) ──")
    step = max(1, len(phases) // 10)
    for i in range(0, len(phases), step):
        print(f"  Frame {i:3d}: {phases[i]}")

    print(f"\n{'═'*60}\n")

    return {
        "prediction":        pred_cls,
        "confidence":        confidence,
        "form_score":        score,
        "issues":            issues,
        "reps":              rep_count,
        "avg_rep_duration":  avg_dur,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",    required=True,         help="Path to .mp4 video")
    parser.add_argument("--model",    default="random_forest", help="Model name (default: random_forest)")
    parser.add_argument("--ensemble", action="store_true",   help="Use soft-vote ensemble")
    args = parser.parse_args()

    cfg = load_cfg()
    analyze(args.video, args.model, cfg, use_ensemble=args.ensemble)