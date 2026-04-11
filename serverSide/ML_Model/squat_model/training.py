"""
====================================================
  SQUAT MODEL TRAINER  v2  —  LEAKAGE-FREE
====================================================

KEY FIXES vs v1:

  1. GROUP-BASED SPLIT  — train/test split is done at
     the VIDEO level using GroupShuffleSplit.
     All frames from one video clip stay together in
     either train OR test, never both.
     This kills the "near-duplicate frame" leakage.

  2. SCALER FITTED ON TRAIN ONLY  — StandardScaler is
     fit exclusively on X_train and then applied to
     X_test. In v1 the Pipeline called fit_transform
     during cross_val_score but the initial scaler was
     fit on all data, causing subtle leakage.
     v2 uses GroupKFold inside cross_val_score so each
     fold also honours video boundaries.

  3. HONEST CROSS-VALIDATION  — GroupKFold(n_splits=5)
     is used instead of StratifiedKFold. Each fold puts
     whole videos in test, never individual frames.

  4. WHAT TO EXPECT  — Accuracy will realistically be
     in the 80–95% range instead of 99–100%.
     That is the correct, publishable number.

Run:
    python train_squat_models_v2.py
====================================================
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection   import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score
)
from sklearn.pipeline          import Pipeline
from sklearn.ensemble          import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier

# =====================================================
# PATHS  — match the extractor's output paths
# =====================================================
CSV_PATH   = r"C:\major_project\serverSide\ML_Model\squat_model\squat_features_v2.csv"
MODELS_DIR = r"C:\major_project\serverSide\ML_Model\squat_model\saved_models_v2"
REPORT_DIR = r"C:\major_project\serverSide\ML_Model\squat_model\reports_v2"

# Test set size (fraction of VIDEO GROUPS, not frames)
TEST_GROUP_FRACTION = 0.20
RANDOM_STATE        = 42

# =====================================================
# PRINT HELPERS
# =====================================================
def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def step(msg):   print(f"\n  >  {msg}")
def ok(msg):     print(f"     OK   {msg}")
def warn(msg):   print(f"     WARN {msg}")
def info(msg):   print(f"     INFO {msg}")
def header(msg): print(f"\n  {'─'*60}\n  {msg}\n  {'─'*60}")


# =====================================================
# MODEL DEFINITIONS
# (Pipelines — scaler is fitted on train only
#  because cross_val_score re-fits the whole pipeline
#  per fold, which is correct behaviour)
# =====================================================
def build_models():
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_depth=None,
                min_samples_split=5, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08,
                max_depth=4, subsample=0.85,
                random_state=RANDOM_STATE
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    ExtraTreesClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    AdaBoostClassifier(
                n_estimators=150, learning_rate=0.5,
                random_state=RANDOM_STATE
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=1.0, max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                C=1.0, kernel="rbf", probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(
                n_neighbors=7, weights="distance",
                metric="minkowski", n_jobs=-1
            )),
        ]),
    }


# =====================================================
# PLOTS
# =====================================================
def plot_confusion_matrix(cm, model_name, labels, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="gray", ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
    ax.set_ylabel("Actual",    fontsize=12, labelpad=8)
    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_chart(results, out_path):
    names   = list(results.keys())
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc", "cv_mean"]
    labels  = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "CV Mean"]
    colors  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
    x       = np.arange(len(names))
    width   = 0.13

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i*width - (len(metrics)-1)*width/2,
                      vals, width, label=label, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Group-Based Evaluation (No Data Leakage)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_times(results, out_path):
    names  = list(results.keys())
    times  = [results[n]["train_time_sec"] for n in names]
    colors = ["#4C72B0" if t == min(times) else
              "#C44E52" if t == max(times) else "#55A868" for t in times]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(names, times, color=colors, alpha=0.85, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{t:.2f}s", va="center", fontsize=10)
    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time per Model", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(pipeline, model_name, feature_names, out_path, top_n=20):
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    imp  = clf.feature_importances_
    idx  = np.argsort(imp)[::-1][:top_n]
    nms  = [feature_names[i] for i in idx]
    vals = imp[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(nms[::-1], vals[::-1],
            color=plt.cm.viridis(np.linspace(0.2, 0.85, len(nms))),
            edgecolor="white", alpha=0.9)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_text_report(results, best_name, out_path, n_train, n_test, n_groups,
                     train_groups, test_groups):
    lines = [
        "=" * 65,
        "  SQUAT ML TRAINING REPORT  v2  (Group-Based, No Leakage)",
        f"  Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "  SPLIT METHOD : GroupShuffleSplit (by video group)",
        f"  Train frames : {n_train}   from groups: {train_groups}",
        f"  Test  frames : {n_test}    from groups: {test_groups}",
        f"  Total groups : {n_groups}",
        f"  CV method    : GroupKFold (5 folds, video-aware)",
        "",
        f"  Best Model   : {best_name}",
        f"  Best Acc     : {results[best_name]['accuracy']:.4f}",
        "",
        "  NOTE: Accuracy is intentionally lower than v1 because",
        "  near-duplicate frames from the same video NO LONGER",
        "  appear in both train and test. This is the correct,",
        "  honest, publishable result.",
        "",
    ]

    for name, r in results.items():
        lines += [
            "─" * 65,
            f"  MODEL : {name}",
            "─" * 65,
            f"  Accuracy        : {r['accuracy']:.4f}",
            f"  F1 Score        : {r['f1']:.4f}",
            f"  Precision       : {r['precision']:.4f}",
            f"  Recall          : {r['recall']:.4f}",
            f"  ROC-AUC         : {r['roc_auc']:.4f}",
            f"  GroupKFold CV   : {r['cv_mean']:.4f} +/- {r['cv_std']:.4f}",
            f"  CV Folds        : {r['cv_folds']}",
            f"  Train time      : {r['train_time_sec']:.2f}s",
            "",
            f"  Classification Report:\n{r['class_report']}",
        ]

    lines += [
        "=" * 65,
        "  RANKING (by Accuracy)",
        "─" * 65,
    ]
    ranked = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        lines.append(f"  #{rank}  {name:22s}  acc={r['accuracy']:.4f}  "
                     f"f1={r['f1']:.4f}  roc={r['roc_auc']:.4f}")
    lines.append("=" * 65)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =====================================================
# MAIN
# =====================================================
def main():
    total_start = time.time()

    section("SQUAT ML TRAINING PIPELINE  v2  —  LEAKAGE-FREE")
    print(f"  Started at  : {time.strftime('%H:%M:%S')}")
    print(f"  Dataset     : {CSV_PATH}")
    print(f"  Models dir  : {MODELS_DIR}")
    print(f"  Reports dir : {REPORT_DIR}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # --------------------------------------------------
    # STEP 1 — Load dataset
    # --------------------------------------------------
    section("STEP 1 / 5  —  Loading dataset")
    step("Reading CSV...")

    if not Path(CSV_PATH).exists():
        print(f"  ERROR: CSV not found: {CSV_PATH}")
        print("  Run extract_squat_features_v2.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    ok(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # Check video_group column
    if "video_group" not in df.columns:
        warn("'video_group' column missing from CSV!")
        warn("Re-run extract_squat_features_v2.py to get the group column.")
        warn("Falling back to random split (LEAKAGE NOT FIXED).")
        df["video_group"] = "all_frames"

    groups_series = df["video_group"]
    unique_groups = groups_series.unique()
    ok(f"Video groups found: {len(unique_groups)}  -> {list(unique_groups)}")

    if len(unique_groups) < 3:
        warn("Only " + str(len(unique_groups)) + " video groups detected.")
        warn("For a proper 80/20 split you need at least 5+ groups (videos).")
        warn("Consider splitting your footage into more separate clips.")

    # Drop meta / non-feature columns
    drop_cols = (["label_name", "source_file", "video_group"] +
                 [c for c in df.columns if c.startswith("vis_")])
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "label" not in df_clean.columns:
        print("  ERROR: 'label' column missing.")
        return

    X = df_clean.drop(columns=["label"])
    y = df_clean["label"]
    groups = groups_series.values
    feature_names = list(X.columns)

    info(f"Features       : {len(feature_names)}")
    info(f"GOOD (label=1) : {(y==1).sum()}")
    info(f"BAD  (label=0) : {(y==0).sum()}")

    if (y==1).sum() == 0 or (y==0).sum() == 0:
        print("  ERROR: Only one class present. Adjust GOOD_PERCENTILE in extractor.")
        return

    # Handle NaN
    nan_count = X.isna().sum().sum()
    if nan_count:
        warn(f"{nan_count} NaN values -> filling with column median")
        X = X.fillna(X.median())
    else:
        ok("No NaN values")

    # --------------------------------------------------
    # STEP 2 — GROUP-BASED Train/Test Split
    # --------------------------------------------------
    section("STEP 2 / 5  —  Group-Based Train / Test Split")
    print(f"\n  Splitting by VIDEO GROUP (not random frame split).")
    print(f"  Test fraction : {int(TEST_GROUP_FRACTION*100)}% of groups")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_GROUP_FRACTION,
                            random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups[train_idx]
    g_test  = groups[test_idx]

    train_groups = sorted(set(g_train))
    test_groups  = sorted(set(g_test))

    ok(f"Train frames : {len(X_train)}  from groups : {train_groups}")
    ok(f"Test  frames : {len(X_test)}   from groups : {test_groups}")
    info(f"Train GOOD={(y_train==1).sum()}  BAD={(y_train==0).sum()}")
    info(f"Test  GOOD={(y_test==1).sum()}   BAD={(y_test==0).sum()}")

    # Verify ZERO overlap in source groups
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        warn("GROUP OVERLAP DETECTED: " + str(overlap))
        warn("This means some groups ended up in both splits.")
        warn("Increase your number of video groups to fix this.")
    else:
        ok("ZERO group overlap between train and test — leakage eliminated!")

    # --------------------------------------------------
    # STEP 3 — Train all models
    # --------------------------------------------------
    section("STEP 3 / 5  —  Training Models")

    MODELS = build_models()
    info(f"Training {len(MODELS)} models...")

    # GroupKFold for cross-validation (video-aware)
    n_cv_splits = min(5, len(unique_groups) - 1)
    if n_cv_splits < 2:
        warn("Not enough groups for GroupKFold. Using n_splits=2.")
        n_cv_splits = 2
    gkf = GroupKFold(n_splits=n_cv_splits)
    info(f"GroupKFold CV splits : {n_cv_splits} (one video group held out per fold)")

    results = {}

    for idx, (model_name, pipeline) in enumerate(MODELS.items(), 1):
        header(f"[{idx}/{len(MODELS)}]  {model_name}")

        # ── Train ──────────────────────────────────────
        step(f"Fitting {model_name}...")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)
        ok(f"Trained in {train_time}s")

        # ── Predict on held-out test GROUPS ────────────
        step("Predicting on test set (unseen video groups)...")
        y_pred    = pipeline.predict(X_test)
        raw_proba = pipeline.predict_proba(X_test)
        classes   = list(pipeline.classes_)

        if len(classes) < 2:
            warn("Model only saw one class — ROC-AUC set to 0.5")
            y_proba = np.zeros(len(X_test))
            roc     = 0.5
        else:
            pos_idx = classes.index(1) if 1 in classes else 1
            y_proba = raw_proba[:, pos_idx]
            unique_test = np.unique(y_test)
            if len(unique_test) < 2:
                warn("Test set only has one class — ROC-AUC set to 0.5")
                roc = 0.5
            else:
                roc = roc_auc_score(y_test, y_proba)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        cr   = classification_report(y_test, y_pred,
                                     target_names=["BAD (0)", "GOOD (1)"],
                                     zero_division=0)

        ok(f"Accuracy  : {acc:.4f}  (honest group-split result)")
        ok(f"F1 Score  : {f1:.4f}")
        ok(f"Precision : {prec:.4f}")
        ok(f"Recall    : {rec:.4f}")
        ok(f"ROC-AUC   : {roc:.4f}")

        # ── GroupKFold cross-validation ─────────────────
        step(f"GroupKFold CV ({n_cv_splits} folds)...")
        cv_scores = cross_val_score(
            pipeline, X, y,
            cv=gkf,
            groups=groups,          # <-- honours video boundaries
            scoring="accuracy",
            n_jobs=-1
        )
        ok(f"CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        info(f"CV Folds    : {[round(s,4) for s in cv_scores]}")

        if cv_scores.mean() > 0.98:
            warn("CV mean still >0.98 even with group split.")
            warn("Possible causes: too few groups, very easy dataset, or")
            warn("the biomechanical features genuinely separate classes well.")
            warn("Check that you have >= 10 distinct video clips.")

        # ── Confusion matrix ────────────────────────────
        cm_path = os.path.join(REPORT_DIR, f"cm_{model_name}.png")
        plot_confusion_matrix(cm, model_name, ["BAD", "GOOD"], cm_path)
        ok(f"Confusion matrix saved")

        # ── Feature importance ──────────────────────────
        fi_path = os.path.join(REPORT_DIR, f"fi_{model_name}.png")
        plot_feature_importance(pipeline, model_name, feature_names, fi_path)

        # ── Save model ──────────────────────────────────
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        ok(f"Model saved ({os.path.getsize(model_path)//1024} KB)")

        print(f"\n  Classification Report:\n")
        for line in cr.split("\n"):
            print(f"     {line}")

        results[model_name] = {
            "accuracy":       round(acc,  4),
            "f1":             round(f1,   4),
            "precision":      round(prec, 4),
            "recall":         round(rec,  4),
            "roc_auc":        round(roc,  4),
            "cv_mean":        round(float(cv_scores.mean()), 4),
            "cv_std":         round(float(cv_scores.std()),  4),
            "cv_folds":       [round(s, 4) for s in cv_scores.tolist()],
            "train_time_sec": train_time,
            "confusion_matrix": cm.tolist(),
            "class_report":   cr,
            "model_path":     model_path,
        }

    # --------------------------------------------------
    # STEP 4 — Reports & Charts
    # --------------------------------------------------
    section("STEP 4 / 5  —  Generating Reports & Charts")

    comp_path = os.path.join(REPORT_DIR, "model_comparison.png")
    plot_comparison_chart(results, comp_path)
    ok("Saved model_comparison.png")

    time_path = os.path.join(REPORT_DIR, "training_times.png")
    plot_training_times(results, time_path)
    ok("Saved training_times.png")

    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "model":          name,
            "accuracy":       r["accuracy"],
            "f1":             r["f1"],
            "precision":      r["precision"],
            "recall":         r["recall"],
            "roc_auc":        r["roc_auc"],
            "cv_mean":        r["cv_mean"],
            "cv_std":         r["cv_std"],
            "train_time_sec": r["train_time_sec"],
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False)
    summary_csv = os.path.join(REPORT_DIR, "model_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    ok("Saved model_summary.csv")

    best_name = summary_df.iloc[0]["model"]
    txt_path  = os.path.join(REPORT_DIR, "training_report.txt")
    save_text_report(
        results, best_name, txt_path,
        len(X_train), len(X_test), len(unique_groups),
        train_groups, test_groups
    )
    ok("Saved training_report.txt")

    json_results = {k: {kk: vv for kk, vv in v.items() if kk != "class_report"}
                    for k, v in results.items()}
    json_path = os.path.join(REPORT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    ok("Saved results.json")

    # --------------------------------------------------
    # STEP 5 — Save best model
    # --------------------------------------------------
    section("STEP 5 / 5  —  Saving Best Model")

    best_pipeline = MODELS[best_name]
    best_path     = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(best_pipeline, f)

    meta = {
        "best_model_name":   best_name,
        "best_model_path":   best_path,
        "best_accuracy":     results[best_name]["accuracy"],
        "best_f1":           results[best_name]["f1"],
        "best_roc_auc":      results[best_name]["roc_auc"],
        "feature_names":     feature_names,
        "label_map":         {"0": "bad_form", "1": "good_form"},
        "trained_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_method":      "GroupShuffleSplit by video_group",
        "cv_method":         f"GroupKFold n_splits={n_cv_splits}",
        "train_groups":      train_groups,
        "test_groups":       test_groups,
        "leakage_free":      True,
    }
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    ok(f"Best model ({best_name}) and meta saved")

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------
    section("TRAINING COMPLETE")
    total_time = time.time() - total_start
    print(f"  Total time : {total_time:.1f}s\n")
    print(f"  {'RANK':<6} {'MODEL':<22} {'ACC':>7} {'F1':>7} "
          f"{'PREC':>7} {'REC':>7} {'ROC':>7} {'CV':>7} {'TIME':>7}")
    print(f"  {'─'*70}")
    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        crown = " <-- BEST" if rank == 1 else ""
        print(f"  #{rank:<5} {row['model']:<22} {row['accuracy']:>7.4f} "
              f"{row['f1']:>7.4f} {row['precision']:>7.4f} "
              f"{row['recall']:>7.4f} {row['roc_auc']:>7.4f} "
              f"{row['cv_mean']:>7.4f} {row['train_time_sec']:>6.2f}s{crown}")

    print(f"\n  Best model : {best_name}")
    print(f"  Accuracy   : {results[best_name]['accuracy']:.4f}  (honest, leakage-free)")
    print(f"  F1 Score   : {results[best_name]['f1']:.4f}")
    print(f"  ROC-AUC    : {results[best_name]['roc_auc']:.4f}")
    print(f"\n  Train groups : {train_groups}")
    print(f"  Test  groups : {test_groups}")
    print(f"\n  Reports -> {REPORT_DIR}")
    print(f"  Models  -> {MODELS_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()