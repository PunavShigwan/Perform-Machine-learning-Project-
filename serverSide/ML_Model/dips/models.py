"""
models.py — Machine Learning model architectures for dip form analysis.
All models operate on flattened/aggregated keypoint + angle features.
Input: (N, feature_dim) — flattened statistical features from (T, 99+6) keypoint sequences.
Output: num_classes (form classification)

Models included:
  1.  Random Forest
  2.  Gradient Boosting (sklearn)
  3.  XGBoost
  4.  LightGBM
  5.  CatBoost
  6.  Extra Trees
  7.  AdaBoost
  8.  SVM (RBF kernel)
  9.  K-Nearest Neighbors
  10. Logistic Regression (multinomial)
  11. Decision Tree
  12. Bagging Classifier (base: Decision Tree)
  13. Voting Ensemble (RF + XGB + LGB)
  14. Stacking Ensemble

Preprocessing utilities (StandardScaler, MinMaxScaler, PCA) are in pipeline.py.
"""

import os
import pickle
import numpy as np
import sklearn

# ── sklearn core ──────────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ── optional boosting libs ────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[MODELS] xgboost not installed — XGBClassifier skipped. "
          "Install: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("[MODELS] lightgbm not installed — LGBMClassifier skipped. "
          "Install: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False
    print("[MODELS] catboost not installed — CatBoostClassifier skipped. "
          "Install: pip install catboost")


# ══════════════════════════════════════════════════════════════════════════════
# VERSION-SAFE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _p(cfg, key):
    """Safely fetch cfg['ml_models'][key], returning {} if either level is missing."""
    return (cfg.get("ml_models") or {}).get(key) or {}


def _base_kwarg():
    """
    AdaBoostClassifier and BaggingClassifier renamed the base estimator
    parameter from 'base_estimator' -> 'estimator' in scikit-learn 1.2.
    Returns the correct keyword name for the currently installed version.
    """
    major, minor = [int(x) for x in sklearn.__version__.split(".")[:2]]
    return "estimator" if (major, minor) >= (1, 2) else "base_estimator"


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER FUNCTIONS
# Each returns a freshly instantiated (untrained) sklearn-compatible estimator.
# ══════════════════════════════════════════════════════════════════════════════

def build_random_forest(cfg):
    p = _p(cfg, "random_forest")
    return RandomForestClassifier(
        n_estimators      = p.get("n_estimators", 300),
        max_depth         = p.get("max_depth", None),
        min_samples_split = p.get("min_samples_split", 2),
        min_samples_leaf  = p.get("min_samples_leaf", 1),
        max_features      = p.get("max_features", "sqrt"),
        class_weight      = "balanced",
        n_jobs            = -1,
        random_state      = 42,
    )


def build_gradient_boosting(cfg):
    p = _p(cfg, "gradient_boosting")
    return GradientBoostingClassifier(
        n_estimators      = p.get("n_estimators", 200),
        learning_rate     = p.get("learning_rate", 0.05),
        max_depth         = p.get("max_depth", 4),
        subsample         = p.get("subsample", 0.8),
        min_samples_split = p.get("min_samples_split", 4),
        random_state      = 42,
    )


def build_xgboost(cfg):
    if not _HAS_XGB:
        return None
    p = _p(cfg, "xgboost")
    return XGBClassifier(
        n_estimators     = p.get("n_estimators", 300),
        learning_rate    = p.get("learning_rate", 0.05),
        max_depth        = p.get("max_depth", 5),
        subsample        = p.get("subsample", 0.8),
        colsample_bytree = p.get("colsample_bytree", 0.8),
        reg_alpha        = p.get("reg_alpha", 0.1),
        reg_lambda       = p.get("reg_lambda", 1.0),
        use_label_encoder= False,
        eval_metric      = "mlogloss",
        n_jobs           = -1,
        random_state     = 42,
        verbosity        = 0,
    )


def build_lightgbm(cfg):
    if not _HAS_LGB:
        return None
    p = _p(cfg, "lightgbm")
    return LGBMClassifier(
        n_estimators     = p.get("n_estimators", 300),
        learning_rate    = p.get("learning_rate", 0.05),
        num_leaves       = p.get("num_leaves", 63),
        max_depth        = p.get("max_depth", -1),
        subsample        = p.get("subsample", 0.8),
        colsample_bytree = p.get("colsample_bytree", 0.8),
        reg_alpha        = p.get("reg_alpha", 0.1),
        reg_lambda       = p.get("reg_lambda", 1.0),
        class_weight     = "balanced",
        n_jobs           = -1,
        random_state     = 42,
        verbose          = -1,
    )


def build_catboost(cfg):
    if not _HAS_CAT:
        return None
    p = _p(cfg, "catboost")
    return CatBoostClassifier(
        iterations         = p.get("iterations", 300),
        learning_rate      = p.get("learning_rate", 0.05),
        depth              = p.get("depth", 6),
        l2_leaf_reg        = p.get("l2_leaf_reg", 3),
        auto_class_weights = "Balanced",
        random_seed        = 42,
        verbose            = 0,
    )


def build_extra_trees(cfg):
    p = _p(cfg, "extra_trees")
    return ExtraTreesClassifier(
        n_estimators      = p.get("n_estimators", 300),
        max_depth         = p.get("max_depth", None),
        min_samples_split = p.get("min_samples_split", 2),
        class_weight      = "balanced",
        n_jobs            = -1,
        random_state      = 42,
    )


def build_adaboost(cfg):
    p    = _p(cfg, "adaboost")
    base = DecisionTreeClassifier(max_depth=p.get("base_max_depth", 3))
    # sklearn < 1.2 uses 'base_estimator'; >= 1.2 uses 'estimator'
    kw   = {_base_kwarg(): base}
    return AdaBoostClassifier(
        **kw,
        n_estimators  = p.get("n_estimators", 200),
        learning_rate = p.get("learning_rate", 0.5),
        random_state  = 42,
    )


def build_svm(cfg):
    p = _p(cfg, "svm")
    return SVC(
        C            = p.get("C", 10.0),
        kernel       = p.get("kernel", "rbf"),
        gamma        = p.get("gamma", "scale"),
        class_weight = "balanced",
        probability  = True,    # enables predict_proba()
        random_state = 42,
    )


def build_knn(cfg):
    p = _p(cfg, "knn")
    return KNeighborsClassifier(
        n_neighbors = p.get("n_neighbors", 7),
        weights     = p.get("weights", "distance"),
        metric      = p.get("metric", "euclidean"),
        n_jobs      = -1,
    )


def build_logistic_regression(cfg):
    p = _p(cfg, "logistic_regression")
    return LogisticRegression(
        C            = p.get("C", 1.0),
        max_iter     = p.get("max_iter", 1000),
        solver       = "lbfgs",
        multi_class  = "multinomial",
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )


def build_decision_tree(cfg):
    p = _p(cfg, "decision_tree")
    return DecisionTreeClassifier(
        max_depth         = p.get("max_depth", 12),
        min_samples_split = p.get("min_samples_split", 4),
        min_samples_leaf  = p.get("min_samples_leaf", 2),
        class_weight      = "balanced",
        random_state      = 42,
    )


def build_bagging(cfg):
    p    = _p(cfg, "bagging")
    base = DecisionTreeClassifier(max_depth=p.get("base_max_depth", 8))
    # sklearn < 1.2 uses 'base_estimator'; >= 1.2 uses 'estimator'
    kw   = {_base_kwarg(): base}
    return BaggingClassifier(
        **kw,
        n_estimators = p.get("n_estimators", 100),
        max_samples  = p.get("max_samples", 0.8),
        max_features = p.get("max_features", 0.8),
        n_jobs       = -1,
        random_state = 42,
    )


def build_voting_ensemble(cfg):
    """Soft-voting ensemble: RF + XGB (if available) + LGB (if available)."""
    estimators = [("rf", build_random_forest(cfg))]
    if _HAS_XGB:
        estimators.append(("xgb", build_xgboost(cfg)))
    if _HAS_LGB:
        estimators.append(("lgb", build_lightgbm(cfg)))
    if len(estimators) < 2:
        estimators.append(("et", build_extra_trees(cfg)))
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def build_stacking_ensemble(cfg):
    """
    Level-0 learners: RF, Extra Trees, GBM (+ XGB/LGB if available).
    Meta-learner: Logistic Regression.
    """
    level0 = [
        ("rf",  build_random_forest(cfg)),
        ("et",  build_extra_trees(cfg)),
        ("gbm", build_gradient_boosting(cfg)),
    ]
    if _HAS_XGB:
        level0.append(("xgb", build_xgboost(cfg)))
    if _HAS_LGB:
        level0.append(("lgb", build_lightgbm(cfg)))

    meta = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        multi_class="multinomial", random_state=42
    )
    return StackingClassifier(
        estimators      = level0,
        final_estimator = meta,
        cv              = 5,
        stack_method    = "predict_proba",
        n_jobs          = -1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

BUILDERS = {
    "random_forest":        build_random_forest,
    "gradient_boosting":    build_gradient_boosting,
    "xgboost":              build_xgboost,
    "lightgbm":             build_lightgbm,
    "catboost":             build_catboost,
    "extra_trees":          build_extra_trees,
    "adaboost":             build_adaboost,
    "svm":                  build_svm,
    "knn":                  build_knn,
    "logistic_regression":  build_logistic_regression,
    "decision_tree":        build_decision_tree,
    "bagging":              build_bagging,
    "voting_ensemble":      build_voting_ensemble,
    "stacking_ensemble":    build_stacking_ensemble,
}

ALL_MODELS = list(BUILDERS.keys())

# Best individual models for soft-vote ensemble
ENSEMBLE_MEMBERS = [
    "random_forest", "xgboost", "lightgbm",
    "gradient_boosting", "extra_trees"
]


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, name: str, checkpoint_dir: str):
    """Serialize a trained sklearn estimator to a .pkl file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVE] Model pickled → {path}")
    return path


def load_model(name: str, checkpoint_dir: str):
    """Load a pickled sklearn estimator."""
    path = os.path.join(checkpoint_dir, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[LOAD] Model loaded ← {path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICT  (soft-vote over saved individual models)
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(X: np.ndarray, checkpoint_dir: str) -> np.ndarray:
    """
    Average predicted probabilities from all available ENSEMBLE_MEMBERS.
    X       : (N, feature_dim) — already preprocessed (scaled) features.
    Returns : (N, num_classes) averaged probability array.
    """
    all_probs = []
    for name in ENSEMBLE_MEMBERS:
        path = os.path.join(checkpoint_dir, f"{name}.pkl")
        if not os.path.exists(path):
            print(f"  [SKIP] {name} — checkpoint not found")
            continue
        model = load_model(name, checkpoint_dir)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        else:
            preds = model.predict(X)
            n_cls = len(np.unique(preds))
            probs = np.eye(n_cls)[preds]
        all_probs.append(probs)
        print(f"  [ENSEMBLE] {name} — probs shape {probs.shape}")

    if not all_probs:
        raise RuntimeError("No ensemble member checkpoints found.")
    return np.mean(all_probs, axis=0)