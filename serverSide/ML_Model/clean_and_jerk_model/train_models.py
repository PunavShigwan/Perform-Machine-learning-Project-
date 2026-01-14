import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from label_map import LABEL_MAP

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\cj_sorted_output"
MODEL_DIR = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
print("\n========== LOADING DATA ==========")

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =====================================================
# MODELS
# =====================================================
models = {
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    "svm": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    ),
    "logistic_regression": LogisticRegression(
        max_iter=3000,
        n_jobs=-1
    )
}

# =====================================================
# TRAIN, EVALUATE, SAVE
# =====================================================
results = {}

for name, model in models.items():
    print(f"\n========== TRAINING {name.upper()} ==========")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", round(acc, 4))

    # 🔥 FIX: Handle missing classes safely
    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_names = [LABEL_MAP[i] for i in present_labels]

    print(classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)
    print("✔ Saved:", model_path)

    results[name] = acc

# =====================================================
# SAVE BEST MODEL + LABEL MAP
# =====================================================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(LABEL_MAP, os.path.join(MODEL_DIR, "label_map.pkl"))

print("\n========== FINAL RESULT ==========")
print("Best Model:", best_model_name)
print("Best Accuracy:", round(results[best_model_name], 4))
print("✔ best_model.pkl saved")
print("✔ label_map.pkl saved")
