import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# =====================
# 1. Load the datasets
# =====================
data_folder = "pushup_dataset_testandtrain"

X_train_correct = np.load(os.path.join(data_folder, "X_train_correct.npy"))
X_test_correct = np.load(os.path.join(data_folder, "X_test_correct.npy"))
y_train_correct = np.load(os.path.join(data_folder, "y_train_correct.npy"))
y_test_correct = np.load(os.path.join(data_folder, "y_test_correct.npy"))

X_train_wrong = np.load(os.path.join(data_folder, "X_train_wrong.npy"))
X_test_wrong = np.load(os.path.join(data_folder, "X_test_wrong.npy"))
y_train_wrong = np.load(os.path.join(data_folder, "y_train_wrong.npy"))
y_test_wrong = np.load(os.path.join(data_folder, "y_test_wrong.npy"))

# =====================
# 2. Merge correct & wrong into one dataset
# =====================
X_train = np.concatenate([X_train_correct, X_train_wrong], axis=0)
y_train = np.concatenate([y_train_correct, y_train_wrong], axis=0)

X_test = np.concatenate([X_test_correct, X_test_wrong], axis=0)
y_test = np.concatenate([y_test_correct, y_test_wrong], axis=0)

# Flatten features if necessary
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# =====================
# 3. Standardize data
# =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================
# 4. Define models
# =====================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel="rbf", probability=True),
    "GradientBoosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# =====================
# 5. Train & Evaluate
# =====================
results = {}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Derived metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    specificity = TN / (TN + FP)
    false_positive_rate = FP / (FP + TN)
    false_negative_rate = FN / (FN + TP)
    npv = TN / (TN + FN)
    fdr = FP / (TP + FP)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "FPR": false_positive_rate,
        "FNR": false_negative_rate,
        "NPV": npv,
        "FDR": fdr
    }

    # Print metrics neatly
    print("Confusion Matrix:\n", cm)
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"📈 Recall (Sensitivity): {recall:.4f}")
    print(f"🧩 Specificity: {specificity:.4f}")
    print(f"⚖️  F1 Score: {f1:.4f}")
    print(f"🚫 FPR: {false_positive_rate:.4f}")
    print(f"❌ FNR: {false_negative_rate:.4f}")
    print(f"📘 NPV: {npv:.4f}")
    print(f"📕 FDR: {fdr:.4f}")

# =====================
# 6. Plot accuracy comparison
# =====================
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), [r["Accuracy"] for r in results.values()], color='skyblue')
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# =====================
# 7. (Optional) Print Summary Table
# =====================
print("\n📊 Summary of All Models:")
print("-" * 90)
print(f"{'Model':<20}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1':<10}{'Specificity':<12}")
print("-" * 90)
for name, m in results.items():
    print(f"{name:<20}{m['Accuracy']:<10.3f}{m['Precision']:<10.3f}{m['Recall']:<10.3f}{m['F1 Score']:<10.3f}{m['Specificity']:<12.3f}")
