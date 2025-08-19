import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# =====================
# 1. Load the datasets
# =====================
X_train_correct = np.load("X_train_correct.npy")
X_test_correct = np.load("X_test_correct.npy")
y_train_correct = np.load("y_train_correct.npy")
y_test_correct = np.load("y_test_correct.npy")

X_train_wrong = np.load("X_train_wrong.npy")
X_test_wrong = np.load("X_test_wrong.npy")
y_train_wrong = np.load("y_train_wrong.npy")
y_test_wrong = np.load("y_test_wrong.npy")

# =====================
# 2. Merge correct & wrong into one dataset
# =====================
X_train = np.concatenate([X_train_correct, X_train_wrong], axis=0)
y_train = np.concatenate([y_train_correct, y_train_wrong], axis=0)

X_test = np.concatenate([X_test_correct, X_test_wrong], axis=0)
y_test = np.concatenate([y_test_correct, y_test_wrong], axis=0)

# Flatten features if necessary (e.g., for ML models)
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

# Create folder to save models
os.makedirs("saved_models", exist_ok=True)

# =====================
# 5. Train & Evaluate
# =====================
accuracy_scores = {}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc

    print(f"✅ Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    with open(f"saved_models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# =====================
# 6. Plot accuracy comparison
# =====================
plt.figure(figsize=(8, 5))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='skyblue')
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("model_accuracy_comparison.png")
plt.show()
