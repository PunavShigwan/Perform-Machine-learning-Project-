import os
import time
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Folders
# --------------------------
train_folder = r"C:\major_project\squat_model\features\train"
test_folder = r"C:\major_project\squat_model\features\test"
model_folder = "saved_models"
report_folder = "evaluation_reports"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

# --------------------------   







# Load Data
# --------------------------
print("✅ Data Loaded Successfully!")
data_train = np.load(os.path.join(train_folder, "train.npy"), allow_pickle=True).item()
data_test = np.load(os.path.join(test_folder, "test.npy"), allow_pickle=True).item()

X_train, y_train = data_train["X"], data_train["y"]
X_test, y_test = data_test["X"], data_test["y"]

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}\n")

# --------------------------
# Models to Train (fast first)
# --------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=30),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=30),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "AdaBoost": AdaBoostClassifier()
}

# --------------------------
# Training & Evaluation
# --------------------------
results = []
with tqdm(total=len(models), desc="Training Progress", unit="model") as pbar:
    for name, model in models.items():
        print(f"\n🚀 Training {name} model...")
        start_time = time.time()

        # Add inner progress bar for GradientBoosting (optional)
        if isinstance(model, GradientBoostingClassifier):
            model.set_params(verbose=1)

        model.fit(X_train, y_train)
        end_time = time.time()

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        train_time = end_time - start_time

        # Save model
        model_path = os.path.join(model_folder, f"{name}.pkl")
        pickle.dump(model, open(model_path, "wb"))

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(report_folder, f"{name}_confusion_matrix.png"))
        plt.close()

        # Store results
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Training Time (s)": train_time
        })

        pbar.update(1)
        print(f"✅ {name} Done! (Time: {train_time:.2f}s, Accuracy: {acc:.4f})")

# --------------------------
# Save Report
# --------------------------
report_path = os.path.join(report_folder, "model_evaluation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=" * 80 + "\n")
    for r in results:
        f.write(f"\nModel: {r['Model']}\n")
        f.write(f"Accuracy: {r['Accuracy']:.4f}\n")
        f.write(f"Precision: {r['Precision']:.4f}\n")
        f.write(f"Recall: {r['Recall']:.4f}\n")
        f.write(f"F1-Score: {r['F1-Score']:.4f}\n")
        f.write(f"Training Time (s): {r['Training Time (s)']:.2f}\n")
        f.write("-" * 80 + "\n")

print("\n✅ All models trained and reports saved successfully!")
