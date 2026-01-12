import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# 1. Load Test Datasets
# ============================================================
print("📂 Loading test datasets...")

data_folder = r"pushup_model\pushup_dataset_testandtrain"

X_test_correct = np.load(os.path.join(data_folder, "X_test_correct.npy"))
y_test_correct = np.load(os.path.join(data_folder, "y_test_correct.npy"))
X_test_wrong = np.load(os.path.join(data_folder, "X_test_wrong.npy"))
y_test_wrong = np.load(os.path.join(data_folder, "y_test_wrong.npy"))

# Choose which dataset to test on
X_test, y_test = X_test_correct, y_test_correct
print(f"✅ Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")

# ============================================================
# 2. Path to Saved Models
# ============================================================
models_folder = r"pushup_model\saved_models"

if not os.path.exists(models_folder):
    raise FileNotFoundError(f"❌ Models folder not found at {models_folder}")

# ============================================================
# 3. Evaluate Models and Collect Metrics
# ============================================================
results = []

for filename in os.listdir(models_folder):
    if filename.endswith(".pkl"):
        model_path = os.path.join(models_folder, filename)
        print(f"\n🔹 Processing model: {filename}")

        try:
            # Load the model safely
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Predict on test data
            y_pred = model.predict(X_test)

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Display metrics
            print(f"✅ Accuracy: {acc:.4f}")
            print("🔸 Confusion Matrix:")
            print(cm)
            print("\n🔸 Classification Report:")
            print(classification_report(y_test, y_pred))

            # Save main metrics
            results.append({
                "Model": filename,
                "Accuracy": acc,
                "Precision (Macro)": report["macro avg"]["precision"],
                "Recall (Macro)": report["macro avg"]["recall"],
                "F1-Score (Macro)": report["macro avg"]["f1-score"]
            })

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

# ============================================================
# 4. Save Results to CSV
# ============================================================
if results:
    results_df = pd.DataFrame(results)
    output_path = os.path.join(models_folder, "model_metrics_summary.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n📊 All model metrics saved to: {output_path}")
else:
    print("⚠️ No valid models processed.")
