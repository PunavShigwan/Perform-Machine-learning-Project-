import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter

# =====================
# 1. Decision Tree Implementation
# =====================
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # number of features to consider at each split
        self.tree = None

    def fit(self, X, y):
        print("🌲 Training a new Decision Tree...")
        self.n_classes = len(set(y))
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.tree = self._grow_tree(X, y)
        print("✅ Decision Tree trained successfully.")

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None

        features = np.random.choice(n, self.n_features, replace=False)
        best_gini = 1.0
        split_idx, split_thresh = None, None

        for feat in features:
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gini = (left_mask.sum() / m) * self._gini(y[left_mask]) + (right_mask.sum() / m) * self._gini(y[right_mask])
                if gini < best_gini:
                    best_gini = gini
                    split_idx, split_thresh = feat, t
        return split_idx, split_thresh

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {"class": predicted_class}

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                left_mask = X[:, idx] <= thr
                right_mask = ~left_mask
                node = {
                    "feature": idx,
                    "threshold": thr,
                    "left": self._grow_tree(X[left_mask], y[left_mask], depth + 1),
                    "right": self._grow_tree(X[right_mask], y[right_mask], depth + 1),
                }
        return node

    def _predict_one(self, inputs, node):
        if "feature" not in node:  # leaf
            return node["class"]
        if inputs[node["feature"]] <= node["threshold"]:
            return self._predict_one(inputs, node["left"])
        else:
            return self._predict_one(inputs, node["right"])

    def predict(self, X):
        print("🔮 Predicting using a Decision Tree...")
        preds = np.array([self._predict_one(inputs, self.tree) for inputs in X])
        print("✅ Predictions done.")
        return preds


# =====================
# 2. Random Forest Implementation
# =====================
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        print(f"\n🌳 Training Random Forest with {self.n_trees} trees...")
        self.trees = []
        n_samples, n_features = X.shape
        for i in range(self.n_trees):
            print(f"   📌 Training tree {i+1}/{self.n_trees}")
            idxs = np.random.choice(n_samples, n_samples, replace=True)  # bootstrap
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=int(np.sqrt(n_features)))  # random subset of features
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        print("✅ Random Forest trained successfully.\n")

    def predict(self, X):
        print("🔮 Predicting with Random Forest...")
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        preds = np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])
        print("✅ Random Forest predictions complete.")
        return preds


# =====================
# 3. Load dataset
# =====================
print("📂 Loading dataset...")

data_path = r"C:\major_project\pushup_model\pushup_dataset_testandtrain"

X_train_correct = np.load(os.path.join(data_path, "X_train_correct.npy"))
X_test_correct = np.load(os.path.join(data_path, "X_test_correct.npy"))
y_train_correct = np.load(os.path.join(data_path, "y_train_correct.npy"))
y_test_correct = np.load(os.path.join(data_path, "y_test_correct.npy"))

X_train_wrong = np.load(os.path.join(data_path, "X_train_wrong.npy"))
X_test_wrong = np.load(os.path.join(data_path, "X_test_wrong.npy"))
y_train_wrong = np.load(os.path.join(data_path, "y_train_wrong.npy"))
y_test_wrong = np.load(os.path.join(data_path, "y_test_wrong.npy"))

X_train = np.concatenate([X_train_correct, X_train_wrong], axis=0)
y_train = np.concatenate([y_train_correct, y_train_wrong], axis=0)
X_test = np.concatenate([X_test_correct, X_test_wrong], axis=0)
y_test = np.concatenate([y_test_correct, y_test_wrong], axis=0)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print("✅ Dataset loaded successfully.")
print(f"   🔹 Training samples: {X_train.shape}, Labels: {y_train.shape}")
print(f"   🔹 Testing samples: {X_test.shape}, Labels: {y_test.shape}")


# =====================
# 4. Train RandomForest from scratch
# =====================
print("\n🚀 Starting training process...")
rf = RandomForest(n_trees=10, max_depth=10)
rf.fit(X_train, y_train)


# =====================
# 5. Evaluate
# =====================
print("\n📊 Evaluating model...")
y_pred = rf.predict(X_test)
acc = np.mean(y_pred == y_test)

print(f"✅ Accuracy: {acc:.4f}")

# Confusion Matrix
conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))), dtype=int)
for t, p in zip(y_test, y_pred):
    conf_matrix[int(t)][int(p)] += 1

print("📌 Confusion Matrix:\n", conf_matrix)


# =====================
# 6. Save model
# =====================
print("\n💾 Saving model...")
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/RandomForest_from_scratch.pkl", "wb") as f:
    pickle.dump(rf, f)
print("✅ Model saved at 'saved_models/RandomForest_from_scratch.pkl'")


# =====================
# 7. Plot accuracy bar
# =====================
print("\n📈 Plotting accuracy graph...")
plt.figure(figsize=(6, 4))
plt.bar(["RandomForest (Scratch)"], [acc], color='skyblue')
plt.ylabel("Accuracy")
plt.title("RandomForest Accuracy (From Scratch)")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("randomforest_from_scratch_accuracy.png")
plt.show()
print("✅ Accuracy graph saved as 'randomforest_from_scratch_accuracy.png'")
