import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

# -------------------------------
# Paths
# -------------------------------
base = Path("sensor-data")
normal_path = base / "normal" / "normal.csv"
spoof_path = base / "spoofed" / "spoofed.csv"
out_dir = base / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

# Output files
out_merged = out_dir / "merged.csv"
out_X_train = out_dir / "X_train.npy"
out_y_train = out_dir / "y_train.npy"

# -------------------------------
# Load and label
# -------------------------------
normal = pd.read_csv(normal_path)
spoofed = pd.read_csv(spoof_path)
normal["label"] = 0
spoofed["label"] = 1

# Segment IDs
normal["segment_id"] = normal.index // 10
spoofed["segment_id"] = spoofed.index // 10 + normal["segment_id"].max() + 1

# Merge and shuffle
merged = pd.concat([normal, spoofed], ignore_index=True)
merged = merged.sort_values("segment_id").reset_index(drop=True)
merged.to_csv(out_merged, index=False)

# -------------------------------
# Fill NaNs
# -------------------------------
df = merged.fillna(0).reset_index(drop=True)

# -------------------------------
# Segment transformations
# -------------------------------
def scramble_segment(mat):
    if np.random.rand() < 0.5:
        np.random.shuffle(mat)
    if np.random.rand() < 0.3:
        mat += np.random.normal(0, 0.1, mat.shape)
    return mat

def blend_segment(mat, label):
    if label == 1 and np.random.rand() < 0.2:
        blend = np.random.normal(0, 0.05, mat.shape)
        mat = mat * 0.8 + blend
    return mat

# -------------------------------
# Extract windows
# -------------------------------
window_size = 10
X, y = [], []

for seg_id, group in df.groupby("segment_id"):
    group_numeric = group.drop(columns=["label", "segment_id"])
    if len(group_numeric) < window_size:
        continue
    mat = group_numeric.iloc[:window_size].values.astype(float)
    label = int(group["label"].iloc[0])

    if label == 1:
        mat = scramble_segment(mat)
    mat = blend_segment(mat, label)

    X.append(mat)
    y.append(label)

X = np.array(X)
y = np.array(y)

# -------------------------------
# Stratified split
# -------------------------------
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# -------------------------------
# Save training set
# -------------------------------
np.save(out_X_train, X_train)
np.save(out_y_train, y_train)

# -------------------------------
# Split and save test set
# -------------------------------
num_parts = 5
test_size = X_test.shape[0]
split_size = test_size // num_parts

for i in range(num_parts):
    start = i * split_size
    end = (i + 1) * split_size if i < num_parts - 1 else test_size
    np.save(out_dir / f"X_test_{i+1}.npy", X_test[start:end])
    np.save(out_dir / f"y_test_{i+1}.npy", y_test[start:end])

# -------------------------------
# Diagnostics
# -------------------------------
print("? Preprocessing complete")
print("Train label distribution:", np.unique(y_train, return_counts=True))
print("Test label distribution:", np.unique(y_test, return_counts=True))
print("Saved merged dataset to:", out_merged)
print("X_train:", X_train.shape, "y_train:", y_train.shape)
for i in range(num_parts):
    xt = np.load(out_dir / f"X_test_{i+1}.npy")
    yt = np.load(out_dir / f"y_test_{i+1}.npy")
    print(f"Test case {i+1}: X_test_{i+1}.npy {xt.shape}, y_test_{i+1}.npy {yt.shape}")
