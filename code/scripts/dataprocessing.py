import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle

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

# Assign segment IDs
normal["segment_id"] = normal.index // 10
spoofed["segment_id"] = spoofed.index // 10 + normal["segment_id"].max() + 1

# Merge datasets
merged = pd.concat([normal, spoofed], ignore_index=True)
merged = merged.fillna(0).reset_index(drop=True)

# -------------------------------
# Split train/test by segment
# -------------------------------
all_segments = merged["segment_id"].unique()
np.random.seed(42)
np.random.shuffle(all_segments)

test_ratio = 0.2
num_test_segments = int(len(all_segments) * test_ratio)
test_segments = set(all_segments[:num_test_segments])
train_segments = set(all_segments[num_test_segments:])

train_df = merged[merged["segment_id"].isin(train_segments)]
test_df = merged[merged["segment_id"].isin(test_segments)]

# -------------------------------
# Window extraction
# -------------------------------
window_size = 10
feature_cols = [c for c in merged.columns if c not in ["label", "segment_id"]]
num_features = len(feature_cols)

def extract_windows(df):
    X, y = [], []
    for seg_id, group in df.groupby("segment_id"):
        if len(group) < window_size:
            continue
        mat = group[feature_cols].iloc[:window_size].values.astype(float)
        # Pad features if fewer than expected
        pad_width = 18 - mat.shape[1]
        if pad_width > 0:
            mat = np.concatenate([mat, np.zeros((window_size, pad_width))], axis=1)
        X.append(mat)
        y.append(int(group["label"].iloc[0]))
    return np.array(X), np.array(y)

X_train, y_train = extract_windows(train_df)
X_test, y_test = extract_windows(test_df)

# Shuffle training set
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# -------------------------------
# Save
# -------------------------------
np.save(out_X_train, X_train)
np.save(out_y_train, y_train)

# Split test set into 5 parts
num_parts = 5
test_size = X_test.shape[0]
split_size = test_size // num_parts

for i in range(num_parts):
    start = i * split_size
    end = (i + 1) * split_size if i < num_parts - 1 else test_size
    np.save(out_dir / f"X_test_{i+1}.npy", X_test[start:end])
    np.save(out_dir / f"y_test_{i+1}.npy", y_test[start:end])

# -------------------------------
# Segment overlap check
# -------------------------------
def segment_signature(x):
    return tuple(np.round(x.flatten(), 4))

train_sigs = {segment_signature(x) for x in X_train}
test_sigs = {segment_signature(x) for x in X_test}
overlap = len(train_sigs & test_sigs)

print("Preprocessing complete!")
print("Train segments:", len(train_segments), "Test segments:", len(test_segments))
print("Approx segment overlap:", overlap)
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
