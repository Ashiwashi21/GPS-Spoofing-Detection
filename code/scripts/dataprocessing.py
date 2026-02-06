#goal of this script: create testing (5) and training (x/y) files, combine normal and spoofing attacks together and create segments for each attack
#imports
import pandas as pd #database
import numpy as np #math
from pathlib import Path #paths
from sklearn.utils import shuffle #shuffle

#file paths and output locations
base = Path("sensor-data")
normal_path = base / "normal" / "normal.csv"
spoof_path = base / "spoofed" / "spoofed.csv"
out_dir = base / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

out_merged = out_dir / "merged.csv"
out_X_train = out_dir / "X_train.npy"
out_y_train = out_dir / "y_train.npy"

#load normal and spoofed data, label them
normal = pd.read_csv(normal_path)
spoofed = pd.read_csv(spoof_path)

normal["label"] = 0
spoofed["label"] = 1

#create segment ideas for time windows (every 10 seconds)
normal["segment_id"] = normal.index // 10
spoofed["segment_id"] = spoofed.index // 10 + normal["segment_id"].max() + 1

#merge, clean, shuffle segment ids
merged = pd.concat([normal, spoofed], ignore_index=True)
merged = merged.fillna(0).reset_index(drop=True)

all_segments = merged["segment_id"].unique()
np.random.seed(42)
np.random.shuffle(all_segments)

#split segments into training (80%) and testing (20%)
test_ratio = 0.2
num_test_segments = int(len(all_segments) * test_ratio)
test_segments = set(all_segments[:num_test_segments])
train_segments = set(all_segments[num_test_segments:])

train_df = merged[merged["segment_id"].isin(train_segments)]
test_df = merged[merged["segment_id"].isin(test_segments)]

#degine window size and feature columns (which colums are features)
window_size = 10
feature_cols = [c for c in merged.columns if c not in ["label", "segment_id"]]
num_features = len(feature_cols)

#function to extract windows
'''
process:
1) checks each segment to see
-if the segment is too short
-takes first 10 rows of data
-pads colums with 0's if needed so every window has 18 features
-adds windows to x and its labels to y
2) returns x as a 3d array (windows x time x features) and y as labels
'''
def extract_windows(df):
    X, y = [], []
    for seg_id, group in df.groupby("segment_id"):
        if len(group) < window_size:
            continue
        mat = group[feature_cols].iloc[:window_size].values.astype(float)
        pad_width = 18 - mat.shape[1]
        if pad_width > 0:
            mat = np.concatenate([mat, np.zeros((window_size, pad_width))], axis=1)
        X.append(mat)
        y.append(int(group["label"].iloc[0]))
    return np.array(X), np.array(y)

#build train and test arrays
X_train, y_train = extract_windows(train_df)
X_test, y_test = extract_windows(test_df)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

#save training data
np.save(out_X_train, X_train)
np.save(out_y_train, y_train)

#split test data into 5 parts
num_parts = 5
test_size = X_test.shape[0]
split_size = test_size // num_parts

for i in range(num_parts):
    start = i * split_size
    end = (i + 1) * split_size if i < num_parts - 1 else test_size
    np.save(out_dir / f"X_test_{i+1}.npy", X_test[start:end])
    np.save(out_dir / f"y_test_{i+1}.npy", y_test[start:end])

#sanity checks
#check for train-test overlap (sanity)
def segment_signature(x):
    return tuple(np.round(x.flatten(), 4))

train_sigs = {segment_signature(x) for x in X_train}
test_sigs = {segment_signature(x) for x in X_test}
overlap = len(train_sigs & test_sigs)

#print summaries
print("Preprocessing complete!")
print("Train segments:", len(train_segments), "Test segments:", len(test_segments))
print("Approx segment overlap:", overlap)
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
