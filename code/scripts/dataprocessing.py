import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
"""
base = Path("sensor-data")
normal_path = base / "normal.csv"
spoof_path = base / "spoofed.csv"
out_merged = base / "merged_tagged.csv"
out_X_train = base / "X_train.npy"
out_X_test = base / "X_test.npy"
out_y_train = base / "y_train.npy"
out_y_test = base / "y_test.npy"

normal = pd.read_csv(normal_path)
spoofed = pd.read_csv(spoof_path)

normal["label"] = 0
spoofed["label"] = 1

normal["segment_id"] = normal.index // 10
spoofed["segment_id"] = spoofed.index // 10 + normal["segment_id"].max() + 1

merged = pd.concat([normal, spoofed], ignore_index=True)
merged = merged.sort_values("segment_id").reset_index(drop=True)
merged.to_csv(out_merged, index=False)

df = merged.dropna().reset_index(drop=True)

window_size = 10
X = []
y = []
for seg_id, group in df.groupby("segment_id"):
    if len(group) == window_size:
        mat = group.drop(columns=["label","segment_id"]).values.astype(float)
        X.append(mat)
        y.append(int(group["label"].iloc[0]))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

np.save(out_X_train, X_train)
np.save(out_X_test, X_test)
np.save(out_y_train, y_train)
np.save(out_y_test, y_test)

print("saved", out_merged)
print("X_train", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
"""

df = pd.read_csv("sensor-data/spoofed/spoofed.csv")
for i in range(0, len(df), 10):
    window = df.iloc[i:i+10]
    duration = window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]
    print(f"Window {i//10}: {duration} seconds")
