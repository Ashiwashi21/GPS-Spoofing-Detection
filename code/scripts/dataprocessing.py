import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

base = Path("sensor-data")
normal_path = base / "normal" / "normal.csv"
spoof_path = base / "spoofed" / "spoofed.csv"
out_dir = base / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

out_merged = out_dir / "merged.csv"
out_X_train = out_dir / "X_train.npy"
out_X_test = out_dir / "X_test.npy"
out_y_train = out_dir / "y_train.npy"
out_y_test = out_dir / "y_test.npy"

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
        mat = group.drop(columns=["label", "segment_id", "latitude", "longitude"]).values.astype(float)
        X.append(mat)
        y.append(int(group["label"].iloc[0]))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save(out_X_train, X_train)
np.save(out_X_test, X_test)
np.save(out_y_train, y_train)
np.save(out_y_test, y_test)

print("Saved merged dataset to", out_merged)
print("X_train", X_train.shape, "X_test", X_test.shape)
print("y_train", y_train.shape, "y_test", y_test.shape)
