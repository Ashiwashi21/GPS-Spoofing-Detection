import pandas as pd
from pathlib import Path
import numpy as np

normal_path = Path("sensor-data/normal/normal.csv")
spoofed_path = Path("sensor-data/spoofed/spoofed.csv")
out_dir = Path("sensor-data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

normal = pd.read_csv(normal_path)
spoofed = pd.read_csv(spoofed_path)

merged = pd.concat([normal, spoofed], ignore_index=True)
merged = merged.sort_values("timestamp").reset_index(drop=True)

merged["window_id"] = (merged.index // 10).astype(int)

unique_windows = merged["window_id"].unique()
np.random.seed(42)
np.random.shuffle(unique_windows)

split_point = int(0.8 * len(unique_windows))
train_windows = set(unique_windows[:split_point])
test_windows = set(unique_windows[split_point:])

train = merged[merged["window_id"].isin(train_windows)].copy()
test = merged[merged["window_id"].isin(test_windows)].copy()

train.to_csv(out_dir / "train.csv", index=False)
test.to_csv(out_dir / "test.csv", index=False)

print("Merged and split successfully by windows.")
print("Total windows:", len(unique_windows))
print("Train windows:", len(train_windows))
print("Test windows:", len(test_windows))
print("Train rows:", len(train))
print("Test rows:", len(test))
