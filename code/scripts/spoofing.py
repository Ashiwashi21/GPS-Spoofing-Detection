from pathlib import Path
import re
import numpy as np
import pandas as pd

np.random.seed(42)
SRC = Path("sensor-data/normal/normal.csv")
OUT_DIR = Path("sensor-data/spoofed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "spoofed.csv"

# (parsing functions same as before...)

# Main
df = pd.read_csv(SRC)
df["lat_decimal"] = df["latitude"].apply(lambda v: parse_latlon_field(v, is_lat=True))
df["lon_decimal"] = df["longitude"].apply(lambda v: parse_latlon_field(v, is_lat=False))

df = df.dropna(subset=["lat_decimal","lon_decimal"]).reset_index(drop=True)

ref_lat, ref_lon = df["lat_decimal"].iloc[0], df["lon_decimal"].iloc[0]
df["x_m"], df["y_m"] = zip(*df.apply(lambda r: latlon_to_meters(r["lat_decimal"], r["lon_decimal"], ref_lat, ref_lon), axis=1))
df["label"] = 0

# assign blocks randomly
block_size = 5  # number of rows per mini-block
num_blocks = len(df) // block_size
all_blocks = list(range(num_blocks))
np.random.shuffle(all_blocks)

attacks = ["slow_drift", "sophisticated"]
tags = []

for blk in all_blocks:
    start_idx = blk * block_size
    end_idx = min(start_idx + block_size, len(df))
    if np.random.rand() < 0.3:  # 30% of blocks are attacks
        attack_type = np.random.choice(attacks)
        target_rows = list(range(start_idx, end_idx))
        if attack_type == "slow_drift":
            drift_rate = np.random.uniform(0.2, 1.5)
            inject_slow_drift(df, target_rows, drift_rate, ref_lat, ref_lon)
            tag = f"block{blk}_drift_{drift_rate:.2f}"
        else:
            offset = np.random.uniform(5.0, 30.0)
            inject_sophisticated(df, target_rows, offset, ref_lat, ref_lon)
            tag = f"block{blk}_soph_{int(offset)}m"
        df.loc[target_rows, "label"] = 1
        tags.append(tag)

# shuffle rows to scramble attack/normal order slightly
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
df.to_csv(OUT_PATH, index=False)
print("Saved spoofed file to:", OUT_PATH)
