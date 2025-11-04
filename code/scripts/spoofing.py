import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

src = Path("sensor-data/normal/normal.csv")
spoof_dir = Path("sensor-data/spoofed")
spoof_dir.mkdir(parents=True, exist_ok=True)
out_path = spoof_dir / "spoofed.csv"


def nmea_to_decimal(val):
    if pd.isna(val) or not val:
        return np.nan
    try:
        num, direction = val.split()
        num = float(num)
        deg = int(num // 100)
        minutes = num - deg * 100
        dec = deg + minutes / 60
        if direction in ("S", "W"):
            dec *= -1
        return dec
    except Exception:
        return np.nan


def decimal_to_nmea(dec, lat=True):
    if pd.isna(dec):
        return ""
    if lat:
        sign = "N" if dec >= 0 else "S"
    else:
        sign = "E" if dec >= 0 else "W"
    dec = abs(dec)
    deg = int(dec)
    mins = (dec - deg) * 60
    return f"{deg * 100 + mins:08.5f} {sign}"


def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))
    return (lon - ref_lon) * m_per_lon, (lat - ref_lat) * m_per_lat


def meters_to_latlon(dx, dy, ref_lat, ref_lon):
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))
    lat = ref_lat + dy / m_per_lat
    lon = ref_lon + dx / m_per_lon
    return lat, lon


def inject_slow_drift(df, idx_list, drift_rate):
    for n, idx in enumerate(idx_list):
        if pd.isna(df.at[idx, "lat_decimal"]):
            continue
        dx = drift_rate * n
        dy = dx * np.sin(0.2)
        dx = dx * np.cos(0.2)
        x, y = df.at[idx, "x_m"], df.at[idx, "y_m"]
        new_lat, new_lon = meters_to_latlon(x + dx, y + dy, ref_lat, ref_lon)
        df.at[idx, "lat_decimal"] = new_lat
        df.at[idx, "lon_decimal"] = new_lon
        df.at[idx, "latitude"] = decimal_to_nmea(new_lat, True)
        df.at[idx, "longitude"] = decimal_to_nmea(new_lon, False)


def inject_sophisticated(df, idx_list, offset_m):
    direction = np.random.rand() * 2 * np.pi
    total_dx = offset_m * np.cos(direction)
    total_dy = offset_m * np.sin(direction)
    n_rows = max(1, len(idx_list))
    for step, idx in enumerate(idx_list):
        if pd.isna(df.at[idx, "lat_decimal"]):
            continue
        progress = (step + 1) / n_rows
        dx = progress * total_dx + np.random.normal(0, 0.5)
        dy = progress * total_dy + np.random.normal(0, 0.5)
        x, y = df.at[idx, "x_m"], df.at[idx, "y_m"]
        lat, lon = meters_to_latlon(x + dx, y + dy, ref_lat, ref_lon)
        df.at[idx, "lat_decimal"] = lat
        df.at[idx, "lon_decimal"] = lon
        df.at[idx, "latitude"] = decimal_to_nmea(lat, True)
        df.at[idx, "longitude"] = decimal_to_nmea(lon, False)


if not src.exists():
    raise FileNotFoundError(f"Expected input file at {src}")

df = pd.read_csv(src)
df["timestamp"] = pd.to_numeric(df.get("timestamp", df.index), errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)

if "lat_decimal" not in df.columns:
    df["lat_decimal"] = df["latitude"].apply(nmea_to_decimal)
if "lon_decimal" not in df.columns:
    df["lon_decimal"] = df["longitude"].apply(nmea_to_decimal)

ref_lat = df["lat_decimal"].dropna().iloc[0]
ref_lon = df["lon_decimal"].dropna().iloc[0]

coords = df.apply(
    lambda r: latlon_to_meters(
        r["lat_decimal"] if not pd.isna(r["lat_decimal"]) else ref_lat,
        r["lon_decimal"] if not pd.isna(r["lon_decimal"]) else ref_lon,
        ref_lat, ref_lon
    ),
    axis=1
)
df["x_m"], df["y_m"] = zip(*coords)

df["label"] = 0
start = df["timestamp"].iloc[0]
df["tensec_idx"] = ((df["timestamp"] - start) // 10).astype(int)

tensecs = sorted(df["tensec_idx"].unique())
attacks = ["slow_drift", "sophisticated"]
tags = []

for t in tensecs:
    attack_type = np.random.choice(attacks)
    group = df.index[df["tensec_idx"] == t].tolist()
    if not group:
        continue
    total = len(group)
    dur = int(np.clip(np.random.randint(1, 11), 1, total))
    start_offset = np.random.randint(0, max(1, total - dur + 1))
    target_rows = group[start_offset:start_offset + dur]
    if attack_type == "slow_drift":
        drift = float(np.random.uniform(0.2, 2.0))
        inject_slow_drift(df, target_rows, drift)
        tag = f"tensec{t}_drift_{drift:.2f}mps_{start_offset}s_{dur}s"
    else:
        offset = float(np.random.uniform(5.0, 50.0))
        inject_sophisticated(df, target_rows, offset)
        tag = f"tensec{t}_soph_{int(offset)}m_{start_offset}s_{dur}s"
    df.loc[target_rows, "label"] = 1
    tags.append(tag)

df.drop(columns=["tensec_idx"]).to_csv(out_path, index=False)


print(f"Spoofed dataset saved to {out_path}")
print("Injected attack segments:")
for t in tags:
    print(" -", t)
