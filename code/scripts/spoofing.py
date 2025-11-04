#!/usr/bin/env python3
# code/scripts/spoofing.py
"""
Robust spoof injector.
- Tries multiple ways to parse lat/lon (NMEA with space, NMEA compact, decimal).
- If no valid coordinates found, prints diagnostics and exits cleanly.
- Injects two spoof types, labels rows, writes sensor-data/spoofed/spoofed.csv.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

np.random.seed(42)
SRC = Path("sensor-data/normal/normal.csv")
OUT_DIR = Path("sensor-data/spoofed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "spoofed.csv"

# Helpers
nmea_space_re = re.compile(r"^\s*([0-9]+\.[0-9]+)\s*([NSEWnsew])\s*$")
nmea_compact_re = re.compile(r"^\s*([0-9]{4,})(\.[0-9]+)?\s*([NSEWnsew])?\s*$")

def try_parse_nmea_with_space(s):
    m = nmea_space_re.match(str(s))
    if not m:
        return None
    num = float(m.group(1))
    dirc = m.group(2).upper()
    deg = int(num // 100)
    minutes = num - deg * 100
    dec = deg + minutes / 60.0
    if dirc in ("S", "W"):
        dec = -dec
    return dec

def try_parse_nmea_compact(s, is_lat=True):
    # handle things like "340311.406N" or "3403.11406N"
    s = str(s).strip()
    m = nmea_compact_re.match(s)
    if not m:
        return None
    num_part = m.group(1)
    frac = m.group(2) or ""
    num = float(num_part + (frac if frac else ""))
    # heuristic: if lat, deg part is first 2 digits for lat, 3 for lon sometimes
    # we try both 2 and 3 splits and pick reasonable result
    for deg_digits in (2, 3):
        if len(num_part) >= deg_digits + 2:  # at least deg + minutes (2)
            deg = int(num // (10**(len(num_part) - deg_digits)))
            minutes = num - deg * (10**(len(num_part) - deg_digits))
            # convert minutes which are expressed with variable digits, fallback:
            try:
                # convert to standard by treating minutes as minutes with decimal
                minutes_val = minutes / (10**(len(num_part) - deg_digits - (len(frac)-1 if frac else 0)))
            except Exception:
                minutes_val = minutes
            # simpler fallback: use original float with deg computed by integer division by 100
            deg2 = int(num // 100)
            minutes2 = num - deg2 * 100
            if 0 <= minutes2 < 60:
                deg, minutes = deg2, minutes2
            if 0 <= minutes < 60:
                dec = deg + minutes / 60.0
                # direction letter
                dirc = (m.group(3) or "").upper()
                if dirc in ("S", "W"):
                    dec = -dec
                return dec
    return None

def try_parse_decimal(s):
    try:
        return float(str(s))
    except Exception:
        return None

def parse_latlon_field(value, is_lat=True):
    # Attempt multiple parsers, return decimal or np.nan
    for fn in (try_parse_nmea_with_space, try_parse_nmea_compact, try_parse_decimal):
        try:
            out = fn(value)
        except Exception:
            out = None
        if out is not None:
            return out
    return np.nan

def decimal_to_nmea(dec, is_lat=True):
    if pd.isna(dec):
        return ""
    dec = float(dec)
    sign = "N" if dec >= 0 else "S" if is_lat else "E" if dec >= 0 else "W"
    dec_abs = abs(dec)
    deg = int(dec_abs)
    mins = (dec_abs - deg) * 60.0
    return f"{deg*100 + mins:08.5f} {sign}"

def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))
    dx = (lon - ref_lon) * m_per_lon
    dy = (lat - ref_lat) * m_per_lat
    return dx, dy

def meters_to_latlon(dx, dy, ref_lat, ref_lon):
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))
    lat = ref_lat + dy / m_per_lat
    lon = ref_lon + dx / m_per_lon
    return lat, lon

# Attack injectors
def inject_slow_drift(df, indices, drift_rate_m_per_step, ref_lat, ref_lon):
    for step, idx in enumerate(indices):
        if pd.isna(df.at[idx, "x_m"]) or pd.isna(df.at[idx, "y_m"]):
            continue
        dx = drift_rate_m_per_step * (step + 1)
        dy = np.random.normal(0, 0.3)
        new_x = df.at[idx, "x_m"] + dx
        new_y = df.at[idx, "y_m"] + dy
        lat, lon = meters_to_latlon(new_x, new_y, ref_lat, ref_lon)
        df.at[idx, "lat_decimal"] = lat
        df.at[idx, "lon_decimal"] = lon
        df.at[idx, "latitude"] = decimal_to_nmea(lat, True)
        df.at[idx, "longitude"] = decimal_to_nmea(lon, False)

def inject_sophisticated(df, indices, total_offset_m, ref_lat, ref_lon):
    direction = np.random.uniform(0, 2 * np.pi)
    total_dx = total_offset_m * np.cos(direction)
    total_dy = total_offset_m * np.sin(direction)
    n = max(1, len(indices))
    for i, idx in enumerate(indices):
        if pd.isna(df.at[idx, "x_m"]) or pd.isna(df.at[idx, "y_m"]):
            continue
        progress = (i + 1) / n
        step_dx = progress * total_dx + np.random.normal(0, 0.5)
        step_dy = progress * total_dy + np.random.normal(0, 0.5)
        new_x = df.at[idx, "x_m"] + step_dx
        new_y = df.at[idx, "y_m"] + step_dy
        lat, lon = meters_to_latlon(new_x, new_y, ref_lat, ref_lon)
        df.at[idx, "lat_decimal"] = lat
        df.at[idx, "lon_decimal"] = lon
        df.at[idx, "latitude"] = decimal_to_nmea(lat, True)
        df.at[idx, "longitude"] = decimal_to_nmea(lon, False)

# Main
if not SRC.exists():
    raise SystemExit(f"Input file not found: {SRC}")

df = pd.read_csv(SRC)
print("Loaded rows:", len(df))
# preserve original imu columns and all other columns
orig_cols = list(df.columns)

# normalize timestamp if exists
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# Ensure lat and lon present
if "latitude" not in df.columns or "longitude" not in df.columns:
    print("Missing 'latitude' or 'longitude' columns. Columns present:", df.columns.tolist())
    raise SystemExit("CSV must contain 'latitude' and 'longitude' columns")

# Parse lat/lon robustly
df["lat_decimal"] = df["latitude"].apply(lambda v: parse_latlon_field(v, is_lat=True))
df["lon_decimal"] = df["longitude"].apply(lambda v: parse_latlon_field(v, is_lat=False))

valid_lat = df["lat_decimal"].notna().sum()
valid_lon = df["lon_decimal"].notna().sum()
print(f"Valid parsed lat count: {valid_lat}, lon count: {valid_lon}")

# If no valid coordinates, print diagnostics and exit
if valid_lat == 0 or valid_lon == 0:
    print("Parsing failed to find any valid lat/lon entries.")
    sample_vals = df[["latitude", "longitude"]].head(20)
    print("Sample latitude/longitude values from CSV (first 20 rows):")
    print(sample_vals.to_string(index=False))
    raise SystemExit("No valid GPS coordinates found after multiple parsing attempts. Fix input format and try again.")

# drop rows without valid lat/lon
df = df.dropna(subset=["lat_decimal", "lon_decimal"]).reset_index(drop=True)
if df.empty:
    raise SystemExit("No rows left after dropping invalid coordinates")

# set reference
ref_lat = float(df["lat_decimal"].iloc[0])
ref_lon = float(df["lon_decimal"].iloc[0])

# compute x_m, y_m
coords = df.apply(lambda r: latlon_to_meters(r["lat_decimal"], r["lon_decimal"], ref_lat, ref_lon), axis=1)
df["x_m"], df["y_m"] = zip(*coords)

# prepare labels and grouping
df["label"] = 0
if "timestamp" in df.columns and not df["timestamp"].isna().all():
    start_ts = int(df["timestamp"].iloc[0])
    df["tensec_idx"] = ((df["timestamp"] - start_ts) // 10).astype(int)
else:
    df["tensec_idx"] = (df.index // 10).astype(int)

tensecs = sorted(df["tensec_idx"].unique())
attacks = ["slow_drift", "sophisticated"]
tags = []

for t in tensecs:
    group_idx = df.index[df["tensec_idx"] == t].tolist()
    if not group_idx:
        continue
    total_rows = len(group_idx)
    dur = int(np.clip(np.random.randint(1, min(11, total_rows + 1)), 1, total_rows))
    start_offset = np.random.randint(0, max(1, total_rows - dur + 1))
    target_rows = group_idx[start_offset:start_offset + dur]
    attack_type = np.random.choice(attacks)
    if attack_type == "slow_drift":
        drift_rate = float(np.random.uniform(0.2, 2.0))
        inject_slow_drift(df, target_rows, drift_rate, ref_lat, ref_lon)
        tag = f"tensec{t}_drift_{drift_rate:.2f}mps_start{start_offset}_dur{dur}"
    else:
        offset = float(np.random.uniform(5.0, 50.0))
        inject_sophisticated(df, target_rows, offset, ref_lat, ref_lon)
        tag = f"tensec{t}_soph_{int(offset)}m_start{start_offset}_dur{dur}"
    df.loc[target_rows, "label"] = 1
    tags.append(tag)

# cleanup
df = df.drop(columns=["tensec_idx"])
df.to_csv(OUT_PATH, index=False)
print("Saved spoofed file to:", OUT_PATH)
print("Injected segments:")
for t in tags[:50]:
    print(" -", t)
if len(tags) > 50:
    print(f"... and {len(tags)-50} more")
