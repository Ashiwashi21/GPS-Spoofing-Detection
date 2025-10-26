import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def nmea_to_decimal(nmea):
    if pd.isna(nmea) or not isinstance(nmea, str):
        return np.nan
    parts = nmea.split()
    val = parts[0]
    direction = parts[1] if len(parts) > 1 else ""
    deg_len = 2 if len(val.split(".")[0]) <= 4 else 3
    deg = float(val[:deg_len])
    mins = float(val[deg_len:])
    dec = deg + mins / 60.0
    if direction in ("S", "W"):
        dec = -dec
    return dec

def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))
    dx = (lon - ref_lon) * m_per_lon
    dy = (lat - ref_lat) * m_per_lat
    return dx, dy

normal_csv = Path("sensor-data/normal/normal.csv")
spoofed_csv = Path("sensor-data/spoofed/spoofed.csv")
out_dir = Path("sensor-data/visuals")
out_dir.mkdir(parents=True, exist_ok=True)

if not normal_csv.exists():
    raise FileNotFoundError(f"missing {normal_csv}")
if not spoofed_csv.exists():
    raise FileNotFoundError(f"missing {spoofed_csv}")

normal = pd.read_csv(normal_csv)
spoofed = pd.read_csv(spoofed_csv)

normal["timestamp"] = pd.to_numeric(normal.get("timestamp", normal.index), errors="coerce")
spoofed["timestamp"] = pd.to_numeric(spoofed.get("timestamp", spoofed.index), errors="coerce")

normal = normal.sort_values("timestamp").reset_index(drop=True)
spoofed = spoofed.sort_values("timestamp").reset_index(drop=True)

if "lat_decimal" not in normal.columns:
    normal["lat_decimal"] = normal["latitude"].apply(nmea_to_decimal)
if "lon_decimal" not in normal.columns:
    normal["lon_decimal"] = normal["longitude"].apply(nmea_to_decimal)
if "lat_decimal" not in spoofed.columns:
    spoofed["lat_decimal"] = spoofed["latitude"].apply(nmea_to_decimal)
if "lon_decimal" not in spoofed.columns:
    spoofed["lon_decimal"] = spoofed["longitude"].apply(nmea_to_decimal)

normal = normal.dropna(subset=["lat_decimal","lon_decimal"]).reset_index(drop=True)
spoofed = spoofed.dropna(subset=["lat_decimal","lon_decimal"]).reset_index(drop=True)

min_len = min(len(normal), len(spoofed))
if min_len == 0:
    raise SystemExit("No overlapping valid GPS points found in one of the CSVs")

# align spoofed rows to nearest normal timestamp within tolerance
tolerance_s = 2
merged = pd.merge_asof(spoofed.sort_values("timestamp"),
                       normal[["timestamp","lat_decimal","lon_decimal"]].rename(
                           columns={"lat_decimal":"lat_decimal_normal","lon_decimal":"lon_decimal_normal"}
                       ).sort_values("timestamp"),
                       on="timestamp", direction="nearest", tolerance=tolerance_s)

# if merge_asof failed for many rows, fall back to index align for first min_len rows
if merged["lat_decimal_normal"].isna().mean() > 0.2:
    merged = spoofed.head(min_len).copy()
    merged["lat_decimal_normal"] = normal.head(min_len)["lat_decimal"].values
    merged["lon_decimal_normal"] = normal.head(min_len)["lon_decimal"].values

merged = merged.reset_index(drop=True)

ref_lat = merged["lat_decimal_normal"].dropna().iloc[0]
ref_lon = merged["lon_decimal_normal"].dropna().iloc[0]

dx_spoof, dy_spoof = latlon_to_meters(merged["lat_decimal"], merged["lon_decimal"], ref_lat, ref_lon)
dx_normal, dy_normal = latlon_to_meters(merged["lat_decimal_normal"], merged["lon_decimal_normal"], ref_lat, ref_lon)

merged["lat_disp_m"] = (merged["lat_decimal"] - merged["lat_decimal_normal"]) * 111320.0
merged["lon_disp_m"] = (merged["lon_decimal"] - merged["lon_decimal_normal"]) * (111320.0 * np.cos(np.deg2rad(ref_lat)))
t = merged["timestamp"] - merged["timestamp"].iloc[0]
t = t.fillna(pd.Series(range(len(merged))))


# FIGURE 1: left = placement map, right = displacement vs time
fig, axes = plt.subplots(1,2, figsize=(16,8))

ax = axes[0]
ax.plot(dx_normal, dy_normal, color="black", linewidth=1.5, alpha=0.6, label="path normal")
ax.scatter(dx_normal, dy_normal, c="lime", s=8, label="normal", alpha=0.9)
ax.scatter(dx_spoof, dy_spoof, c="red", s=8, label="spoofed", alpha=0.9)
ax.set_aspect("equal", "box")
ax.set_title("Placement: normal (green) vs spoofed (red)")
ax.set_xlabel("meters east of ref")
ax.set_ylabel("meters north of ref")
ax.legend()

ax.grid(True, linestyle="--", alpha=0.5)


ax2 = axes[1]
ax2.plot(t, merged["lat_disp_m"].fillna(0), label="lat displacement (m)")
ax2.plot(t, merged["lon_disp_m"].fillna(0), label="lon displacement (m)")
ax2.set_title("Latitude and Longitude displacement vs time")
ax2.set_xlabel("seconds since start")
ax2.set_ylabel("displacement (meters)")
ax2.legend()
ax2.grid(True)
ax2.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
fig_path = out_dir / "placement_and_displacement.png"
fig.savefig(fig_path, dpi=200)
plt.close(fig)

# FIGURE 2: circle plot centered on mean normal position
df = merged.copy()
df["label"] = df.get("label", 0).astype(int)
df["is_spoof"] = df["label"] == 1

# detect contiguous spoof segments
segments = []
in_seg = False
seg_start = None
for i, v in enumerate(df["is_spoof"]):
    if v and not in_seg:
        in_seg = True
        seg_start = i
    if not v and in_seg:
        in_seg = False
        segments.append((seg_start, i-1))
        seg_start = None
if in_seg and seg_start is not None:
    segments.append((seg_start, len(df)-1))

palette = ["#ff9fb1", "#ff6b6b", "#ff2e2e", "#ff0066", "#b3003b", "#9b59b6", "#6a0dad"]
colors = [palette[i % len(palette)] for i in range(len(segments))]

center_lat = df["lat_decimal_normal"].dropna().mean()
center_lon = df["lon_decimal_normal"].dropna().mean()

xs = []
ys = []
for i,row in df.iterrows():
    lat = row["lat_decimal"] if not pd.isna(row["lat_decimal"]) else row["lat_decimal_normal"]
    lon = row["lon_decimal"] if not pd.isna(row["lon_decimal"]) else row["lon_decimal_normal"]
    dx, dy = latlon_to_meters(lat, lon, center_lat, center_lon)
    xs.append(dx)
    ys.append(dy)
df["x_center"] = xs
df["y_center"] = ys

fig2, ax = plt.subplots(figsize=(9,9))
ax.set_aspect("equal", "box")
ax.set_title("Circle plot centered on mean normal position. Start black, end white.")
ax.set_xlabel("Meters east of center")
ax.set_ylabel("Meters north of center")
ax.grid(True, linestyle="--", alpha=0.4)


r = max(np.abs(df[["x_center","y_center"]]).max()) * 1.15
circ = plt.Circle((0,0), r, edgecolor="0.7", facecolor="none", linewidth=0.6)
ax.add_patch(circ)

ax.plot(df["x_center"], df["y_center"], color="lightgray", linewidth=0.6, alpha=0.9, zorder=1)

for idx, (s,e) in enumerate(segments):
    color = colors[idx]
    start_i = max(0, s-1)
    end_i = min(len(df)-1, e+1)
    ax.plot(df["x_center"].iloc[start_i:end_i+1], df["y_center"].iloc[start_i:end_i+1],
            color=color, linewidth=3.0, zorder=3)

start_x, start_y = df["x_center"].iloc[0], df["y_center"].iloc[0]
end_x, end_y = df["x_center"].iloc[-1], df["y_center"].iloc[-1]
ax.scatter([start_x], [start_y], c="black", s=200, edgecolor="white", zorder=6)
ax.scatter([end_x], [end_y], c="white", s=200, edgecolor="black", zorder=6)

attack_count = len(segments)
ax.text(-r*0.95, r*0.92, f"attacks: {attack_count}", fontsize=10, zorder=10)

fig2_path = out_dir / "circle_attack_trace.png"
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close(fig2)

print("Saved images:")
print(" -", fig_path)
print(" -", fig2_path)
print("Rows used:", len(merged))
print("Detected spoof segments:", attack_count)
print("If you run this on a headless Pi, open the PNGs with an image viewer or copy them to your PC to view.")


imu_cols = ["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z","temp"]

# Convert to float (skip if missing)
for col in imu_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
for col in ["accel_x","accel_y","accel_z"]:
    if col in df.columns:
        plt.plot(df[col], label=col)
plt.title("Accelerometer Data Over Time")
plt.xlabel("Sample")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
for col in ["gyro_x","gyro_y","gyro_z"]:
    if col in df.columns:
        plt.plot(df[col], label=col)
plt.title("Gyroscope Data Over Time")
plt.xlabel("Sample")
plt.ylabel("Angular Velocity (deg/s)")
plt.legend()
plt.show()

if "temp" in df.columns:
    plt.figure(figsize=(12,4))
    plt.plot(df["temp"], label="Temperature")
    plt.title("IMU Temperature Over Time")
    plt.xlabel("Sample")
    plt.ylabel("Temp (�C)")
    plt.legend()
    plt.show()
