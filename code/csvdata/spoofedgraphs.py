import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

normal_csv = Path("sensor-data/normal/normal.csv")
spoofed_csv = Path("sensor-data/spoofed/spoofed.csv")
out_dir = Path("sensor-data/visuals")
out_dir.mkdir(parents=True, exist_ok=True)

normal = pd.read_csv(normal_csv)
spoofed = pd.read_csv(spoofed_csv)

# Make sure lat/lon columns exist
for col in ["latitude","longitude"]:
    if col not in normal.columns or col not in spoofed.columns:
        raise SystemExit(f"{col} missing in one of the CSVs")

# Drop NaNs
normal = normal.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
spoofed = spoofed.dropna(subset=["latitude","longitude"]).reset_index(drop=True)

# Align by index
min_len = min(len(normal), len(spoofed))
normal = normal.head(min_len)
spoofed = spoofed.head(min_len)

# Reference point
ref_lat = normal["latitude"].iloc[0]
ref_lon = normal["longitude"].iloc[0]

# Convert to meters
def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))
    dx = (lon - ref_lon) * m_per_lon
    dy = (lat - ref_lat) * m_per_lat
    return dx, dy

dx_spoof, dy_spoof = latlon_to_meters(spoofed["latitude"], spoofed["longitude"], ref_lat, ref_lon)
dx_normal, dy_normal = latlon_to_meters(normal["latitude"], normal["longitude"], ref_lat, ref_lon)

lat_disp_m = (spoofed["latitude"] - normal["latitude"]) * 111320.0
lon_disp_m = (spoofed["longitude"] - normal["longitude"]) * (111320.0 * np.cos(np.deg2rad(ref_lat)))

# Plot
fig, axes = plt.subplots(1,2, figsize=(16,8))

ax = axes[0]
ax.plot(dx_normal, dy_normal, color="green", linewidth=1.5, alpha=0.6, label="normal path")
ax.scatter(dx_spoof, dy_spoof, c="red", s=8, label="spoofed", alpha=0.9)
ax.set_aspect("equal", "box")
ax.set_title("Placement of spoofed coordinates")
ax.set_xlabel("east displacement (m)")
ax.set_ylabel("north displacement (m)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

ax2 = axes[1]
ax2.plot(lat_disp_m, label="lat displacement (m)")
ax2.plot(lon_disp_m, label="lon displacement (m)")
ax2.set_title("Latitude and Longitude displacement")
ax2.set_xlabel("row index")
ax2.set_ylabel("displacement (m)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
fig_path = out_dir / "placement_and_displacement.png"
fig.savefig(fig_path, dpi=200)
plt.close(fig)

print("Plot saved to", fig_path)
def plot_spoofed_sensor(columns, title, filename, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in columns:
        if col not in spoofed.columns:
            print(f"Skipping {col} � missing in spoofed.csv")
            continue
        ax.plot(spoofed[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("Row Index")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)

# Accelerometer (spoofed only)
plot_spoofed_sensor(
    columns=["accel_x", "accel_y", "accel_z"],
    title="Spoofed Accelerometer Data",
    filename="spoofed_accel.png",
    ylabel="Acceleration (m/s�)"
)

# Gyroscope (spoofed only)
plot_spoofed_sensor(
    columns=["gyro_x", "gyro_y", "gyro_z"],
    title="Spoofed Gyroscope Data",
    filename="spoofed_gyro.png",
    ylabel="Angular Velocity (rad/s)"
)

