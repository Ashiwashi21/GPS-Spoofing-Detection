import numpy as np
import pandas as pd
import os

# --- Setup ---
np.random.seed(42)
os.makedirs("sensor-data/spoofed", exist_ok=True)

# Load real normal data
normal_df = pd.read_csv("sensor-data/normal/normal.csv")
num_rows = len(normal_df)
spoofed_df = normal_df.copy()

# --- IMU Spoofing ---
# ? Original synthetic accel generation (too unrealistic)
# accel_base = np.random.normal(0, 2.5, size=(num_rows, 3))

# ? Fix: use real accel data and inject subtle drift + jitter into segments
accel_spoofed = spoofed_df[["accel_x", "accel_y", "accel_z"]].values.copy()
accel_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
accel_drift = np.random.normal(0, 0.5, size=(len(accel_indices), 3))
accel_jitter = np.random.normal(0, 0.2, size=(len(accel_indices), 3))
accel_spoofed[accel_indices] += accel_drift + accel_jitter
accel_spoofed = np.clip(accel_spoofed, -12, 12)

# ? Original synthetic gyro generation (too high variance)
# gyro_base = np.random.normal(0, 10, size=(num_rows, 3))

# ? Fix: use real gyro data and inject smooth lag + jitter per axis
gyro_spoofed = spoofed_df[["gyro_x", "gyro_y", "gyro_z"]].values.copy()
gyro_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
for i in range(3):
    lag = np.convolve(np.random.normal(0, 1, len(gyro_indices)), np.ones(10)/10, mode='same')
    jitter = np.random.normal(0, 1.0, size=len(gyro_indices))
    gyro_spoofed[gyro_indices, i] += lag + jitter
gyro_spoofed = np.clip(gyro_spoofed, -50, 50)

# --- GPS Spoofing ---
# ? Original sinusoidal drift (too clean and predictable)
# lat_drift = 0.0003 * np.sin(np.linspace(0, 12 * np.pi, num_rows))
# lon_drift = 0.0003 * np.cos(np.linspace(0, 12 * np.pi, num_rows))

# ? Fix: random walk + jitter for realistic GPS drift
lat_spoofed = spoofed_df["latitude"].values.copy()
lon_spoofed = spoofed_df["longitude"].values.copy()
gps_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
lat_noise = np.cumsum(np.random.normal(0, 0.00005, size=len(gps_indices)))
lon_noise = np.cumsum(np.random.normal(0, 0.00005, size=len(gps_indices)))
lat_spoofed[gps_indices] += lat_noise
lon_spoofed[gps_indices] += lon_noise

# ? Original synthetic altitude base
# alt_base = 300 + np.random.normal(0, 0.5, size=num_rows)

# ? Fix: use real altitude and perturb slightly
alt_spoofed = spoofed_df["altitude"].values.copy()
alt_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
alt_spoofed[alt_indices] += np.random.normal(0, 0.3, size=len(alt_indices))
alt_spoofed = np.clip(alt_spoofed, 298.5, 301.5)

# ? Speed spoofing: perturb real speed values
speed_spoofed = spoofed_df["speed_knots"].values.copy()
speed_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
speed_spoofed[speed_indices] += np.random.normal(0, 0.5, size=len(speed_indices))
speed_spoofed = np.clip(speed_spoofed, 0.5, 4.5)

# --- Combine and Save ---
df_spoofed = pd.DataFrame({
    "accel_x": accel_spoofed[:, 0],
    "accel_y": accel_spoofed[:, 1],
    "accel_z": accel_spoofed[:, 2],
    "gyro_x": gyro_spoofed[:, 0],
    "gyro_y": gyro_spoofed[:, 1],
    "gyro_z": gyro_spoofed[:, 2],
    "latitude": lat_spoofed,
    "longitude": lon_spoofed,
    "altitude": alt_spoofed,
    "speed_knots": speed_spoofed
})

output_path = "sensor-data/spoofed/spoofed.csv"
df_spoofed.to_csv(output_path, index=False)

# --- Output Summary ---
print(df_spoofed.head())
print(f"\n? Spoofed data generated with {len(df_spoofed)} rows and saved to spoofed.csv")

