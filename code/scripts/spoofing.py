import math
import numpy as np
import pandas as pd
from pathlib import Path

R_EARTH = 6371000.0  # meters

# ---------- GPS HELPERS ----------

def meters_to_latlon(dx, dy, lat0):
    dlat = dy / R_EARTH
    dlon = dx / (R_EARTH * math.cos(math.radians(lat0)))
    return math.degrees(dlat), math.degrees(dlon)

def generate_spoofed_gps_path(n_rows, timestamps, start_lat, start_lon, base_alt=300.0, avg_speed=1.3):
    lat = np.zeros(n_rows)
    lon = np.zeros(n_rows)
    alt = np.zeros(n_rows)
    speed_knots = np.zeros(n_rows)
    lat[0], lon[0], alt[0] = start_lat, start_lon, base_alt
    heading = np.random.uniform(0, 2 * math.pi)

    for i in range(1, n_rows):
        dt = max(1.0, timestamps[i] - timestamps[i - 1])
        speed = np.random.normal(avg_speed, 0.2)

        # Inject subtle spoofing: fake loops, drift, and burst speed
        if np.random.rand() < 0.1:
            heading += math.radians(np.random.uniform(-90, 90))
            speed *= np.random.uniform(1.5, 3.0)

        heading += math.radians(np.random.normal(0, 2))
        dist = speed * dt
        dx = math.sin(heading) * dist
        dy = math.cos(heading) * dist
        dlat, dlon = meters_to_latlon(dx, dy, lat[i - 1])

        lat[i] = lat[i - 1] + dlat + np.random.normal(0, 0.0003)
        lon[i] = lon[i - 1] + dlon + np.random.normal(0, 0.0003)
        alt[i] = base_alt + np.random.normal(0, 2.0)
        speed_knots[i] = speed / 0.514444
    return lat, lon, alt, speed_knots

# ---------- IMU SPOOFING ----------

def generate_spoofed_imu(seq, n_augments=15):
    augmented = []
    for _ in range(n_augments):
        noise = np.random.normal(0, [0.3, 0.3, 0.3, 1.2, 1.2, 1.2, 0.2], size=seq.shape)
        drift = np.linspace(0, np.random.uniform(1.0, 3.0), seq.shape[1])
        drift *= np.random.choice([-1, 1])
        spoof_pattern = np.cos(np.linspace(0, np.pi * 2, seq.shape[1])) * np.random.uniform(0.3, 0.6)

        # Inject fake IMU consistency: match GPS heading but with hidden lag
        lag = np.roll(spoof_pattern, shift=np.random.randint(1, 5))
        spoofed_seq = seq + noise + drift + lag
        augmented.append(spoofed_seq)
    return np.array(augmented)

# ---------- MAIN ----------

def main():
    src = Path("sensor-data/spoofed/spoofed.csv")
    df = pd.read_csv(src)
    key_features = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temp"]
    df = df[key_features].dropna().astype(float)
    timestamps = np.arange(len(df)).astype(float)

    # 1. Generate spoofed IMU data
    all_aug = []
    for _, row in df.iterrows():
        seq = row.values.reshape(1, -1)
        aug_rows = generate_spoofed_imu(seq, n_augments=15)
        all_aug.append(aug_rows)
    spoofed_imu = np.vstack(all_aug).reshape(-1, len(key_features))
    imu_df = pd.DataFrame(spoofed_imu, columns=key_features)

    # 2. Generate spoofed GPS path
    start_lat, start_lon = 34.035, -84.125
    n_rows = len(imu_df)
    lat, lon, alt, speed = generate_spoofed_gps_path(n_rows, timestamps=np.arange(n_rows),
                                                     start_lat=start_lat, start_lon=start_lon)

    gps_df = pd.DataFrame({
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
        "speed_knots": speed
    })

    # 3. Combine IMU + GPS
    full_df = pd.concat([imu_df, gps_df], axis=1)

    out_path = Path("sensor-data/spoofed/spoofed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)

    print(f"?? Generated {len(full_df)} spoofed rows -> {out_path}")
    print(full_df.head())

if __name__ == "__main__":
    main()
