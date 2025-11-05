import numpy as np
from pathlib import Path
import math
import random

R_EARTH = 6371000.0
Path("sensor-data/testing").mkdir(parents=True, exist_ok=True)

def meters_to_latlon(dx, dy, lat0):
    dlat = dy / R_EARTH
    dlon = dx / (R_EARTH * math.cos(math.radians(lat0)))
    return math.degrees(dlat), math.degrees(dlon)

def generate_gps_path(n_rows, timestamps, start_lat, start_lon, base_alt=300.0, avg_speed=1.3, spoof=False):
    lat = np.zeros(n_rows)
    lon = np.zeros(n_rows)
    alt = np.zeros(n_rows)
    speed_knots = np.zeros(n_rows)
    lat[0], lon[0], alt[0] = start_lat, start_lon, base_alt
    heading = np.random.uniform(0, 2 * math.pi)
    for i in range(1, n_rows):
        dt = max(1.0, timestamps[i] - timestamps[i - 1])
        speed = np.random.normal(avg_speed, 0.3)
        if spoof and np.random.rand() < 0.1:
            heading += math.radians(np.random.uniform(-90, 90))
            speed *= np.random.uniform(1.5, 3.0)
        heading += math.radians(np.random.normal(0, 5 if not spoof else 2))
        dist = speed * dt
        dx = math.sin(heading) * dist
        dy = math.cos(heading) * dist
        dlat, dlon = meters_to_latlon(dx, dy, lat[i - 1])
        lat[i] = lat[i - 1] + dlat + np.random.normal(0, 0.0002 if not spoof else 0.0003)
        lon[i] = lon[i - 1] + dlon + np.random.normal(0, 0.0002 if not spoof else 0.0003)
        alt[i] = base_alt + np.random.normal(0, 1.5 if not spoof else 2.0)
        speed_knots[i] = speed / 0.514444
    return lat, lon, alt, speed_knots

def generate_imu(seq, n_augments=10, spoof=False):
    augmented = []
    for _ in range(n_augments):
        noise = np.random.normal(0, [0.3, 0.3, 0.3, 1.5, 1.5, 1.5, 0.2], size=seq.shape)
        drift = np.linspace(0, np.random.uniform(0.5, 2.0 if not spoof else 3.0), seq.shape[1])
        drift *= np.random.choice([-1, 1])
        pattern = np.sin(np.linspace(0, np.pi, seq.shape[1])) * np.random.uniform(0.2, 0.5)
        if spoof:
            lag = np.roll(np.cos(np.linspace(0, np.pi * 2, seq.shape[1])) * np.random.uniform(0.3, 0.6), shift=np.random.randint(1, 5))
            augmented_seq = seq + noise + drift + lag
        else:
            augmented_seq = seq + noise + drift + pattern
        augmented.append(augmented_seq)
    return np.array(augmented)

def create_window(label):
    key_features = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temp"]
    base = np.random.normal(0, 1, (1, len(key_features)))
    imu = generate_imu(base, n_augments=10, spoof=bool(label)).reshape(10, len(key_features))
    lat, lon, alt, speed = generate_gps_path(10, np.arange(10), 34.035, -84.125, spoof=bool(label))
    gps = np.stack([lat, lon, alt, speed], axis=1)
    full = np.concatenate([imu, gps], axis=1)
    return full, label

samples = []
for _ in range(250):
    samples.append(create_window(0))
for _ in range(250):
    samples.append(create_window(1))

random.shuffle(samples)

for i, (x, y) in enumerate(samples):
    np.save(f"sensor-data/testing/x_{i}.npy", x)
    np.save(f"sensor-data/testing/y_{i}.npy", np.array([y]))
