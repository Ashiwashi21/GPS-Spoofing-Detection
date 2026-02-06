#goal of this script: spoof all data types by 20% and put it in a new spoofed csv
#imports
import numpy as np #math
import pandas as pd #dataframes
import os #handles files

#makes randomness repeatable so spoofing is consistent
np.random.seed(42)

#creates output folder incase the folder isn't there
os.makedirs("sensor-data/spoofed", exist_ok=True)

#reads normal csv and builds spoofed csv (must contain same rows)
normal_df = pd.read_csv("sensor-data/normal/normal.csv")
num_rows = len(normal_df)
spoofed_df = normal_df.copy()

#spoof accelerometer data 
'''
process:
1) extracts acceleration columns into numpy array
2) randomly selects 20% of rows to spoof
3) creates slow drift (gradual offset)
4) creates fast jitter (random noise)
5) adds drift + jutter to selected rows
6) prevents impossible acceleration values
'''
accel_spoofed = spoofed_df[["accel_x", "accel_y", "accel_z"]].values.copy()
accel_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
accel_drift = np.random.normal(0, 0.5, size=(len(accel_indices), 3))
accel_jitter = np.random.normal(0, 0.2, size=(len(accel_indices), 3))
accel_spoofed[accel_indices] += accel_drift + accel_jitter
accel_spoofed = np.clip(accel_spoofed, -12, 12)

#spoof gyroscope data
'''
process:
1) extracts gyroscope data into numpy array
2) selects 20% of rows to spoof
3) in the loop (going through each axis):
-3a) creates smoothed lag (drifting rotation)
-3b) creates sharp jitter noise
-3c) applies spoofing to selected rows
4) (outside of loop) keeps values within realisitic limits
'''
gyro_spoofed = spoofed_df[["gyro_x", "gyro_y", "gyro_z"]].values.copy()
gyro_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
for i in range(3):
    lag = np.convolve(np.random.normal(0, 1, len(gyro_indices)), np.ones(10)/10, mode='same')
    jitter = np.random.normal(0, 1.0, size=len(gyro_indices))
    gyro_spoofed[gyro_indices, i] += lag + jitter
gyro_spoofed = np.clip(gyro_spoofed, -50, 50)

#spoof latitude + longitude data
'''
process:
1) extracts latitude and longitude data into numpy arrays
2) selects 20% of rows to spoof
3) creates cumulative drift (gps drifts overtime)
4) applies drift to selected rows
'''
lat_spoofed = spoofed_df["latitude"].values.copy()
lon_spoofed = spoofed_df["longitude"].values.copy()
gps_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
lat_noise = np.cumsum(np.random.normal(0, 0.00005, size=len(gps_indices)))
lon_noise = np.cumsum(np.random.normal(0, 0.00005, size=len(gps_indices)))
lat_spoofed[gps_indices] += lat_noise
lon_spoofed[gps_indices] += lon_noise

#spoof altitude 
'''
process:
1) extracts altitude data into numpy arrays
2) selects 20% of the rows to spoof
3) adds small altitude noise
4) keeps altitude in realisitc ranges
'''
alt_spoofed = spoofed_df["altitude"].values.copy()
alt_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
alt_spoofed[alt_indices] += np.random.normal(0, 0.3, size=len(alt_indices))
alt_spoofed = np.clip(alt_spoofed, 298.5, 301.5)

#spoof speed
'''
process:
1) extracts speed data into numpy arrays
2) selects 20% of the rows to spoof
3) add speed noise
4) keeps speed within realistic limits
'''
speed_spoofed = spoofed_df["speed_knots"].values.copy()
speed_indices = np.random.choice(num_rows, size=int(0.2 * num_rows), replace=False)
speed_spoofed[speed_indices] += np.random.normal(0, 0.5, size=len(speed_indices))
speed_spoofed = np.clip(speed_spoofed, 0.5, 4.5)

#build the dataframe for the spoofed data
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

#saves spoofed data
output_path = "sensor-data/spoofed/spoofed.csv"
df_spoofed.to_csv(output_path, index=False)

#sanity checks
print(df_spoofed.head())
print(f"\n? Spoofed data generated with {len(df_spoofed)} rows and saved to spoofed.csv")