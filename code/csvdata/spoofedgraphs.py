

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
