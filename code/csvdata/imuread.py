import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sensor-data/raw/imu.csv")
time_sec = df["timestamp"] - df["timestamp"].iloc[0]

plt.figure()
plt.plot(time_sec, df["accel_x"], label="X")
plt.plot(time_sec, df["accel_y"], label="Y")
plt.plot(time_sec, df["accel_z"], label="Z")
plt.title("Accelerometer")
plt.xlabel("Time (s)")
plt.ylabel("m/s�")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(time_sec, df["gyro_x"], label="X")
plt.plot(time_sec, df["gyro_y"], label="Y")
plt.plot(time_sec, df["gyro_z"], label="Z")
plt.title("Gyroscope")
plt.xlabel("Time (s)")
plt.ylabel("�/s")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(time_sec, df["temp"], label="Temp", color="red")
plt.title("Temperature")
plt.xlabel("Time (s)")
plt.ylabel("�C")
plt.legend()
plt.grid(True)

plt.show()  