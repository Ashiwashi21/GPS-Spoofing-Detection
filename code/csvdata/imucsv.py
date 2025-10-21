from mpu6050.mpu6050 import mpu6050
import time
import csv
from pathlib import Path

csv_path = Path(__file__).parent.parent.parent / "sensor-data" / "raw" / "imu.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

def accel(accel_data):
    return {
        "accel_x": accel_data['x'],
        "accel_y": accel_data['y'],
        "accel_z": accel_data['z']
    }

def gyro(gyro_data):
    return {
        "gyro_x": gyro_data['x'],
        "gyro_y": gyro_data['y'],
        "gyro_z": gyro_data['z']
    }

def temp(temp_data):
    return {"temp": temp_data}



with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temp"])

    start_time = time.time()
    imu_data = {k: "" for k in ["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z","temp"]}

    while time.time() - start_time < 1800:
        sensor = mpu6050(0x68)
        accel_data = sensor.get_accel_data()
        gyro_data = sensor.get_gyro_data()
        temp_data = sensor.get_temp()

        
        imu_data.update(accel(accel_data))
        imu_data.update(gyro(gyro_data))
        imu_data.update(temp(temp_data))
        

        if all(imu_data.values()):
            timestamp = int(time.time())
            writer.writerow([
                timestamp,
                imu_data["accel_x"],
                imu_data["accel_y"],
                imu_data["accel_z"],
                imu_data["gyro_x"],
                imu_data["gyro_y"],
                imu_data["gyro_z"],
                imu_data["temp"]
            ])
            csvfile.flush()
            imu_data = {k: "" for k in imu_data}
        time.sleep(1)