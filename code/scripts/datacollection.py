import serial
import time
import csv
from pathlib import Path
from mpu6050.mpu6050 import mpu6050

csv_path = Path(__file__).parent.parent.parent / "sensor-data" / "normal" / "normal.csv"


imu_sensor = mpu6050(0x68)

ser = serial.Serial('/dev/serial0', 9600, timeout=1)

def parse_gpgga(parts):
    return {
        "latitude": parts[2] + " " + parts[3],
        "longitude": parts[4] + " " + parts[5],
        "altitude_m": parts[9],
        "satellites": parts[7],
        "hdop": parts[8],
    }

def parse_gprmc(parts):
    return {"speed_knots": parts[7]}

def parse_gpgsa(parts):
    return {"pdop": parts[15], "vdop": parts[16]}

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "timestamp", "latitude", "longitude", "altitude_m", "satellites", "hdop", 
        "speed_knots", "pdop", "vdop", "accel_x", "accel_y", "accel_z", 
        "gyro_x", "gyro_y", "gyro_z", "temp"
    ])

    start_time = time.time()
    duration = 5 * 60
    gps_data = {k: "" for k in ["latitude", "longitude", "altitude_m", "satellites", 
                                "hdop", "speed_knots", "pdop", "vdop"]}

    while time.time() - start_time < duration:
        try:
            line = ser.readline().decode("utf-8").strip()
        except UnicodeDecodeError:
            continue

        if line.startswith("$GPGGA"):
            parts = line.split(",")
            gps_data.update(parse_gpgga(parts))
        elif line.startswith("$GPRMC"):
            parts = line.split(",")
            gps_data.update(parse_gprmc(parts))
        elif line.startswith("$GPGSA"):
            parts = line.split(",")
            gps_data.update(parse_gpgsa(parts))

        accel_data = imu_sensor.get_accel_data()
        gyro_data = imu_sensor.get_gyro_data()
        temp_data = imu_sensor.get_temp()

        timestamp = int(time.time())
        writer.writerow([
            timestamp,
            gps_data.get("latitude", ""),
            gps_data.get("longitude", ""),
            gps_data.get("altitude_m", ""),
            gps_data.get("satellites", ""),
            gps_data.get("hdop", ""),
            gps_data.get("speed_knots", ""),
            gps_data.get("pdop", ""),
            gps_data.get("vdop", ""),
            accel_data['x'], accel_data['y'], accel_data['z'],
            gyro_data['x'], gyro_data['y'], gyro_data['z'],
            temp_data
        ])
        csvfile.flush()

        gps_data = {k: "" for k in gps_data}

        time.sleep(1)
