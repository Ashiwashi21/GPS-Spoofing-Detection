import serial
import time
import csv
from pathlib import Path
from mpu6050.mpu6050 import mpu6050

# ---------- FILE SETUP ----------
csv_path = Path(__file__).parent.parent.parent / "sensor-data" / "normal" / "normal.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

# ---------- GPS PARSERS ----------
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

# ---------- GPS FUNCTION ----------
def get_gps_data(ser):
    gps_data = {k: "" for k in ["latitude", "longitude", "altitude_m", "satellites", "hdop", "speed_knots", "pdop", "vdop"]}
    try:
        line = ser.readline().decode("utf-8").strip()
        if not line:
            return gps_data

        parts = line.split(",")
        if line.startswith("$GPGGA"):
            gps_data.update(parse_gpgga(parts))
        elif line.startswith("$GPRMC"):
            gps_data.update(parse_gprmc(parts))
        elif line.startswith("$GPGSA"):
            gps_data.update(parse_gpgsa(parts))
    except Exception:
        pass
    return gps_data

# ---------- IMU FUNCTION ----------
def get_imu_data(sensor):
    accel_data = sensor.get_accel_data()
    gyro_data = sensor.get_gyro_data()
    temp_data = sensor.get_temp()

    return {
        "accel_x": accel_data["x"],
        "accel_y": accel_data["y"],
        "accel_z": accel_data["z"],
        "gyro_x": gyro_data["x"],
        "gyro_y": gyro_data["y"],
        "gyro_z": gyro_data["z"],
        "temp": temp_data,
    }

# ---------- MAIN LOOP ----------
def main(duration=1800):
    print(f"Logging GPS + IMU for {duration} seconds...")
    header = [
        "timestamp",
        "latitude", "longitude", "altitude_m", "satellites", "hdop", "speed_knots", "pdop", "vdop",
        "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temp"
    ]

    with serial.Serial('/dev/serial0', 9600, timeout=1) as ser, \
         open(csv_path, "w", newline="") as csvfile:
        
        writer = csv.writer(csvfile)
        writer.writerow(header)
        sensor = mpu6050(0x68)
        start = time.time()

        while time.time() - start < duration:
            gps_data = get_gps_data(ser)
            imu_data = get_imu_data(sensor)
            timestamp = int(time.time())

            writer.writerow([
                timestamp,
                gps_data["latitude"], gps_data["longitude"], gps_data["altitude_m"],
                gps_data["satellites"], gps_data["hdop"], gps_data["speed_knots"],
                gps_data["pdop"], gps_data["vdop"],
                imu_data["accel_x"], imu_data["accel_y"], imu_data["accel_z"],
                imu_data["gyro_x"], imu_data["gyro_y"], imu_data["gyro_z"], imu_data["temp"]
            ])
            csvfile.flush()
            time.sleep(1)

    print(f"Done. Data saved to: {csv_path}")

# ---------- RUN ----------
if __name__ == "__main__":
    main(1800)  # change to 1800 for 30 minutes
