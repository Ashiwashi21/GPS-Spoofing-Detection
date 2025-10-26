import serial, time, csv
from pathlib import Path
from mpu6050.mpu6050 import mpu6050

csv_path = Path(__file__).parent.parent.parent / "sensor-data" / "normal" / "normal.csv"
imu_sensor = mpu6050(0x68)
ser = serial.Serial("/dev/serial0", 9600, timeout=1)

def parse_gpgga(parts):
    if len(parts) > 5 and parts[2] and parts[4]:
        return {
            "latitude": parts[2] + " " + parts[3],
            "longitude": parts[4] + " " + parts[5],
            "altitude_m": parts[9],
            "satellites": parts[7],
            "hdop": parts[8]
        }
    return {}

def parse_gprmc(parts):
    return {"speed_knots": parts[7]} if len(parts) > 7 else {}

def parse_gpgsa(parts):
    return {"pdop": parts[15], "vdop": parts[16]} if len(parts) > 16 else {}

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "timestamp","latitude","longitude","altitude_m","satellites","hdop",
        "speed_knots","pdop","vdop","accel_x","accel_y","accel_z",
        "gyro_x","gyro_y","gyro_z","temp"
    ])

    start = time.time()
    gps = {k: "" for k in ["latitude","longitude","altitude_m","satellites","hdop","speed_knots","pdop","vdop"]}

    while time.time() - start < 3600:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line.startswith("$GPGGA"):
            gps.update(parse_gpgga(line.split(",")))
        elif line.startswith("$GPRMC"):
            gps.update(parse_gprmc(line.split(",")))
        elif line.startswith("$GPGSA"):
            gps.update(parse_gpgsa(line.split(",")))

        accel = imu_sensor.get_accel_data()
        gyro = imu_sensor.get_gyro_data()
        temp = imu_sensor.get_temp()

        writer.writerow([
            int(time.time()),
            gps.get("latitude",""), gps.get("longitude",""),
            gps.get("altitude_m",""), gps.get("satellites",""),
            gps.get("hdop",""), gps.get("speed_knots",""),
            gps.get("pdop",""), gps.get("vdop",""),
            accel["x"], accel["y"], accel["z"],
            gyro["x"], gyro["y"], gyro["z"],
            temp
        ])
        csvfile.flush()
        time.sleep(1)
