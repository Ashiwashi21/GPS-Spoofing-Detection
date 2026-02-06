#goal of this script: collect data from the sensors, decode the data, write it into a csv with timestamps every second
#imports
import serial, time, csv #reading GPS + writing data
from pathlib import Path #building file paths
from mpu6050.mpu6050 import mpu6050 #imu sensor
from datetime import datetime #date and time

#creates file for sensor-data using the timestamp as the name
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = Path(__file__).parent.parent.parent / "sensor-data" / "normal" / f"normal_{timestamp_str}.csv"

#initializes sensors, turn them on and open serial ports (hardware)
imu_sensor = mpu6050(0x68)
ser = serial.Serial("/dev/serial0", 9600, timeout=1)

#parses gps nmea sentences 
#latitude, longitude, altitude, satellites, hdop
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

#speed
def parse_gprmc(parts):
    return {"speed_knots": parts[7]} if len(parts) > 7 else {}

#pdop/vdop
def parse_gpgsa(parts):
    return {"pdop": parts[15], "vdop": parts[16]} if len(parts) > 16 else {}

#opens csv, writes headers
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "timestamp","latitude","longitude","altitude_m","satellites","hdop",
        "speed_knots","pdop","vdop","accel_x","accel_y","accel_z",
        "gyro_x","gyro_y","gyro_z","temp"
    ])

    #time for how long the script is running for
    start = time.time()

    #dictionary with all gps fields set to empty strings (these update when a gps sentence arrives)
    gps = {k: "" for k in ["latitude","longitude","altitude_m","satellites","hdop","speed_knots","pdop","vdop"]}

    #main loop running for 90 minutes, decodes gps sentence, identifies sentence, updates the gps dictionary
    while time.time() - start < 5400:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line.startswith("$GPGGA"):
            gps.update(parse_gpgga(line.split(",")))
        elif line.startswith("$GPRMC"):
            gps.update(parse_gprmc(line.split(",")))
        elif line.startswith("$GPGSA"):
            gps.update(parse_gpgsa(line.split(",")))

        #reads imu data
        accel = imu_sensor.get_accel_data()
        gyro = imu_sensor.get_gyro_data()
        temp = imu_sensor.get_temp()

        #updates all values every second into the csv
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
