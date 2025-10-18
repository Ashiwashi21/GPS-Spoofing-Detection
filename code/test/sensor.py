import serial
import time
import csv
from pathlib import Path

csv_path = Path(__file__).parent.parent.parent / "sensor-data" / "raw" / "gps.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

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

with serial.Serial('/dev/serial0', 9600, timeout=1) as ser, open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp", "latitude", "longitude", "altitude_m", "satellites", "hdop", "speed_knots", "pdop", "vdop"])

    start_time = time.time()
    gps_data = {k: "" for k in ["latitude", "longitude", "altitude_m", "satellites", "hdop", "speed_knots", "pdop", "vdop"]}

    while time.time() - start_time < 1800:
        try:
            line = ser.readline().decode("utf-8").strip()
        except UnicodeDecodeError:
            print("Failed to decode line")
            continue

        if not line:
            continue

        parts = line.split(",")

        if line.startswith("$GPGGA"):
            gps_data.update(parse_gpgga(parts))
        elif line.startswith("$GPRMC"):
            gps_data.update(parse_gprmc(parts))
        elif line.startswith("$GPGSA"):
            gps_data.update(parse_gpgsa(parts))

        if all(gps_data.values()):
            timestamp = int(time.time())
            writer.writerow([
                timestamp,
                gps_data["latitude"],
                gps_data["longitude"],
                gps_data["altitude_m"],
                gps_data["satellites"],
                gps_data["hdop"],
                gps_data["speed_knots"],
                gps_data["pdop"],
                gps_data["vdop"]
            ])
            csvfile.flush()
            gps_data = {k: "" for k in gps_data}
