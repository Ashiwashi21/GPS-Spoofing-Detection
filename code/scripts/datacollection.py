import serial, time, csv
from pathlib import Path

csv_path = Path(__file__).parent.parent.parent / "sensor-data" / "normal" / "normal.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

with serial.Serial('/dev/serial0', 9600, timeout=1) as ser, open(csv_path, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp", "latitude", "longitude", "altitude_m", "satellites", "hdop", "speed_knots", "pdop", "vdop"])

    start = time.time()
    data = {"lat": "", "lon": "", "alt": "", "sats": "", "hdop": "", "speed": "", "pdop": "", "vdop": ""}

    while time.time() - start < 600: 
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        parts = line.split(",")
        if line.startswith("$GPGGA"):
            data["lat"] = parts[2] + " " + parts[3]
            data["lon"] = parts[4] + " " + parts[5]
            data["alt"] = parts[9]
            data["sats"] = parts[7]
            data["hdop"] = parts[8]

        elif line.startswith("$GPRMC"):
            data["speed"] = parts[7]

        elif line.startswith("$GPGSA"):
            data["pdop"] = parts[15]
            data["vdop"] = parts[16]

        if all(data.values()):
            writer.writerow([
                int(time.time()),
                data["lat"],
                data["lon"],
                data["alt"],
                data["sats"],
                data["hdop"],
                data["speed"],
                data["pdop"],
                data["vdop"]
            ])
            csvfile.flush()
            data = {k: "" for k in data}  # reset for next reading

print("10-minute data collection complete.")
