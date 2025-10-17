import serial, time, csv
from pathlib import Path

# File path relative to this script
csv_path = Path(__file__).parent.parent.parent / "sensor-data" / "normal" / "normal.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)  # make folder if missing

with serial.Serial('/dev/serial0', 9600, timeout=1) as ser, open(csv_path, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    start = time.time()
    try:
        while time.time() - start < 600:  # run for 60 seconds
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            if line.startswith('$GPGGA'):
                parts = line.split(',')
                lat = parts[2] + " " + parts[3]
                lon = parts[4] + " " + parts[5]
                alt = parts[9] + " " + parts[10]
                sats = parts[7]
                hdop = parts[8]
                writer.writerow([time.time(), lat, lon, alt, sats, hdop])
                print(f"GGA -> Lat:{lat}, Lon:{lon}, Alt:{alt}, Sats:{sats}, HDOP:{hdop}")
            elif line.startswith('$GPRMC'):
                parts = line.split(',')
                speed = parts[7]
                course = parts[8]
                writer.writerow([time.time(), speed, course])
                print(f"RMC -> Speed:{speed} knots, Course:{course}")
            elif line.startswith('$GPGSV'):
                parts = line.split(',')
                total_sats = parts[3]
                writer.writerow([time.time(), total_sats])
                print(f"GSV -> Satellites in view: {total_sats}")
            elif line.startswith('$GPGSA'):
                parts = line.split(',')
                pdop = parts[15]
                vdop = parts[16]
                writer.writerow([time.time(), pdop, vdop])
                print(f"GSA -> PDOP:{pdop}, VDOP:{vdop}")
    except KeyboardInterrupt:
        print("Stopping GPS data reception.")
