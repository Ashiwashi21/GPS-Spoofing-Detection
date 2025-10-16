import serial, time

with serial.Serial('/dev/serial0', 9600, timeout=1) as ser:
    start = time.time()
    try:
        while time.time() - start < 60:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue

            if line.startswith('$GPGGA'):
                parts = line.split(',')
                lat = parts[2] + " " + parts[3]
                lon = parts[4] + " " + parts[5]
                alt = parts[9] + " " + parts[10]
                satellites = parts[7]
                hdop = parts[8]
                print(f"GGA -> Lat:{lat}, Lon:{lon}, Alt:{alt}, Sats:{satellites}, HDOP:{hdop}")

            elif line.startswith('$GPRMC'):
                parts = line.split(',')
                speed = parts[7] 
                course = parts[8]
                print(f"RMC -> Speed:{speed} knots, Course:{course}")

            elif line.startswith('$GPGSV'):
                parts = line.split(',')
                total_sats = parts[3]
                print(f"GSV -> Satellites in view: {total_sats}")
                
            elif line.startswith('$GPGSA'):
                parts = line.split(',')
                pdop = parts[15]
                vdop = parts[16]
                print(f"GSA -> PDOP:{pdop}, VDOP:{vdop}")

    except KeyboardInterrupt:
        print("Stopping GPS data reception.")
