import pandas as pd
import folium

def nmea_to_decimal(coord, direction):
    if pd.isna(coord) or coord == "":
        return None
    deg = int(float(coord) / 100)
    minutes = float(coord) - deg * 100
    decimal = deg + minutes / 60
    if direction in ["S", "W"]:
        decimal = -decimal
    return decimal

df = pd.read_csv("/home/pi/GPS-Spoofing-Detection/sensor-data/raw/gps.csv")

df[["lat_val", "lat_dir"]] = df["latitude"].astype(str).str.split(" ", expand=True)
df[["lon_val", "lon_dir"]] = df["longitude"].astype(str).str.split(" ", expand=True)

df["lat_decimal"] = df.apply(lambda x: nmea_to_decimal(x["lat_val"], x["lat_dir"]), axis=1)
df["lon_decimal"] = df.apply(lambda x: nmea_to_decimal(x["lon_val"], x["lon_dir"]), axis=1)

df = df.dropna(subset=["lat_decimal", "lon_decimal"])

start_coords = [df["lat_decimal"].iloc[0], df["lon_decimal"].iloc[0]]
m = folium.Map(location=start_coords, zoom_start=15)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat_decimal"], row["lon_decimal"]],
        radius=1,
        color="red",
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

m.save("/home/pi/GPS-Spoofing-Detection/sensor-data/raw/gps_path.html")
