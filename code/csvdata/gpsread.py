import pandas as pd
import folium
import numpy as np

def nmea_to_decimal(coord, direction):
    try:
        if pd.isna(coord) or coord == "" or str(coord).lower() == "nan":
            return np.nan
        
        coord = float(coord)
        deg = int(coord / 100)
        minutes = coord - deg * 100
        decimal = deg + minutes / 60

        if direction in ["S", "W"]:
            decimal = -decimal
        return decimal

    except (ValueError, TypeError):
        return np.nan

df = pd.read_csv("/home/pi/GPS-Spoofing-Detection/sensor-data/normal/normal.csv")

if "latitude" not in df.columns or "longitude" not in df.columns:
    raise KeyError("CSV must contain 'latitude' and 'longitude' columns")


df[["lat_val", "lat_dir"]] = df["latitude"].astype(str).str.split(" ", expand=True)
df[["lon_val", "lon_dir"]] = df["longitude"].astype(str).str.split(" ", expand=True)


df["lat_decimal"] = df.apply(lambda x: nmea_to_decimal(x["lat_val"], x["lat_dir"]), axis=1)
df["lon_decimal"] = df.apply(lambda x: nmea_to_decimal(x["lon_val"], x["lon_dir"]), axis=1)

df = df.dropna(subset=["lat_decimal", "lon_decimal"]).reset_index(drop=True)

if df.empty:
    raise SystemExit("No valid GPS data found after cleaning.")

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

out_path = "/home/pi/GPS-Spoofing-Detection/sensor-data/raw/gps_path.html"
m.save(out_path)
print(f"Saved map to: {out_path}")
