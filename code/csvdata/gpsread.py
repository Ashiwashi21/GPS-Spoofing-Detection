import pandas as pd
import folium
from pathlib import Path

# Load the synthesized GPS data
df = pd.read_csv("/home/pi/GPS-Spoofing-Detection/sensor-data/normal/normal.csv")

# Clean GPS data
df = df.dropna(subset=["latitude", "longitude"])
df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

if df.empty:
    raise SystemExit("No valid GPS data found after cleaning.")

# Create map centered around the mean coordinates
start_coords = [df["latitude"].mean(), df["longitude"].mean()]
m = folium.Map(location=start_coords, zoom_start=15)

# Draw the path with popups for altitude and speed
for i, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=f"Altitude: {row['altitude']} m<br>Speed: {row['speed_knots']} knots"
    ).add_to(m)

# Draw polyline path
folium.PolyLine(
    df[["latitude", "longitude"]].values,
    color="blue",
    weight=2.5,
    opacity=0.8
).add_to(m)

# Mark start and end points
folium.Marker(
    location=[df["latitude"].iloc[0], df["longitude"].iloc[0]],
    popup="Start",
    icon=folium.Icon(color="green")
).add_to(m)

folium.Marker(
    location=[df["latitude"].iloc[-1], df["longitude"].iloc[-1]],
    popup="End",
    icon=folium.Icon(color="red")
).add_to(m)

# Save output
out_path = Path("/home/pi/GPS-Spoofing-Detection/sensor-data/visuals/gps_path.html")
out_path.parent.mkdir(parents=True, exist_ok=True)
m.save(out_path)

print(f"? Saved GPS map to: {out_path}")
