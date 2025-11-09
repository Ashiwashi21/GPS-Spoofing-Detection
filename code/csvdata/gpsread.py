import pandas as pd
import folium
from pathlib import Path

# --- Load and Clean GPS Data ---
df = pd.read_csv("/home/pi/GPS-Spoofing-Detection/sensor-data/normal/normal.csv")
df = df.dropna(subset=["latitude", "longitude"])
df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

if df.empty:
    raise SystemExit("No valid GPS data found after cleaning.")

# --- Smooth the Path ---
df["latitude_smooth"] = df["latitude"].rolling(window=5, center=True).mean()
df["longitude_smooth"] = df["longitude"].rolling(window=5, center=True).mean()
df_smooth = df.dropna(subset=["latitude_smooth", "longitude_smooth"])

# --- Create Map ---
start_coords = [df_smooth["latitude_smooth"].mean(), df_smooth["longitude_smooth"].mean()]
m = folium.Map(location=start_coords, zoom_start=15)

# --- Draw Smoothed Polyline ---
folium.PolyLine(
    df_smooth[["latitude_smooth", "longitude_smooth"]].values,
    color="blue",
    weight=2.5,
    opacity=0.8
).add_to(m)

# --- Mark Start and End Points ---
folium.Marker(
    location=[df_smooth["latitude_smooth"].iloc[0], df_smooth["longitude_smooth"].iloc[0]],
    popup="Start",
    icon=folium.Icon(color="green")
).add_to(m)

folium.Marker(
    location=[df_smooth["latitude_smooth"].iloc[-1], df_smooth["longitude_smooth"].iloc[-1]],
    popup="End",
    icon=folium.Icon(color="red")
).add_to(m)

# --- Save Output ---
out_path = Path("/home/pi/GPS-Spoofing-Detection/sensor-data/visuals/gps_path.html")
out_path.parent.mkdir(parents=True, exist_ok=True)
m.save(out_path)

print(f"? Smoothed GPS path saved to: {out_path}")

