import pandas as pd
from pathlib import Path
import folium

# --- Setup ---
csv = Path("sensor-data/spoofed/spoofed.csv")
out = Path("sensor-data/visuals")
out.mkdir(parents=True, exist_ok=True)

# --- Load and Clean ---
df = pd.read_csv(csv)
df_clean = df.dropna(subset=["latitude", "longitude"])
df_clean = df_clean[(df_clean["latitude"] != 0) & (df_clean["longitude"] != 0)]

print(df_clean[["latitude", "longitude"]].head(10))
print("Before:", len(df))
print("After:", len(df_clean))

if df_clean.empty:
    raise SystemExit("No valid lat/lon rows found in CSV")

# --- Map Setup ---
start = [df_clean["latitude"].iloc[0], df_clean["longitude"].iloc[0]]
m = folium.Map(location=start, zoom_start=16)

# --- Draw Path ---
coords = df_clean[["latitude", "longitude"]].values.tolist()
folium.PolyLine(coords, color="blue", weight=2, opacity=0.6).add_to(m)

# --- Optional Markers (if label column exists) ---
for _, r in df_clean.iterrows():
    color = "red" if int(r.get("label", 0)) == 1 else "green"
    folium.CircleMarker(
        location=(r["latitude"], r["longitude"]),
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=f"time:{r.get('timestamp','')}, label:{r.get('label',0)}"
    ).add_to(m)

# --- Save Output ---
out_file = out / "gps_path_spoofed.html"
m.save(out_file)
print("? Map saved to", out_file)

