import pandas as pd
from pathlib import Path
import folium
from folium.plugins import HeatMap, TimestampedGeoJson

csv = Path("sensor-data/spoofed/spoofed.csv")
out = Path("sensor-data/visuals")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv)
import re

def clean_coord(x):
    if pd.isna(x):
        return None
    
    x = str(x).strip().upper()
    x = re.sub(r"[^0-9\.\-]", "", x)
    try:
        return float(x)
    except ValueError:
        return None

df["lat_decimal"] = df["lat_decimal"].apply(clean_coord)
df["lon_decimal"] = df["lon_decimal"].apply(clean_coord)

df_clean = df.dropna(subset=["lat_decimal", "lon_decimal"]).copy()


df_clean["lat_decimal"] = pd.to_numeric(df_clean["lat_decimal"], errors="coerce")
df_clean["lon_decimal"] = pd.to_numeric(df_clean["lon_decimal"], errors="coerce")
df_clean = df_clean.dropna(subset=["lat_decimal", "lon_decimal"])
print(df["lat_decimal"].head(10))
print("Before:", len(df))
print("After:", len(df_clean))



if df_clean.empty:
    raise SystemExit("No valid lat/lon rows found in CSV")

start = [df_clean["lat_decimal"].iloc[0], df_clean["lon_decimal"].iloc[0]]
m = folium.Map(location=start, zoom_start=16)

coords = df_clean[["lat_decimal", "lon_decimal"]].values.tolist()
folium.PolyLine(coords, color="blue", weight=2, opacity=0.6).add_to(m)

for _, r in df_clean.iterrows():
    color = "red" if int(r.get("label", 0)) == 1 else "green"
    folium.CircleMarker(
        location=(r["lat_decimal"], r["lon_decimal"]),
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=f"time:{r.get('timestamp','')}, label:{r.get('label',0)}"
    ).add_to(m)


out_file = out / "gps_path_spoofed.html"
m.save(out_file)
print("Map saved to", out_file)