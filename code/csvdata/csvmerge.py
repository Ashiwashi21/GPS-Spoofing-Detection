import pandas as pd
from pathlib import Path

base = Path(__file__).parent.parent.parent / "sensor-data"
gps_path = base / "raw" / "gps.csv"
imu_path = base / "raw" / "imu.csv"
fused_path = base / "processed" / "fused.csv"

gps = pd.read_csv(gps_path)
imu = pd.read_csv(imu_path)

gps["time"] = gps["timestamp"] - gps["timestamp"].iloc[0]
imu["time"] = imu["timestamp"] - imu["timestamp"].iloc[0]


merged = pd.merge_asof(
    imu.sort_values("time"),
    gps.sort_values("time"),
    on="time",
    direction="nearest",
    tolerance=1  
)

fused_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(fused_path, index=False)

print(f"Fused data saved to: {fused_path}")
print(merged.head())
