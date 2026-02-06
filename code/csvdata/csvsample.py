import pandas as pd
import os

os.makedirs("sensor-data/samples", exist_ok=True)

normal_df = pd.read_csv("sensor-data/normal/normal.csv")
normal_df.sample(500).to_csv("sensor-data/samples/normal_sample.csv", index=False)

spoofed_df = pd.read_csv("sensor-data/spoofed/spoofed.csv")
spoofed_df.sample(500).to_csv("sensor-data/samples/spoofed_sample.csv", index=False)

merged_df = pd.read_csv("sensor-data/processed/merged.csv")
merged_df.sample(500).to_csv("sensor-data/samples/merged_sample.csv", index=False)

print("Samples created successfully.")