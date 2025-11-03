import pandas as pd

df = pd.read_csv("sensor-data/normal/normal.csv")
print(df['speed_knots'].describe())
# assuming walking is below some threshold, like 5 knots
walking = df[df['speed_knots'] <= 5]
driving = df[df['speed_knots'] > 5]

print(len(walking), len(driving))
