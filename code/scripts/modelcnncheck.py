import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load data
normal_df = pd.read_csv("sensor-data/normal/normal.csv")
spoofed_df = pd.read_csv("sensor-data/spoofed/spoofed.csv")

# Select features
features = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
X_normal = normal_df[features].values
X_spoofed = spoofed_df[features].values

# Stack and label
X = np.vstack([X_normal, X_spoofed])
y = np.array([0] * len(X_normal) + [1] * len(X_spoofed))

# Downsample for speed
sample_size = 3000
idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[idx]
y_sample = y[idx]

# Normalize
X_scaled = StandardScaler().fit_transform(X_sample)

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_embedded[y_sample == 0, 0], X_embedded[y_sample == 0, 1], alpha=0.5, label="Normal", s=10)
plt.scatter(X_embedded[y_sample == 1, 0], X_embedded[y_sample == 1, 1], alpha=0.5, label="Spoofed", s=10)
plt.title("t-SNE Embedding of Normal vs Spoofed Samples (Downsampled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import numpy as np

import numpy as np
from scipy.stats import pearsonr

X_train = np.load("sensor-data/processed/X_train.npy")
y_train = np.load("sensor-data/processed/y_train.npy")

corrs = []
for i in range(X_train.shape[2]):
    feature_flat = X_train[:, :, i].flatten()
    y_repeated = np.repeat(y_train, X_train.shape[1])
    r, _ = pearsonr(feature_flat, y_repeated)
    corrs.append(abs(r))

print("Top correlated features:", np.argsort(corrs)[-5:][::-1])
print("Correlation values:", np.sort(corrs)[-5:][::-1])
