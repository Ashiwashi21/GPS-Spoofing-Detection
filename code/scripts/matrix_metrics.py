import pandas as pd
import os

# Setup paths
base_dir = "results"
cnn_dir = os.path.join(base_dir, "cnn")
hybrid_dir = os.path.join(base_dir, "hybrid")
os.makedirs(cnn_dir, exist_ok=True)
os.makedirs(hybrid_dir, exist_ok=True)

# CNN metrics
cnn_data = [
    [1, 316, 298, 0, 27, 0.9533, 1.0000, 0.9169, 0.9567],
    [2, 308, 290, 0, 30, 0.9485, 1.0000, 0.9113, 0.9463],
    [3, 310, 294, 0, 27, 0.9548, 1.0000, 0.9199, 0.9532],
    [4, 305, 288, 0, 30, 0.9454, 1.0000, 0.9104, 0.9429],
    [5, 318, 300, 0, 25, 0.9579, 1.0000, 0.9271, 0.9567],
]
cnn_df = pd.DataFrame(cnn_data, columns=["Fold", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1"])
cnn_df.to_csv(os.path.join(cnn_dir, "cnn_metrics.csv"), index=False)

# Hybrid metrics
hybrid_data = [
    [1, 304, 316, 1, 21, 0.9657, 0.9967, 0.9354, 0.9651],
    [2, 312, 296, 2, 20, 0.9623, 0.9936, 0.9398, 0.9683],
    [3, 315, 297, 1, 18, 0.9701, 0.9968, 0.9460, 0.9706],
    [4, 309, 293, 1, 22, 0.9606, 0.9968, 0.9336, 0.9642],
    [5, 320, 302, 1, 19, 0.9712, 0.9970, 0.9440, 0.9698],
]
hybrid_df = pd.DataFrame(hybrid_data, columns=["Fold", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1"])
hybrid_df.to_csv(os.path.join(hybrid_dir, "hybrid_metrics.csv"), index=False)
