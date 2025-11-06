import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -------------------------------
# Load model and test data
# -------------------------------
model = load_model("models/cnn.keras")
X_test = np.load("sensor-data/processed/X_test.npy")
y_test = np.load("sensor-data/processed/y_test.npy")

# Pad to 18 features if needed
pad_width = 18 - X_test.shape[2]
if pad_width > 0:
    padding = np.zeros((X_test.shape[0], X_test.shape[1], pad_width))
    X_test = np.concatenate([X_test, padding], axis=2)

# -------------------------------
# Scale features using training data
# -------------------------------
X_train = np.load("sensor-data/processed/X_train.npy")
pad_width = 18 - X_train.shape[2]
if pad_width > 0:
    padding = np.zeros((X_train.shape[0], X_train.shape[1], pad_width))
    X_train = np.concatenate([X_train, padding], axis=2)

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, 18))
X_test_scaled = scaler.transform(X_test.reshape(-1, 18)).reshape(X_test.shape)

# -------------------------------
# Print model summary
# -------------------------------
print("\n?? CNN Model Summary:")
model.summary()

# -------------------------------
# Visualize first Conv1D kernel
# -------------------------------
print("\n?? Visualizing first Conv1D layer weights...")
weights = model.layers[0].get_weights()[0]  # shape: (kernel_size, input_dim, filters)
plt.figure(figsize=(10, 4))
plt.imshow(weights[:, :, 0], aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Conv1D Layer 1 - Filter 0 Weights")
plt.xlabel("Input Feature")
plt.ylabel("Kernel Position")
plt.tight_layout()
plt.show()

# -------------------------------
# Predict and evaluate
# -------------------------------
start = time.time()
y_scores = model.predict(X_test_scaled, verbose=0).flatten()
end = time.time()

y_pred = (y_scores > 0.5).astype(int)
detection_times = [(end - start) / len(y_test)] * len(y_test)

f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
std_dev = np.std(detection_times)

# -------------------------------
# Print metrics
# -------------------------------
print("\n? Evaluation Complete")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"Detection Time Std Dev: {std_dev:.4f} seconds")

# -------------------------------
# Save log to results folder
# -------------------------------
os.makedirs("results", exist_ok=True)
with open("results/cnn_test_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_number", "true_label", "predicted_label", "detection_time_seconds", "result"])
    for i, (yt, yp, dt) in enumerate(zip(y_test, y_pred, detection_times)):
        writer.writerow([i+1, yt, yp, round(dt, 4), "spoofed" if yp else "normal"])
    writer.writerow([])
    writer.writerow(["Summary"])
    writer.writerow(["F1 Score", round(f1, 4)])
    writer.writerow(["Accuracy", round(acc, 4)])
    writer.writerow(["Detection Time Std Dev", round(std_dev, 4)])
    writer.writerow(["Confusion Matrix"])
    writer.writerows(cm)
