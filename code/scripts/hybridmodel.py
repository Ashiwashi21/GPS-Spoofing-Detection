import numpy as np
import os
import csv
import io
from contextlib import redirect_stdout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from ekf import ExtendedKalmanFilter

# -------------------------------
# Load training data
# -------------------------------
X_train = np.load("sensor-data/processed/X_train.npy")
y_train = np.load("sensor-data/processed/y_train.npy")

# Pad to 18 features if needed
pad_width = 18 - X_train.shape[2]
if pad_width > 0:
    padding = np.zeros((X_train.shape[0], X_train.shape[1], pad_width))
    X_train = np.concatenate([X_train, padding], axis=2)

# -------------------------------
# Scale features
# -------------------------------
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, 18)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

# -------------------------------
# EKF Setup
# -------------------------------
F = np.eye(18)
H = np.eye(18)
Q = np.eye(18) * 0.01
R = np.eye(18) * 0.1
x0 = np.zeros(18)
P0 = np.eye(18)

ekf = ExtendedKalmanFilter(F, H, Q, R, x0, P0)

def apply_ekf_batch(X):
    return np.array([ekf.run(seq) for seq in X])

X_train_ekf = apply_ekf_batch(X_train_scaled)

# -------------------------------
# Combine EKF features
# -------------------------------
X_train_combined = np.concatenate([X_train_scaled, X_train_ekf], axis=-1)

# -------------------------------
# Build CNN model (same as original)
# -------------------------------
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(10, X_train_combined.shape[2])),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------
# Train model
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(
    X_train_combined, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# Save model
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/hybrid.keras")
print("? EKF-enhanced CNN model saved to models/cnn_ekf.keras")

# -------------------------------
# Print final training accuracy
# -------------------------------
final_acc = history.history["accuracy"][-1]
print(f"? Final training accuracy: {round(final_acc * 100, 2)}%")

# -------------------------------
# Save training history
# -------------------------------
os.makedirs("results", exist_ok=True)
with open("results/hybrid_training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Accuracy", "Train Loss", "Val Accuracy", "Val Loss"])
    for i in range(len(history.history["accuracy"])):
        writer.writerow([
            i + 1,
            round(history.history["accuracy"][i], 4),
            round(history.history["loss"][i], 4),
            round(history.history["val_accuracy"][i], 4),
            round(history.history["val_loss"][i], 4)
        ])

# -------------------------------
# Save model architecture summary
# -------------------------------
with open("results/hybrid_architecture.csv", "w", newline="") as f:
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        summary_text = buf.getvalue()
        for line in summary_text.strip().split("\n"):
            f.write(line + "\n")

print("? Saved architecture and training log to results/")
