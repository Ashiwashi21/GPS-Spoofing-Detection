import numpy as np
import os
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GaussianNoise
from sklearn.preprocessing import StandardScaler
from ekf import ExtendedKalmanFilter

# -------------------------------
# Load and scale training data
# -------------------------------
X_train = np.load("sensor-data/processed/X_train.npy")
y_train = np.load("sensor-data/processed/y_train.npy")

n_samples, timesteps, n_features = X_train.shape
scaler = StandardScaler()

X_train_flat = X_train.reshape(-1, n_features)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, timesteps, n_features)

# -------------------------------
# EKF Setup
# -------------------------------
F = np.eye(n_features)
H = np.eye(n_features)
Q = np.eye(n_features) * 0.01
R = np.eye(n_features) * 0.1
x0 = np.zeros(n_features)
P0 = np.eye(n_features)

ekf = ExtendedKalmanFilter(F, H, Q, R, x0, P0)

def apply_ekf_batch(X):
    return np.array([ekf.run(seq) for seq in X])

X_train_ekf = apply_ekf_batch(X_train_scaled)

# -------------------------------
# Combine features (no noise)
# -------------------------------
X_train_combined = np.concatenate([X_train_scaled, X_train_ekf], axis=-1)

# -------------------------------
# Build Hybrid CNN
# -------------------------------
model = Sequential([
    GaussianNoise(0.05, input_shape=(timesteps, X_train_combined.shape[2])),
    Conv1D(filters=16, kernel_size=2, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),
    Conv1D(filters=32, kernel_size=2, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------
# Train
# -------------------------------
history = model.fit(
    X_train_combined, y_train,
    epochs=25,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# -------------------------------
# Save model
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/hybrid.keras")
print("? Hybrid model saved to models/hybrid.keras")

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
# Save model architecture
# -------------------------------
with open("results/hybrid_architecture.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Layer Name", "Layer Type", "Output Shape", "Param #"])
    for layer in model.layers:
        name = layer.name
        layer_type = layer.__class__.__name__
        output_shape = str(layer.output_shape)
        param_count = layer.count_params()
        writer.writerow([name, layer_type, output_shape, param_count])

    writer.writerow([])
    writer.writerow(["Total Params", model.count_params()])
    writer.writerow(["Trainable Params", np.sum([np.prod(w.shape) for w in model.trainable_weights])])
    writer.writerow(["Non-trainable Params", np.sum([np.prod(w.shape) for w in model.non_trainable_weights])])
    writer.writerow(["Optimizer Params", np.sum([np.prod(w.shape) for w in model.optimizer.weights])])
