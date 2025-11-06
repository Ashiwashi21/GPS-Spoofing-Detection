import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

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
# Build CNN model
# -------------------------------
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(10, 18)),
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
    X_train_scaled, y_train,
    epochs=10,                # reduced from 25
    batch_size=32,            # increased from 8
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# Save model
# -------------------------------
model.save("models/cnn.keras")
print("? CNN model saved to models/cnn.keras")
import csv
import os

# -------------------------------
# Save training history to table
# -------------------------------
os.makedirs("results", exist_ok=True)
with open("results/cnn_training_log.csv", "w", newline="") as f:
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
os.makedirs("results", exist_ok=True)
with open("results/cnn_architecture.csv", "w", newline="") as f:
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
