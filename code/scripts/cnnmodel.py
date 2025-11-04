import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler

# Load data
X_train = np.load("sensor-data/processed/X_train.npy")
X_test = np.load("sensor-data/processed/X_test.npy")
y_train = np.load("sensor-data/processed/y_train.npy")
y_test = np.load("sensor-data/processed/y_test.npy")

# Scale features (important for CNN)
n_samples, timesteps, n_features = X_train.shape
scaler = StandardScaler()

# Flatten each segment to scale, then reshape back
X_train_flat = X_train.reshape(-1, n_features)
X_test_flat = X_test.reshape(-1, n_features)

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, timesteps, n_features)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], timesteps, n_features)

# Build CNN
model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(timesteps, n_features)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),  # 10 -> 5 timesteps
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),  # 5 -> 2 timesteps
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=25,
    batch_size=16,
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
