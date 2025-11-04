import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GaussianNoise
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load data
# -------------------------------
X_train = np.load("sensor-data/processed/X_train.npy")
X_test = np.load("sensor-data/processed/X_test.npy")
y_train = np.load("sensor-data/processed/y_train.npy")
y_test = np.load("sensor-data/processed/y_test.npy")

# -------------------------------
# Scale features
# -------------------------------
n_samples, timesteps, n_features = X_train.shape
scaler = StandardScaler()

X_train_flat = X_train.reshape(-1, n_features)
X_test_flat = X_test.reshape(-1, n_features)

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, timesteps, n_features)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], timesteps, n_features)

# -------------------------------
# Add Gaussian noise to inputs
# -------------------------------
def add_noise(X, std=0.1):
    noise = np.random.normal(0, std, X.shape)
    return X + noise

X_train_noisy = add_noise(X_train_scaled)
X_test_noisy = add_noise(X_test_scaled)

# -------------------------------
# Build CNN model
# -------------------------------
model = Sequential([
    GaussianNoise(0.05, input_shape=(timesteps, n_features)),
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
model.summary()

# -------------------------------
# Train
# -------------------------------
history = model.fit(
    X_train_noisy, y_train,
    validation_data=(X_test_noisy, y_test),
    epochs=25,
    batch_size=16,
    verbose=1
)

# -------------------------------
# Evaluate
# -------------------------------
loss, acc = model.evaluate(X_test_noisy, y_test, verbose=0)
print(f"Test Accuracy (no EKF, noisy): {acc*100:.2f}%")
