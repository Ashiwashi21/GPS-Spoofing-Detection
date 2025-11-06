import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model("models/cnn.keras")
print("\n? Model loaded successfully.")

# Show model summary
print("\n?? Model Summary:")
model.summary()

# Check input shape
expected_input_shape = model.input_shape
print(f"\n?? Expected input shape: {expected_input_shape}")

# Load training sample to test prediction
X_train = np.load("sensor-data/processed/X_train.npy")
y_train = np.load("sensor-data/processed/y_train.npy")

# Pad to 18 features if needed
pad_width = 18 - X_train.shape[2]
if pad_width > 0:
    padding = np.zeros((X_train.shape[0], X_train.shape[1], pad_width))
    X_train = np.concatenate([X_train, padding], axis=2)

# Predict on first 10 samples
print("\n?? Sample Predictions:")
for i in range(10):
    sample = X_train[i].reshape(1, X_train.shape[1], X_train.shape[2])
    score = model.predict(sample, verbose=0)[0][0]
    label = int(score > 0.5)
    print(f"Sample {i+1}: True = {y_train[i]}, Predicted = {label}, Score = {score:.4f}")

# Visualize weights of first Conv1D layer
print("\n?? Visualizing weights of first Conv1D layer...")
weights = model.layers[0].get_weights()[0]  # shape: (kernel_size, input_dim, filters)
plt.figure(figsize=(10, 4))
plt.imshow(weights[:, :, 0], aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Conv1D Layer 1 - Filter 0 Weights")
plt.xlabel("Input Feature")
plt.ylabel("Kernel Position")
plt.tight_layout()
plt.show()
