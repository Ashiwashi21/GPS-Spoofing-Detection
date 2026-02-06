#goal of this script: create and train a hybrid on sensor windows, apply noise, and save model with metrics
#imports
import numpy as np
import os
import csv
import io
from contextlib import redirect_stdout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
from sklearn.preprocessing import StandardScaler
from ekf import ExtendedKalmanFilter
import tensorflow.keras.backend as K

#load training data
X_train = np.load("sensor-data/processed/X_train.npy")
y_train = np.load("sensor-data/processed/y_train.npy")

#pad feature dimensions
pad_width = 18 - X_train.shape[2]
if pad_width > 0:
    padding = np.zeros((X_train.shape[0], X_train.shape[1], pad_width))
    X_train = np.concatenate([X_train, padding], axis=2)

#add label noise (5%)
noise_ratio = 0.05
num_noisy = int(len(y_train) * noise_ratio)
noisy_indices = np.random.choice(len(y_train), num_noisy, replace=False)
y_train[noisy_indices] = 1 - y_train[noisy_indices]

#normalize features by applying standardscaler
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, 18)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

#add gaussian nouse, data augmentation
noise = np.random.normal(0, 0.05, size=X_train_scaled.shape)
scale = np.random.uniform(0.97, 1.03, size=(X_train_scaled.shape[0], 1, X_train_scaled.shape[2]))
mask = np.random.binomial(1, 0.99, size=X_train_scaled.shape)
X_train_scaled = X_train_scaled * scale * mask + noise

#ekf setup, define matrices
F = np.eye(18)
H = np.eye(18)
Q = np.eye(18) * 0.01
R = np.eye(18) * 0.1
x0 = np.zeros(18)
P0 = np.eye(18)

#apply ekf to traning windows
ekf = ExtendedKalmanFilter(F, H, Q, R, x0, P0)

def apply_ekf_batch(X):
    return np.array([ekf.run(seq) for seq in X])

X_train_ekf = apply_ekf_batch(X_train_scaled)

#fuse cnn and ekf
X_train_combined = np.concatenate([X_train_scaled, X_train_ekf], axis=-1)

#define focal loss function, focus more on hard samples
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        import tensorflow as tf
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return loss

#build hybrid cnn with two conv1d layers, batch, pooling, dropout, final output
model = Sequential([
    Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.0005), input_shape=(10, X_train_combined.shape[2])),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.0005)),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])

#compile model
model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', Precision(), Recall()])

#early stopping if accuracy plateaus 
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#train model for 10 epoches at 20% validation split
history = model.fit(
    X_train_combined, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

#creates model folder, saves model as hybrid
os.makedirs("models", exist_ok=True)
model.save("models/hybrid.keras")
print("? EKF-enhanced CNN model saved to models/hybrid.keras")

#prints final accuracy
final_acc = history.history["accuracy"][-1]
print(f"? Final training accuracy: {round(final_acc * 100, 2)}%")

#saves training logs
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

#saves model architecture
with open("results/hybrid_architecture.csv", "w", newline="") as f:
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        summary_text = buf.getvalue()
        for line in summary_text.strip().split("\n"):
            f.write(line + "\n")

#sanity check
print("? Saved architecture and training log to results/")
