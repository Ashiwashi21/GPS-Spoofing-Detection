import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from ekf import ExtendedKalmanFilter

# --- config ---
PAD_TO = 18
NUM_FOLDS = 5
RANDOM_STATE = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CORRUPT_PARAMS = {
    "noise_std": 0.02,
    "drop_prob": 0.002,
    "drift_scale": 0.002,
    "scale_var": 0.003
}

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return loss

model = load_model("models/hybrid.keras", custom_objects={"loss": focal_loss()})

# --- EKF setup ---
F = np.eye(PAD_TO)
H = np.eye(PAD_TO)
Q = np.eye(PAD_TO) * 0.01
R = np.eye(PAD_TO) * 0.1
x0 = np.zeros(PAD_TO)
P0 = np.eye(PAD_TO)
ekf = ExtendedKalmanFilter(F, H, Q, R, x0, P0)

def apply_ekf_batch(X):
    return np.array([ekf.run(seq) for seq in X])

# --- load training data ---
X_train = np.load("sensor-data/processed/X_train.npy")
y_train = np.load("sensor-data/processed/y_train.npy")
pad_width = PAD_TO - X_train.shape[2]
if pad_width > 0:
    X_train = np.concatenate([X_train, np.zeros((X_train.shape[0], X_train.shape[1], pad_width))], axis=2)

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, PAD_TO))

def signature(x):
    return tuple(np.round(x.flatten(), 6))
train_sigs = {signature(x) for x in X_train}

# --- load test sets ---
tests, labels = [], []
for i in range(1, 6):
    xt = np.load(f"sensor-data/processed/X_test_{i}.npy")
    yt = np.load(f"sensor-data/processed/y_test_{i}.npy")
    pad = PAD_TO - xt.shape[2]
    if pad > 0:
        xt = np.concatenate([xt, np.zeros((xt.shape[0], xt.shape[1], pad))], axis=2)
    tests.append(xt)
    labels.append(yt)

X_all = np.concatenate(tests, axis=0)
y_all = np.concatenate(labels, axis=0)

# remove duplicates
keep_mask = np.array([signature(x) not in train_sigs for x in X_all])
if not keep_mask.all():
    removed = np.sum(~keep_mask)
    X_all = X_all[keep_mask]
    y_all = y_all[keep_mask]
else:
    removed = 0

rng = np.random.RandomState(RANDOM_STATE)

def corrupt(X, params):
    Xc = X.astype(float).copy()
    n, T, F = Xc.shape
    Xc += rng.normal(0, params["noise_std"], size=Xc.shape)
    mask = rng.rand(*Xc.shape) < params["drop_prob"]
    Xc[mask] = 0.0
    ramps = (np.linspace(0, 1, T) - 0.5)[None, :, None]
    coeffs = rng.normal(0, params["drift_scale"], size=(n, 1, 1))
    Xc += ramps * coeffs
    scales = 1.0 + rng.uniform(-params["scale_var"], params["scale_var"], size=(n, 1, F))
    Xc *= scales
    return Xc

# --- evaluation ---
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_idx = 0
agg_cm = np.zeros((2, 2), int)
accs, f1s = [], []
metrics_log = []

for _, test_index in skf.split(X_all, y_all):
    fold_idx += 1
    X_fold, y_fold = X_all[test_index], y_all[test_index]
    if np.unique(y_fold).size < 2:
        continue

    Xc = corrupt(X_fold, CORRUPT_PARAMS)
    X_scaled = scaler.transform(Xc.reshape(-1, PAD_TO)).reshape(Xc.shape)
    residuals = apply_ekf_batch(X_scaled)
    X_combined = np.concatenate([X_scaled, residuals], axis=-1)

    y_scores = model.predict(X_combined, verbose=0).flatten()
    y_pred = (y_scores > 0.5).astype(int)

    acc = accuracy_score(y_fold, y_pred)
    f1 = f1_score(y_fold, y_pred, zero_division=0)
    cm = confusion_matrix(y_fold, y_pred, labels=[0, 1])

    metrics_log.append([fold_idx, acc, f1])
    agg_cm += cm
    accs.append(acc)
    f1s.append(f1)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"Hybrid Confusion - Fold {fold_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"hybrid_fold_{fold_idx}_cm.png"))
    plt.close()

    flat_Xc = X_combined.reshape(X_combined.shape[0], -1)
    df_corr = pd.DataFrame(flat_Xc)
    df_corr['label'] = y_fold
    df_corr.to_csv(os.path.join(RESULTS_DIR, f"hybrid_fold_{fold_idx}.csv"), index=False)

# --- aggregate results ---
if accs:
    plt.figure(figsize=(6, 4))
    sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Hybrid Aggregate Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hybrid_cm_aggregate.png"))
    plt.close()

    df_metrics = pd.DataFrame(metrics_log, columns=['fold', 'accuracy', 'f1'])
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "hybrid_fold_metrics.csv"), index=False)

# --- conv1d weight distribution ---
layer = model.get_layer(index=0)
weights = layer.get_weights()[0].flatten()

plt.figure(figsize=(6, 4))
plt.hist(weights, bins=50, color='purple', edgecolor='black')
plt.title("Hybrid Conv1D Weight Distribution")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "hybrid_conv1d_weight_distribution.png"))
plt.close()

# --- mean abs weight per kernel ---
mean_abs = np.mean(np.abs(layer.get_weights()[0]), axis=(0, 1))
plt.figure(figsize=(8, 4))
plt.bar(range(len(mean_abs)), mean_abs)
plt.title("Hybrid Conv1D Kernels - Mean Abs Weight")
plt.xlabel("Kernel Index")
plt.ylabel("Mean Abs Weight")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "hybrid_conv1d_weights_mean_abs.png"))
plt.close()

# --- training curves ---
df_log = pd.read_csv("results/hybrid_training_log.csv")

plt.figure(figsize=(6, 4))
plt.plot(df_log['Epoch'], df_log['Train Accuracy'], label='Train Accuracy')
plt.plot(df_log['Epoch'], df_log['Val Accuracy'], label='Val Accuracy')
plt.title("Hybrid Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "hybrid_train_val_accuracy.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(df_log['Epoch'], df_log['Train Loss'], label='Train Loss')
plt.plot(df_log['Epoch'], df_log['Val Loss'], label='Val Loss')
plt.title("Hybrid Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "hybrid_train_val_loss.png"))
plt.close()

# --- print summary statistics ---
print("\n=== Hybrid Evaluation Summary ===")
print(f"Removed duplicates: {removed}")
print(f"Average Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Average F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print("Aggregate Confusion Matrix:\n", agg_cm)
print(f"Results saved in '{RESULTS_DIR}' directory.")
