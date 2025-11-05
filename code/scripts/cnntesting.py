import numpy as np
import time
import glob
import csv
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from gpiozero import LED

model = load_model("models/cnn.keras")
led = LED(17)

X_files = sorted(glob.glob("sensor-data/testing/x_*.npy"))
y_files = sorted(glob.glob("sensor-data/testing/y_*.npy"))

y_true = []
y_pred = []
detection_times = []

def pad_features(x, target_dim=18):
    pad_width = target_dim - x.shape[1]
    if pad_width > 0:
        padding = np.zeros((x.shape[0], pad_width))
        x = np.concatenate([x, padding], axis=1)
    return x

def flash_led():
    for _ in range(5):
        led.on()
        time.sleep(0.1)
        led.off()
        time.sleep(0.1)

with open("sensor-data/testing/test_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_number", "true_label", "predicted_label", "detection_time_seconds", "result"])

    for i, (x_path, y_path) in enumerate(zip(X_files, y_files)):
        x = np.load(x_path)
        x = pad_features(x)
        y = np.load(y_path)[0]
        start = time.time()
        pred = model.predict(np.expand_dims(x, axis=0))[0][0]
        end = time.time()
        label = int(pred > 0.5)
        y_true.append(y)
        y_pred.append(label)
        detection_times.append(end - start)

        print(f"\nTesting case {i+1}:")
        if label == 0:
            print("normal")
        else:
            print("spoofed")
            flash_led()

        print(f"Prediction: {label}, Actual: {y}")
        print(f"Detection time: {end - start:.4f} seconds")

        result = "normal" if label == 0 else "spoofed"
        writer.writerow([i+1, y, label, round(end - start, 4), result])

f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
std_dev = np.std(detection_times)

print("\n--- Final Metrics ---")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"Detection Time Std Dev: {std_dev:.4f} seconds")
