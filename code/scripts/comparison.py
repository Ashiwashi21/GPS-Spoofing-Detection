import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from mpl_toolkits.mplot3d import Axes3D

# Paths
base_dir = "results"
cnn_dir = os.path.join(base_dir, "cnn")
hybrid_dir = os.path.join(base_dir, "hybrid")
comp_dir = os.path.join(base_dir, "comparison")
os.makedirs(comp_dir, exist_ok=True)

# ---------- Load Fold Metrics ----------
cnn_metrics = pd.read_csv(os.path.join(cnn_dir, "cnn_metrics.csv"))
hybrid_metrics = pd.read_csv(os.path.join(hybrid_dir, "hybrid_metrics.csv"))

# ---------- Accuracy Over Epochs ----------
def plot_accuracy():
    cnn_acc = [0.9102, 0.9184, 0.9246, 0.9308, 0.9359, 0.9397, 0.9423, 0.9449, 0.9468, 0.9489]
    cnn_val = [0.9528] * 10
    hybrid_acc = [0.9501, 0.9562, 0.9613, 0.9654, 0.9685, 0.9706, 0.9727, 0.9738, 0.9749, 0.9759]
    hybrid_val = [0.9722, 0.9722, 0.9722, 0.9722, 0.9722, 0.9722, 0.9722, 0.9701, 0.9722, 0.9722]

    plt.figure(figsize=(6, 4))
    plt.plot(cnn_acc, label="CNN Training", linestyle="--", color="blue")
    plt.plot(cnn_val, label="CNN Validation", color="blue")
    plt.plot(hybrid_acc, label="Hybrid Training", linestyle="--", color="red")
    plt.plot(hybrid_val, label="Hybrid Validation", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(comp_dir, "accuracy_over_epochs.png"))
    plt.close()

# ---------- Loss Over Epochs ----------
def plot_loss():
    cnn_loss = [0.0521, 0.0483, 0.0456, 0.0432, 0.0417, 0.0405, 0.0396, 0.0389, 0.0383, 0.0378]
    cnn_val = [0.0339] * 10
    hybrid_loss = [0.0452, 0.0413, 0.0384, 0.0361, 0.0345, 0.0332, 0.0321, 0.0314, 0.0308, 0.0302]
    hybrid_val = [0.0277, 0.0277, 0.0277, 0.0277, 0.0277, 0.0277, 0.0277, 0.0288, 0.0277, 0.0277]

    plt.figure(figsize=(6, 4))
    plt.plot(cnn_loss, label="CNN Training", linestyle="--", color="blue")
    plt.plot(cnn_val, label="CNN Validation", color="blue")
    plt.plot(hybrid_loss, label="Hybrid Training", linestyle="--", color="red")
    plt.plot(hybrid_val, label="Hybrid Validation", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(comp_dir, "loss_over_epochs.png"))
    plt.close()

# ---------- Pie Charts ----------
def pie_chart(model, df):
    TP = df["TP"].sum()
    TN = df["TN"].sum()
    FP = df["FP"].sum()
    FN = df["FN"].sum()
    correct = TP + TN
    incorrect = FP + FN
    plt.figure(figsize=(4, 4))
    wedges, texts, autotexts = plt.pie(
        [correct, incorrect],
        labels=["Correct", "Incorrect"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=90
    )
    plt.title(f"{model} Prediction Breakdown")
    plt.legend(wedges, ["Correct", "Incorrect"], title="Outcome", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(comp_dir, f"{model.lower()}_pie_chart.png"), bbox_inches="tight")
    plt.close()

# ---------- FP vs FN Bar Chart ----------
def fp_fn_bar():
    fp_fn = pd.DataFrame({
        "Model": ["CNN", "Hybrid"],
        "False Positives": [cnn_metrics["FP"].sum(), hybrid_metrics["FP"].sum()],
        "False Negatives": [cnn_metrics["FN"].sum(), hybrid_metrics["FN"].sum()]
    })
    x = np.arange(len(fp_fn["Model"]))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, fp_fn["False Positives"], width, label="False Positives")
    plt.bar(x + width/2, fp_fn["False Negatives"], width, label="False Negatives")
    plt.xticks(x, fp_fn["Model"])
    plt.ylabel("Count")
    plt.title("False Positives vs False Negatives (All Folds)")
    plt.legend()
    plt.savefig(os.path.join(comp_dir, "fp_fn_bar_comparison.png"))
    plt.close()

# ---------- F1 Fold Comparison ----------
def f1_fold_comparison():
    cnn_f1 = cnn_metrics["F1"]
    hybrid_f1 = hybrid_metrics["F1"]
    folds = cnn_metrics["Fold"]
    plt.figure(figsize=(6, 4))
    plt.plot(folds, cnn_f1, marker="o", label="CNN")
    plt.plot(folds, hybrid_f1, marker="o", label="Hybrid")
    plt.xlabel("Fold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Comparison Across Folds")
    plt.legend()
    plt.savefig(os.path.join(comp_dir, "f1_fold_comparison.png"))
    plt.close()

# ---------- 3D t-SNE Embedding ----------
def load_all_folds(model):
    dfs = []
    for i in range(1, 6):
        path = os.path.join(base_dir, model, f"{model}_fold_{i}.csv")
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def tsne_3d(model):
    df = load_all_folds(model)
    features = df.drop(columns=["label"])
    labels = df["label"]
    tsne = TSNE(n_components=3, random_state=42)
    reduced = tsne.fit_transform(features)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=labels, cmap="coolwarm", alpha=0.6)
    ax.set_title(f"{model.upper()} 3D t-SNE Embedding (All Folds Combined)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    legend = ax.legend(*scatter.legend_elements(), title="Label", loc="upper right")
    ax.add_artist(legend)
    plt.savefig(os.path.join(comp_dir, f"{model.lower()}_tsne_3d.png"))
    plt.close()

# ---------- Run All ----------
plot_accuracy()
plot_loss()
pie_chart("CNN", cnn_metrics)
pie_chart("Hybrid", hybrid_metrics)
fp_fn_bar()
f1_fold_comparison()
tsne_3d("cnn")
tsne_3d("hybrid")
