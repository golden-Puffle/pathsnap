"""Report graphs for the 5-model comparison (all values read from results/*.csv).

Usage:
    python -m training.plot_model_graphs

Writes to results/graphs/:
  1_cv_macro_f1.png           CV macro F1 per model (mean of 5 trip-grouped folds, +/- 1 std)
  2_student_test_performance.png   student-test Top-1 accuracy and macro F1
  3_log_loss.png              CV and student-test log loss (lower is better)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {"lgbm": "LightGBM (selected)", "rf": "Random Forest", "mlp": "MLP",
         "knn": "KNN", "lr": "Logistic Regression"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e5e4e0"
OUT = Path("results/graphs")


def style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def grouped(ax, labels, a, b, a_name, b_name, fmt):
    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y - h / 2 - 0.02, a, height=h, color=BLUE, label=a_name)
    ax.barh(y + h / 2 + 0.02, b, height=h, color=ORANGE, label=b_name)
    for yi, v in zip(y - h / 2 - 0.02, a):
        ax.text(v, yi, f" {v:{fmt}}", va="center", fontsize=9, color=INK)
    for yi, v in zip(y + h / 2 + 0.02, b):
        ax.text(v, yi, f" {v:{fmt}}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv("results/model_comparison.csv")
    folds = pd.read_csv("results/cv_folds.csv")
    stats = folds.groupby("model")["macro_f1"].agg(["mean", "std"])
    comp = comp.sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
    labels = [NAMES.get(m, m) for m in comp["model"]]

    # Graph 1: CV macro F1 with fold spread
    fig, ax = plt.subplots(figsize=(9, 4.2))
    mean = stats.loc[comp["model"], "mean"].to_numpy()
    std = stats.loc[comp["model"], "std"].to_numpy()
    y = np.arange(len(labels))
    ax.barh(y, mean, height=0.55, color=BLUE, xerr=std,
            error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 4})
    for yi, m, s in zip(y, mean, std):
        ax.text(m + s + 0.005, yi, f"{m:.4f} ± {s:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    style(ax, "Graph 1: Cross-validation macro F1 (5 folds grouped by trip, mean ± 1 std)",
          "CV macro F1 (higher is better)")
    fig.tight_layout()
    fig.savefig(OUT / "1_cv_macro_f1.png", dpi=160)
    plt.close(fig)

    # Graph 2: student-test Top-1 accuracy and macro F1
    fig, ax = plt.subplots(figsize=(9, 5.2))
    grouped(ax, labels, comp["test_top1"], comp["test_macro_f1"],
            "Top-1 accuracy (per GPS point)", "Macro F1 (per candidate row)", ".4f")
    ax.set_xlim(0, 1.1)
    style(ax, "Graph 2: Student-test performance (10 unseen trips, classifier only)",
          "Score (higher is better)")
    fig.tight_layout()
    fig.savefig(OUT / "2_student_test_performance.png", dpi=160)
    plt.close(fig)

    # Graph 3: log loss, CV and student test
    fig, ax = plt.subplots(figsize=(9, 5.2))
    grouped(ax, labels, comp["cv_log_loss"], comp["test_log_loss"],
            "Cross-validation log loss", "Student-test log loss", ".4f")
    ax.set_xlim(0, max(comp["cv_log_loss"].max(), comp["test_log_loss"].max()) * 1.2)
    style(ax, "Graph 3: Log loss comparison (probability quality)", "Log loss (lower is better)")
    fig.tight_layout()
    fig.savefig(OUT / "3_log_loss.png", dpi=160)
    plt.close(fig)
    print("written:", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
