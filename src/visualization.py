"""Visualization utilities for training metrics."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_online_training_metrics(
    ht_metrics_path: str = "models/ht/hoeffding_online_metrics.csv",
    wf_metrics_path: str = "models/wf/weighted_forest_online_metrics.csv",
    figsize: tuple[int, int] = (14, 10),
) -> Figure:
    """Plot online training metrics for HT and WF."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ht_exists = os.path.exists(ht_metrics_path)
    wf_exists = os.path.exists(wf_metrics_path)

    if ht_exists:
        ht = pd.read_csv(ht_metrics_path)
        axes[0, 0].plot(ht["episode"], ht["survival_seconds"], marker="o")
        axes[0, 0].set_title("Hoeffding Tree - Survival Time")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Seconds")

        axes[0, 1].plot(ht["episode"], ht["progressive_accuracy"], marker="o")
        axes[0, 1].set_title("Hoeffding Tree - Progressive Accuracy")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Accuracy")

    if wf_exists:
        wf = pd.read_csv(wf_metrics_path)
        axes[1, 0].plot(wf["episode"], wf["survival_seconds"], marker="o")
        axes[1, 0].set_title("Weighted Forest - Survival Time")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Seconds")

        axes[1, 1].plot(wf["episode"], wf["num_cells"], marker="o")
        axes[1, 1].set_title("Weighted Forest - Active Cells")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Cells")

    plt.tight_layout()
    plt.show()
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "survival_time",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Plot boxplot comparison of models for a given metric."""
    fig, ax = plt.subplots(figsize=figsize)

    models = results_df["model"].unique()
    data = [results_df[results_df["model"] == m][metric].values for m in models]

    ax.boxplot(data, labels=models, patch_artist=True)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return fig
