import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def save_sorted_results(detailed_results, prefix="results"):
    """Save benchmark results sorted by recall and latency."""
    df = pd.DataFrame(detailed_results)

    # Sort by recall (ascending)
    recall_sorted = df.sort_values(by="avg_execution_recall", ascending=True)
    recall_sorted.to_csv(
        f"{prefix}_sorted_by_recall.csv",
        columns=["name", "avg_execution_recall", "avg_execution_time"],
        index=False
    )

    # Sort by latency (descending)
    latency_sorted = df.sort_values(by="avg_execution_time", ascending=False)
    latency_sorted.to_csv(
        f"{prefix}_sorted_by_latency.csv",
        columns=["name", "avg_execution_time", "avg_execution_recall"],
        index=False
    )

    print(f"Saved sorted results to {prefix}_sorted_by_recall.csv and {prefix}_sorted_by_latency.csv")


def plot_distribution(detailed_results, prefix, figsize=(12, 5)):
    """Plot the distribution of recall and latency for benchmark results."""
    df = pd.DataFrame(detailed_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Query Performance Distribution", fontsize=16, fontweight="bold")
    
    # Academic-style color palette
    recall_color = "#1f77b4"   # main blue
    latency_color = "#4287f5"  # lighter blue
    
    # 1. Recall distribution
    recall_data = df["avg_execution_recall"]
    recall_bins = np.linspace(recall_data.min(), recall_data.max(), 15)
    ax1.hist(
        recall_data,
        bins=recall_bins,
        color=recall_color,
        alpha=0.7,
        edgecolor="black"
    )
    
    ax1.set_title("Recall Distribution", fontsize=14, fontweight="medium")
    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Number of Queries", fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # 2. Latency distribution
    latency_data = df["avg_execution_time"]
    if latency_data.max() > 0:
        min_latency = max(latency_data.min(), 1e-9)
        latency_bins = np.logspace(np.log10(min_latency), np.log10(latency_data.max()), 15)
        ax2.hist(
            latency_data,
            bins=latency_bins,
            color=latency_color,
            alpha=0.7,
            edgecolor="black"
        )
        ax2.set_xscale("log")
    else:
        ax2.hist(
            latency_data,
            bins=15,
            color=latency_color,
            alpha=0.7,
            edgecolor="black"
        )
    
    ax2.set_title("Latency Distribution (s)", fontsize=14, fontweight="medium")
    ax2.set_xlabel("Average Execution Time (s)", fontsize=12)
    ax2.set_ylabel("Number of Queries", fontsize=12)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{prefix}_FVS_result_distribution.png", dpi=300, bbox_inches="tight")

