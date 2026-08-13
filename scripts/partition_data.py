"""Partition the dataset across simulated hospitals with a Dirichlet label
skew, and plot each client's resulting class distribution.

Usage (from repo root, with the venv active):
    python scripts/partition_data.py
    python scripts/partition_data.py --num-clients 5 --alpha 0.05  # more skew
    python scripts/partition_data.py --num-clients 5 --alpha 50    # near-IID
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from partition import client_class_counts, dirichlet_partition
from preprocessing import load_raw, preprocess

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "maternal_health_risk.csv")
SHARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

CLASS_NAMES = ["Low", "Mid", "High"]
# Fixed categorical order (dataviz skill palette, slots 1-3: blue/orange/aqua)
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# Chart chrome, from the dataviz skill's light-mode reference palette
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-clients", type=int, default=5, help="number of simulated hospitals")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration (lower = more skew)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def plot_distribution(counts, alpha, num_clients, out_path):
    """Stacked bar chart: one bar per client, segments = class counts.

    Part-to-whole per client is the story here (each hospital's case mix),
    so a stacked bar carries both the composition and the total-volume
    difference between clients in one shape.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    clients = counts.index.to_numpy()
    bottoms = np.zeros(len(clients))
    for class_id, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        values = counts[class_id].to_numpy()
        bars = ax.bar(
            clients, values, bottom=bottoms, width=0.6,
            color=color, label=name, edgecolor=SURFACE, linewidth=2,
        )
        # direct labels: required "relief" for the aqua segment's low
        # contrast, and just generally clearer than reading off the axis
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(value)),
                    ha="center", va="center",
                    color=INK_PRIMARY, fontsize=9,
                )
        bottoms += values

    ax.set_title(
        f"Class distribution per client (Dirichlet alpha={alpha}, {num_clients} clients)",
        color=INK_PRIMARY, fontsize=12, pad=12,
    )
    ax.set_xlabel("Client (simulated hospital)", color=INK_SECONDARY)
    ax.set_ylabel("Number of patients", color=INK_SECONDARY)
    ax.set_xticks(clients)
    ax.set_xticklabels([f"Client {c}" for c in clients], color=INK_MUTED)
    ax.tick_params(axis="y", colors=INK_MUTED)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)

    legend = ax.legend(
        title="Risk level", frameon=False, loc="upper right",
        labelcolor=INK_SECONDARY,
    )
    legend.get_title().set_color(INK_SECONDARY)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()

    df = load_raw(RAW_PATH)
    X, y, _ = preprocess(df)

    client_indices = dirichlet_partition(y, args.num_clients, args.alpha, args.seed)
    counts = client_class_counts(y, client_indices)
    counts.columns = [0, 1, 2]  # ensure predictable column order Low/Mid/High

    print(f"num_clients={args.num_clients} alpha={args.alpha} seed={args.seed}\n")
    print("Samples per client:", [len(idx) for idx in client_indices])
    print(f"Total: {sum(len(idx) for idx in client_indices)} (expected {len(y)})\n")
    print("Class counts per client (columns = Low, Mid, High):")
    print(counts.to_string())

    shards_path = os.path.join(SHARDS_DIR, f"client_shards_a{args.alpha}_n{args.num_clients}.npz")
    os.makedirs(SHARDS_DIR, exist_ok=True)
    np.savez(shards_path, **{f"client_{i}": idx for i, idx in enumerate(client_indices)})
    print(f"\nSaved shard indices to {shards_path}")

    figure_path = os.path.join(FIGURES_DIR, "client_class_distribution.png")
    plot_distribution(counts, args.alpha, args.num_clients, figure_path)
    print(f"Saved chart to {figure_path}")


if __name__ == "__main__":
    main()
