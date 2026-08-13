"""Run FedAvg and FedProx on the identical non-IID split and plot
accuracy/loss vs. round for both - the main demo visual.

Usage (from repo root, with the venv active):
    python scripts/compare_strategies.py
"""

import argparse
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

# From scripts/train_baseline.py's default run (see step 4) - centralized,
# non-federated, same stratified 80/20 split / seed the FL runs also use.
CENTRALIZED_BASELINE_ACCURACY = 0.6995

STRATEGY_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx"}
STRATEGY_COLORS = {"fedavg": "#2a78d6", "fedprox": "#eb6834"}  # dataviz skill, slots 1-2

# Chart chrome, from the dataviz skill's light-mode reference palette
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-rounds", type=int, default=15)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--proximal-mu", type=float, default=0.1)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_strategy(strategy, args) -> pd.DataFrame:
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_fl.py"),
        "--strategy", strategy,
        "--num-rounds", str(args.num_rounds),
        "--num-clients", str(args.num_clients),
        "--alpha", str(args.alpha),
        "--proximal-mu", str(args.proximal_mu),
        "--local-epochs", str(args.local_epochs),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
    ]
    print(f"--- running {strategy} ---")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)

    match = re.search(r"Saved round-by-round history to (.+\.csv)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not find output CSV path in {strategy} run's output")
    return pd.read_csv(match.group(1).strip())


def plot_comparison(histories: dict[str, pd.DataFrame], out_path: str):
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)

    for ax in (ax_acc, ax_loss):
        ax.set_facecolor(SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_xlabel("Round", color=INK_SECONDARY)
        ax.tick_params(colors=INK_MUTED)

    for strategy, df in histories.items():
        color = STRATEGY_COLORS[strategy]
        label = STRATEGY_LABELS[strategy]
        ax_acc.plot(df["round"], df["centralized_accuracy"], color=color, linewidth=2,
                    solid_capstyle="round", solid_joinstyle="round", label=label)
        ax_acc.scatter(df["round"].iloc[-1], df["centralized_accuracy"].iloc[-1],
                        color=color, s=50, zorder=3, edgecolor=SURFACE, linewidth=2)

        ax_loss.plot(df["round"], df["centralized_loss"], color=color, linewidth=2,
                     solid_capstyle="round", solid_joinstyle="round", label=label)
        ax_loss.scatter(df["round"].iloc[-1], df["centralized_loss"].iloc[-1],
                         color=color, s=50, zorder=3, edgecolor=SURFACE, linewidth=2)

    ax_acc.axhline(CENTRALIZED_BASELINE_ACCURACY, color=INK_MUTED, linewidth=1.5, linestyle="--")
    ax_acc.annotate(
        f"Centralized baseline {CENTRALIZED_BASELINE_ACCURACY:.1%}",
        (1, CENTRALIZED_BASELINE_ACCURACY), textcoords="offset points", xytext=(0, 6),
        color=INK_SECONDARY, fontsize=9,
    )

    ax_acc.set_ylabel("Centralized test accuracy", color=INK_SECONDARY)
    ax_acc.set_title("Accuracy vs. round", color=INK_PRIMARY, fontsize=12)
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(frameon=False, loc="lower right", labelcolor=INK_SECONDARY)

    ax_loss.set_ylabel("Centralized test loss", color=INK_SECONDARY)
    ax_loss.set_title("Loss vs. round", color=INK_PRIMARY, fontsize=12)
    ax_loss.legend(frameon=False, loc="upper right", labelcolor=INK_SECONDARY)

    fig.suptitle(
        "FedProx vs. FedAvg on non-IID client shards", color=INK_PRIMARY, fontsize=13, y=1.02
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    histories = {strategy: run_strategy(strategy, args) for strategy in ["fedavg", "fedprox"]}

    print("\n=== Final round summary ===")
    print(f"{'strategy':<10} {'accuracy':>10} {'loss':>10}")
    for strategy, df in histories.items():
        last = df.iloc[-1]
        print(f"{STRATEGY_LABELS[strategy]:<10} {last['centralized_accuracy']:>10.4f} {last['centralized_loss']:>10.4f}")
    print(f"{'baseline':<10} {CENTRALIZED_BASELINE_ACCURACY:>10.4f} {'-':>10}")

    out_path = os.path.join(FIGURES_DIR, "fedprox_vs_fedavg.png")
    plot_comparison(histories, out_path)
    print(f"\nSaved comparison chart to {out_path}")


if __name__ == "__main__":
    main()
