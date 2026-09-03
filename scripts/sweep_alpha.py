"""Sweep Dirichlet alpha (client skew) x seed for both strategies, to test
the claim in the README's caveats: FedProx's proximal term is expected to
matter more under *stronger* non-IID skew (lower alpha) than the single
alpha=0.5 run in the main comparison shows.

Reuses run_fl.py via subprocess exactly like compare_strategies.py does,
just looped over alpha and seed as well as strategy. Averaging across
seeds also gives error bars, so multi-seed variance and the alpha sweep
come out of one script instead of two.

Usage (from repo root, with the venv active):
    python scripts/sweep_alpha.py
    python scripts/sweep_alpha.py --alphas 0.1,0.3,0.5 --seeds 42,43,44
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
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")

# Same constant as compare_strategies.py - centralized, non-federated,
# same stratified 80/20 split / seed the FL runs also use.
CENTRALIZED_BASELINE_ACCURACY = 0.6995

STRATEGY_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx"}
STRATEGY_COLORS = {"fedavg": "#2a78d6", "fedprox": "#eb6834"}  # dataviz skill, validated pair

# Chart chrome, from the dataviz skill's light-mode reference palette
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", type=str, default="0.1,0.3,0.5")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--proximal-mu", type=float, default=0.1)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--heterogeneous-epochs", action="store_true",
        help="passthrough to run_fl.py - simulate clients with different local compute budgets, "
             "the systems-heterogeneity scenario FedProx's proximal term targets",
    )
    p.add_argument("--epochs-min", type=int, default=2)
    p.add_argument("--epochs-max", type=int, default=8)
    return p.parse_args()


def run_one(strategy, alpha, seed, args) -> float:
    """Runs run_fl.py once, returns its final-round centralized_accuracy."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_fl.py"),
        "--strategy", strategy,
        "--alpha", str(alpha),
        "--seed", str(seed),
        "--num-rounds", str(args.num_rounds),
        "--num-clients", str(args.num_clients),
        "--proximal-mu", str(args.proximal_mu),
        "--local-epochs", str(args.local_epochs),
        "--lr", str(args.lr),
    ]
    if args.heterogeneous_epochs:
        cmd += ["--heterogeneous-epochs", "--epochs-min", str(args.epochs_min),
                "--epochs-max", str(args.epochs_max)]
    print(f"--- {strategy} alpha={alpha} seed={seed} ---")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    match = re.search(r"Saved round-by-round history to (.+\.csv)", result.stdout)
    if not match:
        raise RuntimeError(
            f"Could not find output CSV path in {strategy}/alpha={alpha}/seed={seed} run's output"
        )
    df = pd.read_csv(match.group(1).strip())
    final_acc = df["centralized_accuracy"].iloc[-1]
    print(f"    final centralized accuracy: {final_acc:.4f}")
    return final_acc


def plot_sweep(summary: pd.DataFrame, out_path: str, heterogeneous: bool, epochs_min: int, epochs_max: int):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED)

    for strategy in ["fedavg", "fedprox"]:
        rows = summary[summary["strategy"] == strategy].sort_values("alpha")
        color = STRATEGY_COLORS[strategy]
        ax.errorbar(
            rows["alpha"], rows["mean_accuracy"], yerr=rows["std_accuracy"],
            color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5,
            capsize=4, elinewidth=1.5, ecolor=color,
            label=STRATEGY_LABELS[strategy],
        )

    ax.axhline(CENTRALIZED_BASELINE_ACCURACY, color=INK_MUTED, linewidth=1.5, linestyle="--")
    ax.annotate(
        f"Centralized baseline {CENTRALIZED_BASELINE_ACCURACY:.1%}",
        (ax.get_xlim()[0], CENTRALIZED_BASELINE_ACCURACY), textcoords="offset points",
        xytext=(0, 6), color=INK_SECONDARY, fontsize=9,
    )

    ax.set_xlabel("Dirichlet alpha (lower = more skewed clients)", color=INK_SECONDARY)
    ax.set_ylabel("Final centralized test accuracy", color=INK_SECONDARY)
    subtitle = (
        f"(mean ± std over 3 seeds, heterogeneous local epochs {epochs_min}-{epochs_max} per client)"
        if heterogeneous else
        "(mean ± std over 3 seeds)"
    )
    ax.set_title(
        f"FedProx vs. FedAvg across client skew\n{subtitle}",
        color=INK_PRIMARY, fontsize=13,
    )
    ax.set_ylim(0, 1)
    ax.invert_xaxis()  # skew increases left-to-right, matching "stronger skew ->" reading
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_SECONDARY)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for alpha in alphas:
        for seed in seeds:
            for strategy in ["fedavg", "fedprox"]:
                acc = run_one(strategy, alpha, seed, args)
                rows.append({"alpha": alpha, "seed": seed, "strategy": strategy, "final_accuracy": acc})

    results = pd.DataFrame(rows)
    het_suffix = "_heterogeneous" if args.heterogeneous_epochs else ""
    results_path = os.path.join(DATA_DIR, f"alpha_sweep{het_suffix}_results.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"\nSaved raw sweep results to {results_path}")

    summary = (
        results.groupby(["alpha", "strategy"])["final_accuracy"]
        .agg(mean_accuracy="mean", std_accuracy="std")
        .reset_index()
    )
    summary["std_accuracy"] = summary["std_accuracy"].fillna(0.0)

    print("\n=== Summary (mean ± std across seeds) ===")
    print(f"{'alpha':>6} {'strategy':<10} {'mean':>8} {'std':>8}")
    for _, row in summary.sort_values(["alpha", "strategy"]).iterrows():
        print(f"{row['alpha']:>6} {STRATEGY_LABELS[row['strategy']]:<10} "
              f"{row['mean_accuracy']:>8.4f} {row['std_accuracy']:>8.4f}")
    print(f"{'':>6} {'baseline':<10} {CENTRALIZED_BASELINE_ACCURACY:>8.4f} {'-':>8}")

    out_path = os.path.join(FIGURES_DIR, f"alpha_sweep{het_suffix}.png")
    plot_sweep(summary, out_path, args.heterogeneous_epochs, args.epochs_min, args.epochs_max)
    print(f"\nSaved sweep chart to {out_path}")


if __name__ == "__main__":
    main()
