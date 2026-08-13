"""Federated training loop over the non-IID client shards, using Flower's
real FedProx/FedAvg strategy objects for aggregation.

Flower's actual simulation transport (flwr.simulation.start_simulation /
run_simulation) requires the `ray` package, which has no build for Python
3.14 on Windows (confirmed: `pip install ray` resolves to zero versions).
So this script drives the round loop itself in a plain sequential `for`
loop instead of through Flower's Ray-based transport - but the aggregation
math is still the real, unmodified flwr.server.strategy.FedProx / FedAvg
(.aggregate_fit / .aggregate_evaluate / .evaluate), called directly. Only
the network/parallelism layer is swapped for direct Python calls; the ML
result is unaffected.

Usage (from repo root, with the venv active):
    python scripts/run_fl.py
    python scripts/run_fl.py --strategy fedavg
    python scripts/run_fl.py --strategy fedprox --proximal-mu 1.0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from flwr.common import Code, EvaluateRes, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedProx
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fl_client import get_parameters, make_client_fn, set_parameters
from model import MaternalRiskMLP
from partition import dirichlet_partition
from preprocessing import load_raw, preprocess

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "maternal_health_risk.csv")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedprox")
    p.add_argument("--proximal-mu", type=float, default=0.1, help="ignored for fedavg")
    p.add_argument("--num-rounds", type=int, default=15)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def weighted_average(metrics_list):
    """metrics_list: list of (num_examples, {metric_name: value}) -> weighted mean per key."""
    total = sum(n for n, _ in metrics_list)
    keys = metrics_list[0][1].keys()
    return {k: sum(n * m[k] for n, m in metrics_list) / total for k in keys}


def make_leakage_safe_shards(X, y, num_clients, alpha, seed, shards_path):
    """Same stratified 80/20 split as step 4's baseline, then Dirichlet-
    partition ONLY the train side, mapping shard indices back to their
    position in the full array - so clients never see held-out test rows.
    """
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices, test_size=0.2, stratify=y, random_state=seed
    )
    client_local_indices = dirichlet_partition(y_train, num_clients, alpha, seed)
    client_global_indices = [train_idx[local_idx] for local_idx in client_local_indices]

    os.makedirs(os.path.dirname(shards_path), exist_ok=True)
    np.savez(shards_path, **{f"client_{i}": idx for i, idx in enumerate(client_global_indices)})

    return X_train, y_train, X_test, y_test


def make_evaluate_fn(X_test, y_test):
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    def evaluate_fn(server_round, parameters_ndarrays, config):
        model = MaternalRiskMLP()
        set_parameters(model, parameters_ndarrays)
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            loss = torch.nn.functional.cross_entropy(logits, y_test_t).item()
            accuracy = (logits.argmax(dim=1) == y_test_t).float().mean().item()
        return loss, {"accuracy": accuracy}

    return evaluate_fn


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    df = load_raw(RAW_PATH)
    X, y, _ = preprocess(df)

    shards_path = os.path.join(DATA_DIR, f"fl_shards_a{args.alpha}_n{args.num_clients}.npz")
    X_train, y_train, X_test, y_test = make_leakage_safe_shards(
        X, y, args.num_clients, args.alpha, args.seed, shards_path
    )
    print(f"Split: {len(X_train)} train / {len(X_test)} test "
          f"(expect 811 / 203, matching step 4's baseline split)\n")

    client_fn = make_client_fn(shards_path, RAW_PATH)
    clients = [client_fn(str(i)) for i in range(args.num_clients)]

    evaluate_fn = make_evaluate_fn(X_test, y_test)

    def on_fit_config_fn(server_round):
        return {"local_epochs": args.local_epochs, "lr": args.lr}

    strategy_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    if args.strategy == "fedprox":
        strategy = FedProx(proximal_mu=args.proximal_mu, **strategy_kwargs)
    else:
        strategy = FedAvg(**strategy_kwargs)
    print(f"Strategy: {strategy}\n")

    model = MaternalRiskMLP()
    parameters = ndarrays_to_parameters(get_parameters(model))

    history = []
    for server_round in range(1, args.num_rounds + 1):
        fit_config = on_fit_config_fn(server_round)
        if isinstance(strategy, FedProx):
            fit_config = {**fit_config, "proximal_mu": strategy.proximal_mu}

        params_ndarrays = parameters_to_ndarrays(parameters)
        fit_results = []
        for client in clients:
            new_params, num_examples, metrics = client.fit(params_ndarrays, fit_config)
            fit_res = FitRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(new_params),
                num_examples=num_examples,
                metrics=metrics,
            )
            fit_results.append((None, fit_res))
        parameters, fit_metrics = strategy.aggregate_fit(server_round, fit_results, [])

        params_ndarrays = parameters_to_ndarrays(parameters)
        eval_results = []
        for client in clients:
            loss, num_examples, metrics = client.evaluate(params_ndarrays, {})
            eval_res = EvaluateRes(
                status=Status(Code.OK, ""), loss=loss, num_examples=num_examples, metrics=metrics
            )
            eval_results.append((None, eval_res))
        fed_loss, fed_metrics = strategy.aggregate_evaluate(server_round, eval_results, [])

        central_loss, central_metrics = strategy.evaluate(server_round, parameters)

        print(
            f"round {server_round:3d}/{args.num_rounds}  "
            f"centralized: loss={central_loss:.4f} acc={central_metrics['accuracy']:.4f}  |  "
            f"federated: loss={fed_loss:.4f} acc={fed_metrics['accuracy']:.4f}  |  "
            f"train_loss={fit_metrics['train_loss']:.4f}"
        )
        history.append({
            "round": server_round,
            "centralized_loss": central_loss,
            "centralized_accuracy": central_metrics["accuracy"],
            "federated_loss": fed_loss,
            "federated_accuracy": fed_metrics["accuracy"],
            "train_loss": fit_metrics["train_loss"],
        })

    out_path = os.path.join(
        DATA_DIR, f"fl_history_{args.strategy}_a{args.alpha}_n{args.num_clients}_r{args.num_rounds}.csv"
    )
    pd.DataFrame(history).to_csv(out_path, index=False)
    print(f"\nSaved round-by-round history to {out_path}")
    print(f"Final centralized accuracy: {history[-1]['centralized_accuracy']:.4f} "
          f"(step 4 centralized baseline: 0.6995)")


if __name__ == "__main__":
    main()
