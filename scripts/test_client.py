"""Standalone smoke test for MaternalRiskClient - no Flower server involved
yet (that's step 6). Manually drives get_parameters() -> fit() -> evaluate()
for one client to prove the wiring works.

Usage (from repo root, with the venv active):
    python scripts/test_client.py
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fl_client import make_client_fn

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "maternal_health_risk.csv")
SHARDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "client_shards_a0.5_n5.npz"
)


def main():
    if not os.path.exists(SHARDS_PATH):
        print(f"{SHARDS_PATH} not found, generating it with default settings first...")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "partition_data.py")], check=True)

    client_fn = make_client_fn(SHARDS_PATH, RAW_PATH)
    client = client_fn("0")

    initial_params = client.get_parameters(config={})
    print(f"get_parameters(): {len(initial_params)} arrays "
          f"(expect 6: 3 Linear layers x weight+bias)")
    for arr in initial_params:
        print(f"  shape {arr.shape}")

    print("\nfit() with proximal_mu=0.0 (plain local training, as under FedAvg):")
    updated_params, num_examples, fit_metrics = client.fit(initial_params, {"proximal_mu": 0.0})
    print(f"  num_examples={num_examples}  metrics={fit_metrics}")

    print("\nfit() again with proximal_mu=1.0 (as under FedProx):")
    _, _, fit_metrics_prox = client.fit(initial_params, {"proximal_mu": 1.0})
    print(f"  metrics={fit_metrics_prox}")
    print("  (proximal loss should differ from the mu=0.0 run above - "
          "confirms the proximal term is actually being added)")

    print("\nevaluate() with the fit()-updated parameters:")
    loss, num_examples, eval_metrics = client.evaluate(updated_params, {})
    print(f"  num_examples={num_examples}  loss={loss:.4f}  metrics={eval_metrics}")


if __name__ == "__main__":
    main()
