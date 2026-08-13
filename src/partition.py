"""Non-IID data partitioning for FL client simulation.

Splits a labeled dataset across N clients with a Dirichlet-distribution
label skew, the standard way to simulate non-IID data in FL benchmarks
(Hsu et al., 2019). Low alpha -> clients dominated by one class (extreme
skew, like a real specialist clinic). High alpha -> every client's class
mix looks close to the global mix (near-IID, like well-mixed hospitals).
"""

import numpy as np
import pandas as pd


def dirichlet_partition(
    y: np.ndarray, num_clients: int, alpha: float, seed: int = 42
) -> list[np.ndarray]:
    """Split sample indices across clients with per-class Dirichlet skew.

    For each class independently, draw a Dirichlet(alpha) vector over the
    clients and hand that class's indices out in those proportions. Total
    samples per client end up unequal as a side effect - real hospitals
    don't see equal patient volumes either.
    """
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for c in np.unique(y):
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet([alpha] * num_clients)
        # cumulative proportions -> split points into class_idx, dropping
        # the trailing 1.0 which np.split would otherwise treat as an
        # extra (empty) split point
        split_points = (np.cumsum(proportions)[:-1] * len(class_idx)).astype(int)
        for client_id, idx_chunk in enumerate(np.split(class_idx, split_points)):
            client_indices[client_id].extend(idx_chunk)

    # dtype=int matters here: a client can legitimately end up with zero
    # samples at low alpha, and np.array([]) defaults to float64, which
    # breaks integer indexing (y[idx]) downstream.
    return [np.array(idx, dtype=int) for idx in client_indices]


def client_class_counts(y: np.ndarray, client_indices: list[np.ndarray]) -> pd.DataFrame:
    """Per-client counts of each class, for the distribution chart/summary."""
    classes = np.unique(y)
    rows = {
        client_id: [int(np.sum(y[idx] == c)) for c in classes]
        for client_id, idx in enumerate(client_indices)
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=classes)
