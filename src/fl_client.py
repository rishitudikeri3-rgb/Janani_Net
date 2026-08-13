"""Flower NumPyClient wrapping MaternalRiskMLP - each simulated hospital
runs one of these, training on its own Dirichlet shard from step 3 and
never sharing raw data, only model parameters.
"""

from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import MaternalRiskMLP
from preprocessing import load_raw, preprocess


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """PyTorch state_dict -> Flower's flat list-of-NumPy-arrays format."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load Flower's flat list-of-NumPy-arrays format into a PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def load_client_data(
    client_id: int, shards_path: str, raw_path: str, val_fraction: float = 0.2, seed: int = 42
):
    """Full dataset + this client's shard indices -> local train/val split.

    Preprocessing (scaling/label encoding) is global, per step 2 - only the
    split into per-client shards is what makes this non-IID. Falls back to
    an unstratified split when the shard is too skewed/small to stratify
    (a class with <2 samples), which happens at low Dirichlet alpha.
    """
    df = load_raw(raw_path)
    X, y, _ = preprocess(df)

    with np.load(shards_path) as shards:
        idx = shards[f"client_{client_id}"]
    X_client, y_client = X[idx], y[idx]

    class_counts = np.bincount(y_client, minlength=3)
    can_stratify = np.all(class_counts[class_counts > 0] >= 2) and len(X_client) >= 5
    stratify = y_client if can_stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_client, y_client, test_size=val_fraction, stratify=stratify, random_state=seed
    )
    return X_train, y_train, X_val, y_val


class MaternalRiskClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.long)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        # snapshot of the global weights the server just sent, needed for
        # the FedProx proximal term below
        global_params = [p.clone().detach() for p in self.model.parameters()]

        local_epochs = config.get("local_epochs", 5)
        lr = config.get("lr", 1e-3)
        mu = config.get("proximal_mu", 0.0)
        batch_size = config.get("batch_size", 16)

        train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train), batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epoch_loss = 0.0
        for _ in range(local_epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                if mu > 0:
                    proximal_term = sum(
                        (local_w - global_w).norm(2) ** 2
                        for local_w, global_w in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (mu / 2) * proximal_term
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(self.X_train)

        return get_parameters(self.model), len(self.X_train), {"train_loss": epoch_loss}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_val)
            loss = criterion(logits, self.y_val)
            accuracy = (logits.argmax(dim=1) == self.y_val).float().mean().item()

        return float(loss.item()), len(self.X_val), {"accuracy": accuracy}


def make_client_fn(shards_path: str, raw_path: str):
    """Closure Flower's start_simulation calls with a client id string."""

    def client_fn(cid: str) -> MaternalRiskClient:
        X_train, y_train, X_val, y_val = load_client_data(int(cid), shards_path, raw_path)
        model = MaternalRiskMLP()
        return MaternalRiskClient(model, X_train, y_train, X_val, y_val)

    return client_fn
