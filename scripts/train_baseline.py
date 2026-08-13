"""Centralized (non-federated) training run - sanity-checks that the MLP
can learn on this data at all, and gives a baseline accuracy number to
compare FedProx/FedAvg against later.

Usage (from repo root, with the venv active):
    python scripts/train_baseline.py
    python scripts/train_baseline.py --epochs 200 --lr 0.0005
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import MaternalRiskMLP
from preprocessing import load_raw, preprocess

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "maternal_health_risk.csv")
CLASS_NAMES = ["Low", "Mid", "High"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    df = load_raw(RAW_PATH)
    X, y, _ = preprocess(df)

    # stratified so the smaller High-risk class (272/1014) isn't starved
    # out of the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = MaternalRiskMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train_loss {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
    accuracy = (preds == y_test_t).float().mean().item()

    print(f"\nTest accuracy: {accuracy:.4f} ({int((preds == y_test_t).sum())}/{len(y_test_t)})")
    print("\nPer-class report:")
    print(classification_report(y_test_t.numpy(), preds.numpy(), target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
