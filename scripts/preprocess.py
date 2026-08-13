"""Run preprocessing on the raw dataset and write the processed CSV.

Usage (from repo root, with the venv active):
    python scripts/preprocess.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import FEATURE_COLUMNS, load_raw, preprocess

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "maternal_health_risk.csv")
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "maternal_health_risk_processed.csv"
)


def main():
    df = load_raw(RAW_PATH)
    X, y, scaler = preprocess(df)

    out = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    out["RiskLevel"] = y
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out)} rows to {OUT_PATH}\n")

    print("Feature min/max after scaling (expect 0.0 / 1.0 for every column):")
    print(out[FEATURE_COLUMNS].agg(["min", "max"]))

    print("\nLabel counts (expect 406 low / 336 mid / 272 high -> 0/1/2):")
    print(out["RiskLevel"].value_counts().sort_index())


if __name__ == "__main__":
    main()
