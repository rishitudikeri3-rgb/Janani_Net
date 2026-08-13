"""Preprocessing for the UCI Maternal Health Risk dataset.

Turns the raw CSV into model-ready arrays:
  - the 6 vital-sign features Min-Max scaled to [0, 1]
  - RiskLevel label-encoded to Low=0, Mid=1, High=2

Pure functions, no file I/O at import time, so this module can be imported
and unit-tested directly without touching disk.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

# Explicit mapping instead of sklearn.LabelEncoder: LabelEncoder assigns
# ids alphabetically (high risk=0, low risk=1, mid risk=2), which does not
# match the Low=0/Mid=1/High=2 ordering the project requires.
RISK_LABEL_MAP = {"low risk": 0, "mid risk": 1, "high risk": 2}


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw UCI CSV.

    encoding="utf-8-sig" strips the UTF-8 BOM the source file starts with
    (otherwise the first column reads as "﻿Age" instead of "Age").
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["RiskLevel"] = df["RiskLevel"].str.strip().str.lower()
    return df


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale the 6 features to [0, 1] and label-encode RiskLevel.

    Returns (X, y, scaler): the fitted scaler is returned too so callers can
    inspect it or reuse it later, rather than scaling in place with no
    handle on what was fitted.
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS].values)
    y = df["RiskLevel"].map(RISK_LABEL_MAP).values
    return X, y, scaler
