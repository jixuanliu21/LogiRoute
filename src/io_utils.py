"""Strict input validation for uploaded orders."""
import numpy as np
import pandas as pd

REQUIRED = ["id", "x", "y", "demand"]

def parse_orders(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    data = df[REQUIRED].copy()
    if len(data) < 2 or len(data) > 201:
        raise ValueError("Expected 1 depot and 1-200 customers")
    if data["id"].isna().any() or data["id"].astype(str).duplicated().any():
        raise ValueError("IDs must be unique and nonempty")
    for col in ["x", "y", "demand"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if not np.isfinite(data[["x", "y", "demand"]].to_numpy()).all():
        raise ValueError("Coordinates and demands must be finite numbers")
    if (data["demand"] < 0).any() or (data["demand"] % 1 != 0).any():
        raise ValueError("Demands must be nonnegative integers")
    if data.iloc[0]["demand"] != 0:
        raise ValueError("First row must be the depot with demand 0")
    if (data.iloc[1:]["demand"] <= 0).any():
        raise ValueError("Each customer must have positive demand")
    data["demand"] = data["demand"].astype(int)
    return data.reset_index(drop=True)
