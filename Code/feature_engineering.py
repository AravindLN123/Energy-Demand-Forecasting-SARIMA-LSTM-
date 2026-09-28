"""Feature engineering for the LSTM model.

SARIMA captures seasonality through its seasonal AR/MA terms; LSTM is a
generic sequence model and benefits from explicit temporal features.

The functions here are deliberately framework-agnostic — they operate on
pandas DataFrames so the same logic can later feed the Streamlit GUI.
"""

from typing import Iterable, List

import numpy as np
import pandas as pd


CYCLICAL_PERIODS = {
    "hour": 24,
    "dayofweek": 7,
    "month": 12,
}

DEFAULT_LAGS = (24, 168)  # 1 day, 1 week


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings of hour-of-day, day-of-week, and month-of-year.

    Cyclical encoding gives the network a continuous representation that
    avoids the artificial 23 -> 0 jump a raw integer would produce.
    """
    out = df.copy()
    idx = out.index
    parts = {
        "hour": idx.hour.values,
        "dayofweek": idx.dayofweek.values,
        "month": (idx.month.values - 1),  # zero-based for symmetric sin/cos
    }
    for name, values in parts.items():
        period = CYCLICAL_PERIODS[name]
        radians = 2 * np.pi * values / period
        out[f"{name}_sin"] = np.sin(radians)
        out[f"{name}_cos"] = np.cos(radians)
    return out


def add_lag_features(
    df: pd.DataFrame,
    column: str = "PJME_MW",
    lags: Iterable[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add ``column``_lag_{k} for each k in ``lags`` (in hours).

    Rows with missing lag values (the first ``max(lags)`` rows) are dropped.
    """
    out = df.copy()
    for k in lags:
        out[f"{column}_lag_{k}"] = out[column].shift(k)
    return out.dropna(subset=[f"{column}_lag_{k}" for k in lags])


def build_lstm_features(
    df: pd.DataFrame,
    target: str = "PJME_MW",
    lags: Iterable[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Compose cyclical + lag features. Returns a DataFrame whose first column
    is the target so callers can split features and target by index.
    """
    out = add_cyclical_features(df[[target]])
    out = add_lag_features(out, column=target, lags=lags)
    cols: List[str] = [target] + [c for c in out.columns if c != target]
    return out[cols]
