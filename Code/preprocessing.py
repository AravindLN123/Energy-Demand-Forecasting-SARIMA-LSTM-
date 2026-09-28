"""Cleaning and train/test splitting for hourly time-series load data."""

from typing import Tuple

import pandas as pd


def ensure_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a regular hourly frequency and linearly interpolate gaps.

    Adds a boolean ``imputed`` column flagging rows whose value was filled.
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    reindexed = df.reindex(full_index)
    imputed = reindexed.iloc[:, 0].isna()
    reindexed = reindexed.interpolate(method="linear")
    reindexed["imputed"] = imputed
    reindexed.index.name = "Datetime"
    return reindexed


def train_test_split_timeseries(
    df: pd.DataFrame,
    test_hours: int = 720,
    train_years: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train (last ``train_years`` before the test window) and test
    (last ``test_hours`` of the series). The full historical series is rarely
    needed for SARIMA — recent patterns dominate seasonal behaviour and shorter
    training windows keep the MLE fit tractable.
    """
    if len(df) < test_hours + train_years * 365 * 24:
        raise ValueError(
            "Series is too short for the requested split: "
            f"need >= {test_hours + train_years * 365 * 24} rows, got {len(df)}."
        )

    test = df.iloc[-test_hours:]
    train_start = test.index[0] - pd.DateOffset(years=train_years)
    train = df.loc[train_start : test.index[0] - pd.Timedelta(hours=1)]
    return train, test
