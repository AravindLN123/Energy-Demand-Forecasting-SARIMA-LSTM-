"""Load the PJME hourly energy consumption dataset."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_pjme(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load PJME hourly load data and return a DataFrame indexed by Datetime.

    Accepts either a filesystem path to the CSV or an already-loaded DataFrame
    (the latter lets the future Streamlit GUI pass an in-memory upload directly).

    Returns a DataFrame with a DatetimeIndex and a single ``PJME_MW`` column,
    sorted ascending by time and de-duplicated on the index.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    if "Datetime" not in df.columns:
        raise ValueError(
            f"Expected a 'Datetime' column, found: {list(df.columns)}"
        )

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()

    # Collapse DST fall-back duplicates by averaging the two readings.
    if df.index.has_duplicates:
        df = df.groupby(level=0).mean()

    return df
