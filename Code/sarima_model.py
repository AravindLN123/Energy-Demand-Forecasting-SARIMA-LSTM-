"""SARIMA fitting and forecasting wrappers around statsmodels SARIMAX."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 24)  # daily seasonality, hourly data


def fit_sarima(
    train: pd.Series,
    order: Tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: Tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    maxiter: int = 200,
) -> SARIMAXResults:
    """Fit a SARIMA(p,d,q)(P,D,Q,s) model via state-space MLE.

    Stationarity and invertibility enforcement are disabled because the
    optimizer otherwise tends to fail on hourly load data.
    """
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=maxiter)


def forecast(
    fit: SARIMAXResults, steps: int = 720, alpha: float = 0.05
) -> Tuple[pd.Series, pd.DataFrame]:
    """Return point forecast and (1-alpha) confidence interval for ``steps`` ahead."""
    result = fit.get_forecast(steps=steps)
    point = result.predicted_mean
    conf_int = result.conf_int(alpha=alpha)
    return point, conf_int


def save_model(fit: SARIMAXResults, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fit.save(str(path))


def load_model(path: Path) -> SARIMAXResults:
    return SARIMAXResults.load(str(path))
