"""Forecast evaluation metrics and diagnostic plots.

Plotting / Ljung-Box dependencies (matplotlib, statsmodels.graphics) are
imported lazily inside the relevant functions so that lightweight callers
which only need MAE/RMSE — like the GUI runtime bundle — don't have to
ship matplotlib.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def plot_forecast(
    train_tail: pd.Series,
    test: pd.Series,
    point_forecast: pd.Series,
    conf_int: Optional[pd.DataFrame] = None,
    title: str = "SARIMA forecast vs actual",
):
    """Plot the last slice of training data, the actual test values, and the
    forecast (with optional confidence band). Returns the Figure so callers
    (notebooks, Streamlit) can render or save it themselves.
    """
    import matplotlib.pyplot as plt  # lazy: notebook-only dependency

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_tail.index, train_tail.values, label="Train (recent)", color="gray")
    ax.plot(test.index, test.values, label="Actual", color="black")
    ax.plot(point_forecast.index, point_forecast.values, label="Forecast", color="tab:blue")
    if conf_int is not None:
        ax.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )
    ax.set_title(title)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    fig.tight_layout()
    return fig


def residual_diagnostics(fit, lags: int = 24) -> dict:
    """Run Ljung-Box on residuals and return the test statistic + p-value at ``lags``.

    A p-value > 0.05 suggests residuals are indistinguishable from white noise.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox  # lazy

    residuals = pd.Series(fit.resid).dropna()
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return {
        "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "lags": lags,
    }


def plot_residual_acf(fit, lags: int = 48):
    import matplotlib.pyplot as plt  # lazy
    from statsmodels.graphics.tsaplots import plot_acf  # lazy

    residuals = pd.Series(fit.resid).dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, lags=lags, ax=ax)
    ax.set_title("Residual ACF")
    fig.tight_layout()
    return fig


def append_metrics(
    metrics_path: Path,
    model: str,
    region: str,
    horizon: int,
    metrics: dict,
) -> None:
    """Append a metrics record to ``results/metrics.json`` (creating it if absent).

    The file stores a list so multiple model runs (SARIMA, LSTM, ...) coexist.
    """
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        records = json.loads(metrics_path.read_text())
        if not isinstance(records, list):
            records = [records]
    else:
        records = []

    records.append(
        {"model": model, "region": region, "horizon": horizon, **metrics}
    )
    metrics_path.write_text(json.dumps(records, indent=2))
