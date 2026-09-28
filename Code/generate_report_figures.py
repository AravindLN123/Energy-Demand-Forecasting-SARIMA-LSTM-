"""Generate matplotlib figures for the Packages chapter of the report.

Outputs three PNG files to report/Images/:
    fig_pandas_timeseries.png  -- PJME daily demand + 7-day moving average
    fig_sarima_forecast.png    -- SARIMA 720-step forecast vs actual + 95% CI
    fig_sarima_acf.png         -- Residual ACF plot

Usage (run from the project root directory):
    python Code/Python/EnergyDemandForecaster/scripts/generate_report_figures.py \\
        --data <path/to/PJME_hourly.csv>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent   # BA26-01-Time-Series/
IMAGES_DIR   = PROJECT_ROOT / "report" / "Images"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pjme(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
    df = df[~df.index.duplicated(keep="first")].asfreq("h")
    return df["PJME_MW"]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
    })


# ---------------------------------------------------------------------------
# Figure 1: Pandas time-series + 7-day moving average
# ---------------------------------------------------------------------------

def fig_pandas_timeseries(series: pd.Series, out_dir: Path) -> None:
    daily    = series.resample("D").mean()
    smoothed = daily.rolling(window=7).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily,    color="steelblue", alpha=0.45, linewidth=0.8,
            label="Daily mean (MW)")
    ax.plot(smoothed, color="darkorange", linewidth=1.8, linestyle="--",
            label="7-day rolling average")
    ax.set_title("PJME Daily Energy Consumption (2002–2018)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Consumption (MW)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    out = out_dir / "fig_pandas_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: SARIMA 720-step forecast vs actual + 95% CI
# ---------------------------------------------------------------------------

def fig_sarima_forecast(series: pd.Series, out_dir: Path) -> None:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Forward-fill any NaN gaps so the DatetimeIndex stays strictly hourly.
    # dropna() would create an irregular index, causing statsmodels to infer
    # wrong forecast timestamps and produce an empty gap on the plot.
    subset = series.iloc[-8760:].ffill()   # last ~1 year, hourly index intact
    train  = subset.iloc[:-720]
    test   = subset.iloc[-720:]

    print("  Fitting SARIMA (this may take ~30 s)…")
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit    = model.fit(disp=False, maxiter=200)
    result = fit.get_forecast(steps=720)
    point  = result.predicted_mean
    ci     = result.conf_int(alpha=0.05)

    # Zoom to the first 7 days (168 h) so forecast and actual overlap cleanly.
    SHOW = 168
    point_show = point.iloc[:SHOW]
    ci_show    = ci.iloc[:SHOW]
    test_show  = test.iloc[:SHOW]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train.iloc[-168:],  color="gray",      linewidth=0.8,
            label="Training (last 7 days)")
    ax.plot(test_show,          color="black",     linewidth=0.9,
            label="Actual")
    ax.plot(point_show,         color="steelblue", linewidth=1.2,
            label="SARIMA forecast")
    ax.fill_between(ci_show.index, ci_show.iloc[:, 0], ci_show.iloc[:, 1],
                    color="steelblue", alpha=0.20, label="95% CI")
    ax.set_title("SARIMA(1,1,1)(1,1,1,24) — First 7 Days of 720-Hour Forecast",
                 fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (MW)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    out = out_dir / "fig_sarima_forecast.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    return fit   # return for ACF plot


# ---------------------------------------------------------------------------
# Figure 3: Residual ACF plot
# ---------------------------------------------------------------------------

def fig_sarima_acf(fit, out_dir: Path) -> None:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    residuals = pd.Series(fit.resid).dropna()

    fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(14, 4))

    # zero=False drops the trivial lag-0 bar (ACF=1) so the y-axis rescales
    # to the actual residual structure instead of being pinned at [-1, 1].
    plot_acf(residuals,  lags=48, zero=False, ax=ax_acf,
             color="steelblue", vlines_kwargs={"colors": "steelblue"})
    plot_pacf(residuals, lags=48, zero=False, ax=ax_pacf, method="ywm",
              color="darkorange", vlines_kwargs={"colors": "darkorange"})

    for ax, title in zip((ax_acf, ax_pacf),
                         ("Residual ACF", "Residual PACF")):
        ax.set_title(f"{title} — SARIMA(1,1,1)(1,1,1,24)", fontsize=12)
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Autocorrelation")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()

    out = out_dir / "fig_sarima_acf.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "PJME_hourly.csv",
        help="Path to PJME_hourly.csv (default: <project_root>/data/PJME_hourly.csv)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            "Pass the correct path with --data <path/to/PJME_hourly.csv>"
        )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    print("Loading data…")
    series = load_pjme(args.data)
    print(f"  {len(series):,} hourly rows loaded ({series.index.min()} → {series.index.max()})")

    print("\nFigure 1: Pandas time series…")
    fig_pandas_timeseries(series, IMAGES_DIR)

    print("\nFigure 2: SARIMA forecast…")
    fit = fig_sarima_forecast(series, IMAGES_DIR)

    print("\nFigure 3: Residual ACF…")
    fig_sarima_acf(fit, IMAGES_DIR)

    print("\nAll figures saved to:", IMAGES_DIR)


if __name__ == "__main__":
    main()
