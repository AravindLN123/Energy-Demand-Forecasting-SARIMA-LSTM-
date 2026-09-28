"""Generate lstm_forecast_q70.png using the saved q70 model checkpoint.

Keeps the existing lstm_forecast.png (MSE model) untouched so both can be
shown side-by-side in the presentation as a comparison.

Usage (run from the EnergyDemandForecaster directory):
    python scripts/regenerate_lstm_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_pjme                    # noqa: E402
from src.preprocessing import ensure_hourly              # noqa: E402
from src.feature_engineering import build_lstm_features  # noqa: E402
from src.lstm_model import (                             # noqa: E402
    INPUT_LEN, OUTPUT_LEN, TARGET_COL,
    load_model, load_scaler, forecast,
)

MODELS_DIR  = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RAW_CSV     = PROJECT_ROOT / "data" / "raw" / "PJME_hourly.csv"


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: {RAW_CSV} not found.", file=sys.stderr)
        return 1

    print("Loading data and building features…")
    raw      = load_pjme(RAW_CSV)
    clean    = ensure_hourly(raw)[[TARGET_COL]]
    features = build_lstm_features(clean, target=TARGET_COL)

    # Input window = the 168 hours immediately before the 720-hour test window.
    test_df = features.iloc[-OUTPUT_LEN:]
    win_df  = features.iloc[-(OUTPUT_LEN + INPUT_LEN):-OUTPUT_LEN]

    print("Loading baked scaler and q70 checkpoint…")
    scaler = load_scaler(MODELS_DIR / "lstm_pjme.scaler.json")
    model  = load_model(
        MODELS_DIR / "lstm_pjme_q70.pt",
        n_features=len(scaler.feature_names),
    )

    input_window = scaler.transform(win_df.values)
    scaled_pred  = forecast(model, input_window)
    pred         = scaler.inverse_transform_target(scaled_pred)
    actual       = test_df[TARGET_COL].values

    print(f"Forecast done — MAE: {abs(actual - pred).mean():.1f} MW")

    plt.rcParams.update({
        "font.family":       "serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
    })

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_df.index, actual, color="black",       linewidth=0.9, label="Actual")
    ax.plot(test_df.index, pred,   color="forestgreen", linewidth=1.2,
            label="LSTM-q70 forecast (τ=0.70)")
    ax.set_title("LSTM-q70 (τ=0.70) — 720-Hour Forecast vs Actual", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (MW)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    out = FIGURES_DIR / "lstm_forecast_q70.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
