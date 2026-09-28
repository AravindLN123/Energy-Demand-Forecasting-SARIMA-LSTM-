"""Train a peak-aware LSTM with quantile (pinball) loss.

Reproduces the exact train/val/test split from ``notebooks/04_lstm_model.ipynb``
so the resulting checkpoint is directly comparable to ``models/lstm_pjme.pt``,
the difference being the loss function only:

* Standard checkpoint: MSE loss      — symmetric, learns the conditional mean,
                                       under-predicts peaks by ~25 %.
* This checkpoint:    Pinball loss   — at tau=0.7, under-predictions are
                                       penalised 0.7× while over-predictions
                                       are penalised 0.3×, biasing the model
                                       toward higher (peak-recovering) outputs.

Reports MAE / RMSE on the held-out test window for the report's comparison
table. Writes:
  * models/lstm_pjme_q{TAU*100}.pt   — model state_dict
  * results/metrics.json              — appended metric record
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_pjme  # noqa: E402
from src.preprocessing import ensure_hourly  # noqa: E402
from src.feature_engineering import build_lstm_features  # noqa: E402
from src.lstm_model import (  # noqa: E402
    INPUT_LEN,
    OUTPUT_LEN,
    TARGET_COL,
    TrainConfig,
    train_lstm,
    forecast as lstm_forecast,
    save_model,
)
from src.evaluation import mae, rmse, append_metrics  # noqa: E402


TEST_HOURS = OUTPUT_LEN              # 720
VAL_HOURS = 24 * 30 * 6              # ~6 months
TRAIN_HOURS = 24 * 365 * 10          # ~10 years
TAU = 0.7                            # bias toward peak recovery

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "PJME_hourly.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS = PROJECT_ROOT / "results" / "metrics.json"
OUT_PATH = MODELS_DIR / f"lstm_pjme_q{int(TAU * 100):02d}.pt"


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: {RAW_CSV} not found.", file=sys.stderr)
        return 1

    print(f"Loading {RAW_CSV.name} …")
    raw = load_pjme(RAW_CSV)
    clean = ensure_hourly(raw)[[TARGET_COL]]
    features = build_lstm_features(clean, target=TARGET_COL)

    test_df = features.iloc[-TEST_HOURS:]
    val_df = features.iloc[-(TEST_HOURS + VAL_HOURS):-TEST_HOURS]
    train_df = features.iloc[
        -(TEST_HOURS + VAL_HOURS + TRAIN_HOURS):-(TEST_HOURS + VAL_HOURS)
    ]
    print(
        f"Split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    scaler = MinMaxScaler().fit(train_df.values)
    train_arr = scaler.transform(train_df.values)
    val_arr = scaler.transform(val_df.values)
    test_arr = scaler.transform(test_df.values)
    target_min = float(scaler.data_min_[0])
    target_range = float(scaler.data_max_[0] - target_min)
    print(f"Target {TARGET_COL}: min={target_min:.1f} range={target_range:.1f}")

    cfg = TrainConfig()
    cfg.loss_type = "quantile"
    cfg.quantile_tau = TAU
    print(
        f"Training LSTM with pinball loss (tau={TAU}) on {cfg.device.upper()} "
        f"for up to {cfg.max_epochs} epochs…"
    )

    model, history = train_lstm(train_arr, val_arr, cfg)
    epochs_run = len(history["train_loss"])
    print(f"Trained {epochs_run} epochs (early-stopping honored)")

    # ---- Evaluate on held-out test window ------------------------------
    # Use the last input_len rows of val as the input window — same as the
    # notebook's evaluation, so MAE/RMSE are directly comparable.
    input_window = val_arr[-cfg.input_len:]
    scaled_pred = lstm_forecast(model, input_window, device=cfg.device)
    pred = scaled_pred * target_range + target_min
    actual = test_df[TARGET_COL].values

    metrics = {
        "MAE": float(mae(actual, pred)),
        "RMSE": float(rmse(actual, pred)),
        "epochs_run": epochs_run,
        "input_len": cfg.input_len,
        "output_len": cfg.output_len,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "loss_type": cfg.loss_type,
        "quantile_tau": cfg.quantile_tau,
        "device": cfg.device,
    }
    print("Metrics:", metrics)

    save_model(model, OUT_PATH)
    print(f"Saved model → {OUT_PATH}")

    # Tag the metrics record with the quantile so it's distinguishable from
    # the MSE-trained baseline already in the file.
    append_metrics(
        METRICS,
        model=f"LSTM-q{int(TAU * 100):02d}",
        region="PJME",
        horizon=len(test_df),
        metrics=metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
