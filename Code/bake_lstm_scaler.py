"""Recompute the training-time MinMaxScaler stats and save them next to the
LSTM checkpoint as JSON.

The notebook only persisted ``model.state_dict()`` to ``models/lstm_pjme.pt``;
the scaler that mapped the 9 input features into [0, 1] was never saved. The
GUI therefore had no way to load the pre-trained model and reuse its scaling.

This script reproduces the exact train split from ``notebooks/04_lstm_model
.ipynb`` (TEST_HOURS=720, VAL_HOURS=4320, TRAIN_HOURS=87600), fits a
MinMaxScaler on the train slice, and writes the per-feature ``data_min_`` /
``data_max_`` plus column names to ``models/lstm_pjme.scaler.json``.

Idempotent: rerunning produces the same file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_pjme  # noqa: E402
from src.preprocessing import ensure_hourly  # noqa: E402
from src.feature_engineering import build_lstm_features  # noqa: E402
from src.lstm_model import INPUT_LEN, OUTPUT_LEN, TARGET_COL  # noqa: E402


# Match the notebook's split exactly.
TEST_HOURS = OUTPUT_LEN              # 720
VAL_HOURS = 24 * 30 * 6              # ~6 months
TRAIN_HOURS = 24 * 365 * 10          # ~10 years

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "PJME_hourly.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_pjme.pt"
OUT_PATH = PROJECT_ROOT / "models" / "lstm_pjme.scaler.json"


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: {RAW_CSV} not found. Cannot bake scaler.", file=sys.stderr)
        return 1
    if not MODEL_PATH.exists():
        print(f"WARNING: {MODEL_PATH} not found — scaler will be saved anyway, "
              "but no checkpoint exists to pair it with.", file=sys.stderr)

    raw = load_pjme(RAW_CSV)
    clean = ensure_hourly(raw)[[TARGET_COL]]
    features = build_lstm_features(clean, target=TARGET_COL)

    train_df = features.iloc[
        -(TEST_HOURS + VAL_HOURS + TRAIN_HOURS): -(TEST_HOURS + VAL_HOURS)
    ]
    print(
        f"Reproducing train split: {len(train_df):,} rows  "
        f"{train_df.index.min()} → {train_df.index.max()}"
    )

    scaler = MinMaxScaler().fit(train_df.values)

    payload = {
        "feature_names": list(train_df.columns),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "target_col": TARGET_COL,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "train_rows": len(train_df),
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    target_min = scaler.data_min_[0]
    target_range = scaler.data_max_[0] - target_min
    print(
        f"Wrote {OUT_PATH.name}  "
        f"(target {TARGET_COL}: min={target_min:.1f}  range={target_range:.1f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
