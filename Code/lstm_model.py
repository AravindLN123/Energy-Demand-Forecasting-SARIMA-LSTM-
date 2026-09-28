"""PyTorch LSTM forecaster for hourly load.

Predicts a fixed horizon (default 720 hours) directly from a fixed input
window of past observations + features. Direct multi-step output is chosen
so that one forward pass produces the whole forecast — matching the
"single 720-hour call" framing of the SARIMA baseline.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

INPUT_LEN = 168       # one week of past observations
OUTPUT_LEN = 720      # 30-day forecast horizon
TARGET_COL = "PJME_MW"


@dataclass
class TrainConfig:
    input_len: int = INPUT_LEN
    output_len: int = OUTPUT_LEN
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 30
    patience: int = 5
    train_stride: int = 6   # subsample windows during training to keep epochs fast
    seed: int = 42
    # Loss selection.
    # * "mse"      — default symmetric squared error (the original baseline).
    # * "quantile" — pinball loss at ``quantile_tau``; asymmetric, biases
    #                toward higher predictions to recover peaks. The shipped
    #                GUI checkpoint (``lstm_pjme_q70.pt``) was trained with
    #                this loss at tau=0.7.
    loss_type: str = "mse"
    quantile_tau: float = 0.5
    device: str = field(default_factory=lambda: (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    ))


class QuantileLoss(nn.Module):
    """Pinball loss for direct quantile regression.

    For ``tau > 0.5`` the loss penalises under-prediction more than
    over-prediction, which counteracts the peak-flattening bias of MSE on
    skewed targets like hourly energy load. Reduces to symmetric absolute
    error at ``tau = 0.5`` (a.k.a. MAE).
    """

    def __init__(self, tau: float = 0.5):
        super().__init__()
        if not 0.0 < tau < 1.0:
            raise ValueError(f"quantile tau must be in (0, 1); got {tau}")
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred
        return torch.mean(torch.maximum(self.tau * diff, (self.tau - 1.0) * diff))


class WindowedSeriesDataset(Dataset):
    """Yield (input_window, target_window) pairs from a feature matrix.

    ``features`` must be a numpy array shaped (n_rows, n_features). The first
    column is treated as the forecast target. Windows are extracted with
    a stride of ``stride`` to keep training tractable.
    """

    def __init__(
        self,
        features: np.ndarray,
        input_len: int,
        output_len: int,
        stride: int = 1,
    ):
        self.features = features.astype(np.float32)
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride
        last_start = len(features) - input_len - output_len
        if last_start < 0:
            raise ValueError(
                f"Not enough rows: need >= {input_len + output_len}, got {len(features)}"
            )
        self.starts = np.arange(0, last_start + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.input_len
        x = self.features[start:end]                                       # (input_len, n_features)
        y = self.features[end : end + self.output_len, 0]                  # (output_len,) — target column
        return torch.from_numpy(x), torch.from_numpy(y)


class LSTMForecaster(nn.Module):
    """Stacked LSTM with a Dense head that emits the full forecast horizon.

    The last layer's final hidden state is fed to a single Linear layer that
    outputs ``output_len`` values in one shot — direct multi-step forecasting.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_len: int = OUTPUT_LEN,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_len, n_features)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                # (batch, hidden_size)
        return self.head(last_hidden)        # (batch, output_len)


def train_lstm(
    train_features: np.ndarray,
    val_features: np.ndarray,
    cfg: TrainConfig = TrainConfig(),
) -> Tuple[LSTMForecaster, dict]:
    """Train an LSTM with early stopping on validation MSE.

    Returns the model in eval mode plus a dict of training history for plots.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_ds = WindowedSeriesDataset(
        train_features, cfg.input_len, cfg.output_len, stride=cfg.train_stride
    )
    val_ds = WindowedSeriesDataset(
        val_features, cfg.input_len, cfg.output_len, stride=cfg.input_len  # disjoint val windows
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    n_features = train_features.shape[1]
    model = LSTMForecaster(
        n_features=n_features,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        output_len=cfg.output_len,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    if cfg.loss_type == "quantile":
        loss_fn: nn.Module = QuantileLoss(tau=cfg.quantile_tau)
    elif cfg.loss_type == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type!r}")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            train_n += x.size(0)
        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                pred = model(x)
                val_loss_sum += loss_fn(pred, y).item() * x.size(0)
                val_n += x.size(0)
        val_loss = val_loss_sum / max(val_n, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch:02d}  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def forecast(
    model: LSTMForecaster,
    input_window: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Produce a full-horizon forecast from a single input window.

    ``input_window`` must be shaped (input_len, n_features) — features in the
    same order as during training.
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(input_window.astype(np.float32)).unsqueeze(0).to(device)
        y = model(x).squeeze(0).cpu().numpy()
    return y


def save_model(model: LSTMForecaster, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_model(
    path: Path,
    n_features: int,
    cfg: TrainConfig = TrainConfig(),
) -> LSTMForecaster:
    model = LSTMForecaster(
        n_features=n_features,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        output_len=cfg.output_len,
        dropout=cfg.dropout,
    )
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    return model


@dataclass
class ScalerStats:
    """Frozen MinMax scaling parameters paired with a saved checkpoint.

    Mirrors the subset of ``sklearn.preprocessing.MinMaxScaler`` fields the
    GUI needs at inference time, plus the feature column ordering so we can
    detect mismatched inputs.
    """

    feature_names: list[str]
    data_min: np.ndarray
    data_max: np.ndarray
    target_col: str
    input_len: int
    output_len: int

    @property
    def data_range(self) -> np.ndarray:
        return self.data_max - self.data_min

    def transform(self, x: np.ndarray) -> np.ndarray:
        rng = self.data_range
        rng = np.where(rng == 0, 1.0, rng)  # avoid divide-by-zero on degenerate cols
        return (x - self.data_min) / rng

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        target_min = float(self.data_min[0])
        target_range = float(self.data_max[0] - self.data_min[0])
        return y_scaled * target_range + target_min


def load_scaler(path: Path) -> ScalerStats:
    """Load scaler stats produced by ``scripts/bake_lstm_scaler.py``."""
    payload = json.loads(Path(path).read_text())
    return ScalerStats(
        feature_names=list(payload["feature_names"]),
        data_min=np.asarray(payload["data_min"], dtype=np.float64),
        data_max=np.asarray(payload["data_max"], dtype=np.float64),
        target_col=payload["target_col"],
        input_len=int(payload["input_len"]),
        output_len=int(payload["output_len"]),
    )
