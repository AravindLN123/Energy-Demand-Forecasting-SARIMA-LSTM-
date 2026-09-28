"""Streamlit GUI for the Energy Demand Forecaster.

A non-technical user can:
  1. Upload a CSV (or load the bundled PJME sample)
  2. Pick a model (SARIMA or LSTM)
  3. Click Run
  4. See live status, the forecast plot, and download the predictions

Architecture: this file imports pure functions from `src/` — no model logic
lives here. Run with: `streamlit run app/streamlit_app.py` from the project root.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import MinMaxScaler

from src.preprocessing import ensure_hourly
from src.feature_engineering import build_lstm_features
from src.sarima_model import (
    DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER,
    fit_sarima,
    forecast as sarima_forecast,
)
from src.lstm_model import (
    INPUT_LEN,
    TrainConfig,
    forecast as lstm_forecast,
    load_model as load_lstm_model,
    load_scaler as load_lstm_scaler,
    train_lstm,
)


SAMPLE_CSV = PROJECT_ROOT / "data" / "raw" / "PJME_hourly.csv"
# The shipped LSTM is the quantile-loss (tau=0.7) variant — MAE 3,307 / RMSE
# 4,156 on the PJME test window, vs 4,995 / 6,022 for the original MSE
# baseline. Symmetric MSE training collapses peaks toward the conditional
# mean; pinball loss with tau > 0.5 penalises under-prediction more heavily
# and recovers them.
LSTM_CHECKPOINT = PROJECT_ROOT / "models" / "lstm_pjme_q70.pt"
LSTM_SCALER = PROJECT_ROOT / "models" / "lstm_pjme.scaler.json"


# ---- Page setup ------------------------------------------------------------

st.set_page_config(
    page_title="Energy Demand Forecaster",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Demand Forecaster")
st.caption("Upload an hourly time-series CSV, choose a model, and forecast.")


# ---- Helpers --------------------------------------------------------------

@st.cache_data(show_spinner=False)
def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def coerce_series(df: pd.DataFrame, dt_col: str, val_col: str) -> pd.Series:
    """Return a clean, hourly-indexed Series of the chosen value column."""
    work = df[[dt_col, val_col]].copy()
    work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
    work = work.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
    if work.index.has_duplicates:
        work = work.groupby(level=0).mean()
    work = work.rename(columns={val_col: "value"})
    return ensure_hourly(work)["value"]


# ---- Sidebar: inputs ------------------------------------------------------

with st.sidebar:
    st.header("Inputs")

    # Persist the active dataset across reruns. st.button only returns True on
    # the click that triggered the rerun, so without session_state the sample
    # would disappear the moment the user clicks anything else.
    if "file_bytes" not in st.session_state:
        st.session_state.file_bytes = None
        st.session_state.source_name = None
        st.session_state.last_upload_id = None

    sample_clicked = st.button("📂 Load PJME sample", use_container_width=True)
    uploaded = st.file_uploader("Or upload your CSV", type=["csv"], key="csv_uploader")

    if sample_clicked and SAMPLE_CSV.exists():
        st.session_state.file_bytes = SAMPLE_CSV.read_bytes()
        st.session_state.source_name = SAMPLE_CSV.name
        st.session_state.last_upload_id = None

    # The uploader widget is sticky across reruns — only treat it as a new
    # source when the file_id changes, otherwise a previously-uploaded file
    # would silently override a later "Load sample" click on every rerun.
    if uploaded is not None and uploaded.file_id != st.session_state.last_upload_id:
        st.session_state.file_bytes = uploaded.getvalue()
        st.session_state.source_name = uploaded.name
        st.session_state.last_upload_id = uploaded.file_id

    file_bytes = st.session_state.file_bytes
    if st.session_state.source_name:
        st.success(f"Active dataset: {st.session_state.source_name}")

    df: pd.DataFrame | None = None
    dt_col = val_col = None
    if file_bytes is not None:
        df = parse_csv(file_bytes)
        cols = list(df.columns)
        dt_default = next(
            (i for i, c in enumerate(cols) if c.lower() in {"datetime", "date", "timestamp", "time"}),
            0,
        )
        val_default = next(
            (i for i, c in enumerate(cols) if c != cols[dt_default] and pd.api.types.is_numeric_dtype(df[c])),
            min(1, len(cols) - 1),
        )
        dt_col = st.selectbox("Datetime column", cols, index=dt_default)
        val_col = st.selectbox("Value column", cols, index=val_default)

    horizon = st.number_input(
        "Forecast horizon (hours)",
        min_value=24,
        max_value=2160,
        value=720,
        step=24,
        help="Same horizon both models are evaluated on.",
    )
    horizon_days = horizon / 24
    days_label = f"{horizon_days:.0f} days" if horizon_days == int(horizon_days) else f"{horizon_days:.1f} days"
    st.caption(f"≈ {days_label}")

    model_choice = st.radio(
        "Model",
        [
            "SARIMA — fast (~1 min)",
            "LSTM — instant on PJME sample, ~3–5 min on uploads",
        ],
        index=0,
    )


    run_clicked = st.button(
        "▶ Run forecast",
        type="primary",
        use_container_width=True,
        disabled=(df is None),
    )

    st.divider()
    with st.expander("📞 Contact Support"):
        st.markdown(
            "Need help with the app or have feedback?\n\n"
            "**Email:** [support@energy-forecaster.example.com]"
            "(mailto:support@energy-forecaster.example.com)  \n"
            "**Phone:** +49 (0) 4921 807-0123  \n"
            "**Hours:** Mon–Fri, 09:00–17:00 CET\n\n"
            "_Contact details shown above are placeholders for demonstration._"
        )


# ---- Main panel ----------------------------------------------------------

status_placeholder = st.container()
viz_placeholder = st.container()
insights_placeholder = st.container()
metrics_placeholder = st.container()


def build_forecast_insights(forecast: pd.Series, history: pd.Series) -> dict:
    """Derive future-looking summary stats from the forecast series.

    Pure pandas — no LLM call. Mirrors the BA25-04 'AI Forecasted Insight'
    panel: peaks, troughs, trend direction, weekday with highest demand.
    """
    daily = forecast.resample("D").mean()

    # Trend: compare the first quarter of the horizon to the last quarter.
    q = max(len(forecast) // 4, 1)
    delta_pct = (forecast.iloc[-q:].mean() - forecast.iloc[:q].mean()) / forecast.iloc[:q].mean() * 100
    if delta_pct > 2:
        trend_label, trend_arrow = "Rising", "↗"
    elif delta_pct < -2:
        trend_label, trend_arrow = "Falling", "↘"
    else:
        trend_label, trend_arrow = "Stable", "→"

    weekday_avg = forecast.groupby(forecast.index.day_name()).mean()
    hour_avg = forecast.groupby(forecast.index.hour).mean()

    hist_mean = history.iloc[-24 * 30:].mean()
    vs_history_pct = (forecast.mean() - hist_mean) / hist_mean * 100

    peak_ts = forecast.idxmax()
    trough_ts = forecast.idxmin()
    peak_day_ts = daily.idxmax()
    trough_day_ts = daily.idxmin()

    return {
        "peak": (peak_ts, float(forecast.max())),
        "trough": (trough_ts, float(forecast.min())),
        "peak_day": (peak_day_ts, float(daily.max())),
        "trough_day": (trough_day_ts, float(daily.min())),
        "trend": (trend_label, trend_arrow, float(delta_pct)),
        "busiest_weekday": (weekday_avg.idxmax(), float(weekday_avg.max())),
        "calmest_weekday": (weekday_avg.idxmin(), float(weekday_avg.min())),
        "peak_hour": (int(hour_avg.idxmax()), float(hour_avg.max())),
        "vs_history_pct": float(vs_history_pct),
        "horizon_days": len(forecast) // 24,
    }

if df is None:
    with status_placeholder:
        st.info(
            "👈 Upload a CSV or click **Load PJME sample** in the sidebar to begin.\n\n"
            "Your CSV needs at least two columns: a datetime column and a numeric value column."
        )
else:
    with status_placeholder:
        st.write(f"**Preview** — {len(df):,} rows, {len(df.columns)} columns")
        st.dataframe(df, use_container_width=True, height=320)


def run_pipeline(series: pd.Series, model_label: str, horizon: int):
    """Execute the end-to-end pipeline with live status updates."""
    with status_placeholder:
        with st.status("Running pipeline…", expanded=True) as status:
            st.write(f"Series length: {len(series):,} hourly observations")
            st.write(f"Range: {series.index.min()} → {series.index.max()}")

            if model_label.startswith("SARIMA"):
                # ---------- SARIMA ----------
                train = series.iloc[-24 * 365 * 2:]  # last 2 years
                if len(train) < 24 * 60:
                    status.update(label="Need at least ~60 days of data for SARIMA", state="error")
                    return None
                st.write(f"Training SARIMA{DEFAULT_ORDER}{DEFAULT_SEASONAL_ORDER} on "
                         f"{len(train):,} rows… this typically takes ~1 minute")
                fit = fit_sarima(train, order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL_ORDER)
                st.write("Forecasting…")
                point, _ = sarima_forecast(fit, steps=horizon)
                future_idx = pd.date_range(
                    series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
                )
                forecast_series = pd.Series(point.values, index=future_idx, name="forecast")

            else:
                # ---------- LSTM ----------
                st.write("Engineering features (cyclical + lag)…")
                features = build_lstm_features(series.to_frame("value"), target="value")

                # Prefer the pre-trained PJME checkpoint when the user is on
                # the bundled sample and asking for the canonical 720h horizon
                # — instant, and matches the documented MAE = 3,307 quality.
                use_pretrained = (
                    st.session_state.get("source_name") == SAMPLE_CSV.name
                    and horizon == 720
                    and LSTM_CHECKPOINT.exists()
                    and LSTM_SCALER.exists()
                    and len(features) >= INPUT_LEN
                )

                if use_pretrained:
                    st.write("Loading pre-trained PJME checkpoint (quantile τ=0.7) + scaler…")
                    scaler_stats = load_lstm_scaler(LSTM_SCALER)
                    cfg = TrainConfig()
                    model = load_lstm_model(
                        LSTM_CHECKPOINT,
                        n_features=len(scaler_stats.feature_names),
                        cfg=cfg,
                    )

                    input_window_df = features.iloc[-scaler_stats.input_len:]
                    input_arr = scaler_stats.transform(input_window_df.values)

                    st.write("Forecasting…")
                    scaled_pred = lstm_forecast(model, input_arr, device="cpu")
                    pred = scaler_stats.inverse_transform_target(scaled_pred)

                else:
                    cfg = TrainConfig()
                    cfg.max_epochs = 20
                    cfg.train_stride = 6

                    min_required = cfg.input_len + cfg.output_len + 24 * 30 * 6
                    if len(features) < min_required:
                        status.update(
                            label=f"Need at least {min_required:,} hourly rows after feature "
                                  f"engineering; got {len(features):,}",
                            state="error",
                        )
                        return None

                    val_hours = 24 * 30 * 6
                    train_hours = min(24 * 365 * 10, len(features) - val_hours - cfg.input_len)
                    train_df = features.iloc[-(val_hours + train_hours + cfg.input_len):-(val_hours + cfg.input_len)]
                    val_df = features.iloc[-(val_hours + cfg.input_len):-cfg.input_len]
                    input_window_df = features.iloc[-cfg.input_len:]

                    st.write(f"Train window: {len(train_df):,} rows | Val: {len(val_df):,} rows")
                    st.write("Scaling features (fit on train only)…")
                    scaler = MinMaxScaler().fit(train_df.values)
                    train_arr = scaler.transform(train_df.values)
                    val_arr = scaler.transform(val_df.values)
                    input_arr = scaler.transform(input_window_df.values)
                    target_min = scaler.data_min_[0]
                    target_range = scaler.data_max_[0] - target_min

                    cfg.output_len = horizon
                    st.write(f"Training LSTM on {cfg.device.upper()} for up to {cfg.max_epochs} epochs…")
                    model, history = train_lstm(train_arr, val_arr, cfg)
                    st.write(f"Trained {len(history['train_loss'])} epochs (early stop honored)")

                    st.write("Forecasting…")
                    scaled_pred = lstm_forecast(model, input_arr, device=cfg.device)
                    pred = scaled_pred * target_range + target_min

                future_idx = pd.date_range(
                    series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
                )
                forecast_series = pd.Series(pred, index=future_idx, name="forecast")

            status.update(label="✅ Forecast complete", state="complete")
            return forecast_series


if run_clicked and df is not None and dt_col and val_col:
    try:
        series = coerce_series(df, dt_col, val_col)
    except Exception as exc:
        with status_placeholder:
            st.error(f"Could not parse the selected columns: {exc}")
        st.stop()

    forecast = run_pipeline(series, model_choice, int(horizon))

    if forecast is not None:
        # ---- Visualization ----
        with viz_placeholder:
            st.subheader("Forecast")

            history_tail = series.iloc[-24 * 30:]  # last 30 days for context
            chart_df = pd.concat(
                [
                    history_tail.rename("history"),
                    forecast.rename("forecast"),
                ],
                axis=1,
            )
            st.line_chart(chart_df, use_container_width=True, height=400)

            with st.expander("Forecast values"):
                st.dataframe(forecast.to_frame(), use_container_width=True)

        # ---- AI Forecasted Insight Panel ----
        with insights_placeholder:
            st.subheader("🔮 AI Forecasted Insights")
            st.caption(
                f"Future-looking commentary on the next {len(forecast):,} hours, "
                f"derived directly from the model's predictions."
            )
            ins = build_forecast_insights(forecast, series)

            row1 = st.columns(4)
            row1[0].metric(
                "Highest forecasted hour",
                f"{ins['peak'][1]:,.0f} MW",
                ins["peak"][0].strftime("%a %d %b, %H:%M"),
            )
            row1[1].metric(
                "Lowest forecasted hour",
                f"{ins['trough'][1]:,.0f} MW",
                ins["trough"][0].strftime("%a %d %b, %H:%M"),
            )
            row1[2].metric(
                "Peak demand day",
                ins["peak_day"][0].strftime("%a %d %b"),
                f"{ins['peak_day'][1]:,.0f} MW avg",
            )
            row1[3].metric(
                "Calmest demand day",
                ins["trough_day"][0].strftime("%a %d %b"),
                f"{ins['trough_day'][1]:,.0f} MW avg",
            )

            row2 = st.columns(4)
            row2[0].metric(
                "Overall trend",
                f"{ins['trend'][1]} {ins['trend'][0]}",
                f"{ins['trend'][2]:+.1f}% start → end",
            )
            row2[1].metric(
                "Busiest weekday",
                ins["busiest_weekday"][0],
                f"{ins['busiest_weekday'][1]:,.0f} MW avg",
            )
            row2[2].metric(
                "Typical peak hour",
                f"{ins['peak_hour'][0]:02d}:00",
                f"{ins['peak_hour'][1]:,.0f} MW avg",
            )
            row2[3].metric(
                "vs. last 30 days",
                f"{ins['vs_history_pct']:+.1f}%",
                "higher" if ins["vs_history_pct"] >= 0 else "lower",
                delta_color="off",
            )

            trend_label = ins["trend"][0].lower()
            st.info(
                f"**Outlook:** Demand is expected to be **{trend_label}** over the next "
                f"{ins['horizon_days']} days, peaking on "
                f"**{ins['peak_day'][0].strftime('%A %d %B')}** at around "
                f"**{ins['peak'][1]:,.0f} MW**. "
                f"**{ins['busiest_weekday'][0]}s** are typically the busiest weekday, "
                f"with daily demand cresting near **{ins['peak_hour'][0]:02d}:00**. "
                f"Average load is **{ins['vs_history_pct']:+.1f}%** "
                f"versus the last 30 days of observed history."
            )

        # ---- Summary stats ----
        with metrics_placeholder:
            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast horizon", f"{len(forecast):,} hours")
            c2.metric("Mean forecast", f"{forecast.mean():,.0f}")
            c3.metric("Peak forecast", f"{forecast.max():,.0f}")

            csv_bytes = forecast.to_frame().to_csv().encode()
            st.download_button(
                "⬇ Download forecast CSV",
                data=csv_bytes,
                file_name="forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---- Footer bar ---------------------------------------------------------

# The footer markup lives in the parent document (st.markdown) so the fixed
# positioning works against the app viewport. The clock-driver script is
# injected via st.components.v1.html (an iframe) and reaches up to the
# parent doc with window.parent.document — st.markdown strips <script> tags.
st.markdown(
    """
    <style>
      .block-container { padding-bottom: 5rem; }
      #edf-footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background: rgba(20, 24, 38, 0.92); color: #f0f3fa;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 13px; padding: 10px 24px;
        display: flex; justify-content: space-between; align-items: center;
        border-top: 1px solid rgba(255,255,255,0.08);
        z-index: 9999;
      }
      #edf-clock { font-variant-numeric: tabular-nums; }
    </style>
    <div id="edf-footer">
      <span>© 2026 Hochschule Emden-Leer | Energy Demand Forecaster</span>
      <span id="edf-clock">—</span>
    </div>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <script>
      (function () {
        const root = window.parent ? window.parent.document : document;
        function tick() {
          const el = root.getElementById('edf-clock');
          if (!el) return;
          const now = new Date();
          el.textContent = now.toLocaleString(undefined, {
            weekday: 'short', year: 'numeric', month: 'short', day: '2-digit',
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
          });
        }
        tick();
        setInterval(tick, 1000);
      })();
    </script>
    """,
    height=0,
)
