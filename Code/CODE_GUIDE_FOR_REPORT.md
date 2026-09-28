# Code Guide for the Report

---

## 1. Quick orientation — where things live

```
Code/Python/EnergyDemandForecaster/
├── src/                        ← reusable functions (the "library")
│   ├── data_loader.py          load + parse the CSV
│   ├── preprocessing.py        clean, resample, train/test split
│   ├── feature_engineering.py  cyclical + lag features (LSTM only)
│   ├── sarima_model.py         fit + forecast SARIMA
│   ├── lstm_model.py           PyTorch LSTM: model, training loop, forecast
│   └── evaluation.py           MAE / RMSE / plots / Ljung-Box
├── notebooks/                  ← step-by-step walkthroughs (run order is in filename)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_sarima_model.ipynb
│   └── 04_lstm_model.ipynb
├── app/streamlit_app.py        ← the GUI (uses src/ functions)
├── results/figures/            ← all plots used in the report (PNG)
├── results/metrics.json        ← MAE / RMSE numbers
└── data/raw/PJME_hourly.csv    ← the dataset
```

**Rule of thumb:** if the report mentions a *function* or an *algorithm step*, look in `src/`. If it mentions a *figure* or an *experiment result*, look in `notebooks/` or `results/`.

---

## 2. Report section → code map

The current report skeleton is in [`report/Contents/General/DomainMachineLearning.tex`](../../../report/Contents/General/DomainMachineLearning.tex). Each row below tells you which file/lines to use as your source material.

### Packages chapter

| Report subsection | What we actually use | Code reference |
|---|---|---|
| pandas — Usage / Example Code | `pd.read_csv`, `to_datetime`, `groupby`, `asfreq`, `interpolate` | [`src/data_loader.py:9-35`](src/data_loader.py), [`src/preprocessing.py:8-20`](src/preprocessing.py) |
| NumPy — Usage / Example Code | array math underneath everything; explicit use in scaling for LSTM | [`src/lstm_model.py:113-195`](src/lstm_model.py) (look for `np.` calls) |
| matplotlib — Usage / Example Code | all plots in `results/figures/` are matplotlib | [`src/evaluation.py:23-77`](src/evaluation.py) |
| scikit-learn — Usage / Example Code | `MinMaxScaler` (LSTM), train/test pattern (we do this manually for time series) | [`src/lstm_model.py`](src/lstm_model.py) for scaler, [`src/preprocessing.py:22-41`](src/preprocessing.py) for split |
| statsmodels — Usage / Example Code | `SARIMAX` model + `seasonal_decompose` + ADF + Ljung-Box | [`src/sarima_model.py:13-31`](src/sarima_model.py), [`src/evaluation.py:55-77`](src/evaluation.py) |


### Data Mining Algorithm chapter — SARIMA

| Report subsection | Source in code |
|---|---|
| 1. Description | Conceptual — see *Further reading* at the end of this doc |
| 2. Application | We forecast 720 hours of PJME hourly load — see [`notebooks/03_sarima_model.ipynb`](notebooks/03_sarima_model.ipynb) |
| Relevance to Project | The classical statistical baseline against which LSTM is compared |
| Requirements | Stationarity (after differencing) — proven in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (ADF test cell) |
| Hyperparameters | `order=(1,1,1)`, `seasonal_order=(1,1,1,24)` — defined in [`src/sarima_model.py:13-31`](src/sarima_model.py) |
| Input | Univariate hourly series, `pd.Series` indexed by datetime |
| Output | Point forecast `pd.Series` (length 720) + 95% confidence interval `pd.DataFrame` |
| 7. Sample Code | [`src/sarima_model.py:13-42`](src/sarima_model.py) — see snippet below |
| 8. SARIMA Forecasting Pipeline | [`notebooks/03_sarima_model.ipynb`](notebooks/03_sarima_model.ipynb) — top-to-bottom run is the pipeline |

### Data Mining Algorithm chapter — LSTM

| Report subsection | Source in code |
|---|---|
| 1. Description | Conceptual — see *Further reading* at the end of this doc |
| 2. Application | Same 720-hour PJME forecast, directly comparable to SARIMA |
| 3. Relevance to Project | The "modern" sequence model; we expect it to beat SARIMA on accuracy |
| 4. Requirements | PyTorch ≥ 2.5, Python ≥ 3.10 — see [`requirements.txt`](requirements.txt) |
| 5. Hyperparameters | All in `TrainConfig` — [`src/lstm_model.py:25-42`](src/lstm_model.py) |
| 6. Input | Sliding window of 168 past hours × `n_features` (`PJME_MW` + sin/cos + lags) |
| Output | Vector of 720 predicted hours (one forward pass — direct multi-step) |
| 7. Sample Code | [`src/lstm_model.py:81-111`](src/lstm_model.py) (model class), [`src/lstm_model.py:113-195`](src/lstm_model.py) (training loop) — see snippet below |
| 8. LSTM Forecasting Workflow | [`notebooks/04_lstm_model.ipynb`](notebooks/04_lstm_model.ipynb) |
| 9. Further Reading for LSTM | See last section of this doc |

---

## 3. Ready-to-paste LaTeX snippets

These are the minimum-viable code blocks for the report. Copy as-is and adjust the wording around them.

### Setting up `lstlisting` for Python (once, in the preamble)

If the report does not already have it, add this to the LaTeX preamble (your supervisor's template likely already does):

```latex
\usepackage{listings}
\usepackage{xcolor}
\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{gray}\itshape,
    stringstyle=\color{teal},
    numbers=left, numberstyle=\tiny\color{gray},
    breaklines=true,
    frame=single,
    captionpos=b
}
```

### SARIMA model — for the "7. Sample Code" subsection

```latex
\begin{lstlisting}[caption={Fitting the SARIMA model. Source: \texttt{src/sarima\_model.py}}]
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)

def forecast(fit, steps=720):
    result = fit.get_forecast(steps=steps)
    return result.predicted_mean, result.conf_int()
\end{lstlisting}
```

### LSTM model — for the "7. Sample Code" subsection

```latex
\begin{lstlisting}[caption={LSTM architecture (PyTorch). Source: \texttt{src/lstm\_model.py}}]
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2,
                 output_len=720, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])
\end{lstlisting}
```

### pandas example — for the "Example Code" subsection of pandas

```latex
\begin{lstlisting}[caption={Loading and cleaning the PJME dataset. Source: \texttt{src/data\_loader.py}, \texttt{src/preprocessing.py}}]
import pandas as pd

df = pd.read_csv("data/raw/PJME_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.groupby("Datetime").mean().sort_index()   # dedupe DST hours
df = df.asfreq("H")                                # regular hourly grid
df["PJME_MW"] = df["PJME_MW"].interpolate()        # fill gaps
\end{lstlisting}
```

### matplotlib example — for the "Example Code" subsection of matplotlib

```latex
\begin{lstlisting}[caption={Forecast plot. Source: \texttt{src/evaluation.py}}]
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(train_tail.index, train_tail.values, label="History", color="steelblue")
ax.plot(test.index,       test.values,       label="Actual",  color="black")
ax.plot(forecast.index,   forecast.values,   label="Forecast", color="crimson")
ax.legend(); ax.set_xlabel("Time"); ax.set_ylabel("Load (MW)")
fig.tight_layout()
\end{lstlisting}
```

### Including a figure from `results/figures/`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{Bilder/sarima_forecast.png}
    \caption{SARIMA 720-hour forecast versus actual PJME load.}
    \label{fig:sarima_forecast}
\end{figure}
```

> The figures live under `results/figures/` in the code folder. Either copy the PNG you need into `report/Bilder/` or reference it with a relative path. Ask Karrar which figures are "report-final" vs draft.

---

## 4. Numbers you can cite (so you don't have to re-run anything)

These are the headline results — already produced and stored. **Don't recompute them; just cite.**

| Metric | SARIMA | LSTM |
|---|---|---|
| MAE  (MW) | **11,661** | **4,995** |
| RMSE (MW) | ~13,800 | ~6,200 |
| Forecast horizon | 720 hours (30 days) | 720 hours (30 days) |
| Test window | Final 720 h of PJME | Final 720 h of PJME |
| Train window | Last 2 years (~17.5 k rows) | Last 10 years (~87.6 k rows) |

Source of truth: [`results/metrics.json`](results/metrics.json). If you cite a number, double-check it matches that file — the JSON is updated whenever the model is re-run.

LSTM beats SARIMA by **~57 % MAE reduction**. Honest framing for the report:
- LSTM wins on overall accuracy.
- LSTM systematically **under-predicts peak loads** (documented limitation — discuss in the conclusion).
- SARIMA's win: simpler, faster, fully interpretable parameters.