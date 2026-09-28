 Energy Demand Forecaster

![Logo](./report/General/Logo.png "Project Logo")

## Project Overview

This project explores **electricity load forecasting** using hourly time series data and compares a classical statistical baseline with a deep-learning approach. The goal is to generate a **30-day forecast horizon (720 hours)** and evaluate whether an **LSTM** model can outperform a traditional **SARIMA** model for medium-term demand forecasting.

The work is documented as an academic term paper following the **Knowledge Discovery in Databases (KDD)** process. It covers the full analytical workflow from domain understanding and preprocessing to feature engineering, model development, evaluation, deployment, and monitoring. A fully working **Streamlit application** and a **portable offline bundle** (macOS / Windows) ship alongside the report.


Use the relative paths documented below so collaborators on any operating system can locate the same files consistently.

## Project Snapshot

| Item | Details |
|------|---------|
| Domain | Electricity load forecasting |
| Data frequency | Hourly |
| Forecast horizon | 30 days / 720 hours |
| Models compared | SARIMA, LSTM (MSE), LSTM-q70 |
| Best model | **LSTM-q70** — MAE **3 307 MW**, **−72 %** vs SARIMA |
| Evaluation metrics | MAE and RMSE |
| Dataset source | PJM Hourly Energy Consumption via Kaggle |
| Deployment | Streamlit GUI + portable bundle (macOS / Windows) |

## Headline Results

Evaluated on the final **720-hour** test window of the PJME series.

| Model | MAE (MW) | RMSE (MW) | vs SARIMA |
|-------|---------:|----------:|----------:|
| SARIMA(1,1,1)(1,1,1)₂₄ | 11 662 | 12 932 | — |
| LSTM (MSE loss) | 4 996 | 6 023 | −57 % |
| **LSTM-q70** (τ = 0.70) | **3 307** | **4 156** | **−72 %** |

The quantile / pinball loss at τ = 0.70 reduces peak under-prediction by a further −34 % MAE compared with the MSE-trained LSTM — a meaningful gain for grid operators, who pay more for under-prediction than for over-prediction.

## Problem Statement

Electricity providers and grid operators must continuously balance energy supply and demand. Poor forecasts lead to inefficient generation planning, higher operational costs, and increased reliance on backup power.

Forecasting electricity demand is challenging because the data is:

- Seasonal across multiple time scales (daily, weekly, annual)
- Non-linear and non-stationary
- Affected by calendar effects and external conditions
- Sensitive to missing values, anomalies, and daylight-saving-time shifts

## Methodology

The project is organised around the **KDD process**:

1. **Problem understanding** — define the forecasting task and its business relevance.
2. **Data understanding** — study the PJME load dataset, its quality and structure.
3. **Data preparation** — clean the data, handle DST gaps, and engineer time features.
4. **Data mining** — train SARIMA (statistical baseline) and LSTM (sequence model).
5. **Evaluation** — measure forecasting quality using MAE and RMSE.
6. **Deployment & monitoring** — Streamlit GUI, portable bundle, drift considerations.

## Dataset

- **Name:** Hourly Energy Consumption (PJM Interconnection)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- **Unit:** Megawatts (MW)
- **Coverage:** Multi-year hourly load history suitable for time-series forecasting

## Repository Structure

```text
.
├── README.md
├── Code/
│   ├── Python/EnergyDemandForecaster/   # Forecasting library, scripts, GUI
│   │   ├── src/                         # data_loader, preprocessing, features,
│   │   │                                  sarima_model, lstm_model, evaluation
│   │   ├── scripts/                     # train_lstm_quantile, bake_scaler,
│   │   │                                  regenerate_lstm_figures, …
│   │   ├── app/                         # streamlit_app.py
│   │   ├── notebooks/                   # 01_eda → 04_lstm_model
│   │   ├── data/, models/, results/     # (gitignored generated artefacts)
│   │   ├── dist/                        # portable bundles (gitignored)
│   │   ├── requirements.txt
│   │   └── build_portable.sh
│   └── MicroPython/                     # Small hardware demo (HelloWorld/Blink)
├── Doxygen/                             # Code reference (HTML)  — see below
│   ├── Doxyfile                         # Doxygen configuration
│   ├── mainpage.md                      # Landing page content
│   ├── custom.css                       # Professional theme overrides
│   ├── README.md                        # Build instructions
│   └── html/                            # Pre-built HTML (open index.html)
├── report/                              # LaTeX report
│   ├── Contents/General/                # Main report chapters
│   ├── General/                         # Shared LaTeX config and assets
│   ├── Images/                          # Figures
│   ├── System/EdgeComputer/             # Main LaTeX entry point + compiled PDF
│   └── tikz/                            # TikZ drawings
├── Presentations/                       # Beamer presentation (4-part)
├── Poster/                              # Poster sources and PDF
├── Manual/                              # End-user manual
├── ProjectManagement/                   # Planning / Gantt / minutes
├── MLbib/                               # Shared bibliography
└── author.xlsx                          # Author / project metadata
```

## Main Report File

The LaTeX entry point for the full written report is `report/System/EdgeComputer/EnergyDemandForecaster.tex`.

To compile from a fresh clone:

```bash
git clone https://github.com/Wings-hub/BA26-01-Time-Series.git
cd BA26-01-Time-Series/report/System/EdgeComputer
pdflatex EnergyDemandForecaster.tex
biber    EnergyDemandForecaster
pdflatex EnergyDemandForecaster.tex
pdflatex EnergyDemandForecaster.tex
```

If your environment supports `latexmk`, an equivalent automated workflow can be used.

## Report Structure

- **Introduction** — background, problem, challenges, proposed solution, objectives
- **Domain Knowledge** — electricity forecasting context and data relevance
- **Machine Learning** — packages, concepts, SARIMA and LSTM algorithms
- **Methodology** — application of the KDD process
- **Development** — data loading, transformation, feature engineering, training
- **Development to Deployment** — moving from experimentation to a reusable workflow
- **Deployment** — Streamlit GUI and portable bundle
- **Monitoring** — tracking model quality and drift
- **Evaluation & Conclusion** — performance, findings, limitations, next steps

## Python Code

The complete forecasting implementation lives in [Code/Python/EnergyDemandForecaster/](Code/Python/EnergyDemandForecaster/).

Quick start (development environment):

```bash
cd Code/Python/EnergyDemandForecaster
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

End users do **not** need Python — a portable, pre-built bundle is produced by `build_portable.sh` (macOS) and shipped to `dist/portable/`. Users double-click `run.command` (macOS) or `run.bat` (Windows) and the GUI launches with the pre-trained `lstm_pjme_q70.pt` checkpoint already loaded.

## Code Documentation (Doxygen)

A **Doxygen-generated HTML reference** of the Python codebase is provided so the source can be browsed by **module, class, function, and file** — with a search box — without installing any tools.

- **Open directly:** [Doxygen/html/index.html](Doxygen/html/index.html)
- **Configuration:** [Doxygen/Doxyfile](Doxygen/Doxyfile)
- **Landing page source:** [Doxygen/mainpage.md](Doxygen/mainpage.md)
- **Build instructions:** [Doxygen/README.md](Doxygen/README.md)

To regenerate after changing the Python source:

```bash
cd Doxygen
doxygen Doxyfile
open html/index.html        # macOS
```

The Doxyfile indexes `Code/Python/EnergyDemandForecaster/{src,scripts,app}` and excludes virtual-env, build, and data folders.

## Tech Stack

- **Languages:** Python 3.12, LaTeX
- **Forecasting libraries:** statsmodels (SARIMA), PyTorch (LSTM), scikit-learn
- **Application:** Streamlit
- **Documentation:** LaTeX + BibLaTeX/Biber, Doxygen (HTML reference)
- **Process framework:** KDD
- **Version control:** Git and GitHub

## Authors

- Karrar Al-Ameeri
- Aravind Lakshmi Narayanan
- Abhilash Nayak
- Pranoti Patil

## Academic Context

- **Course:** Business Intelligence and Data Analytics
- **Institution:** University of Applied Sciences Emden/Leer
