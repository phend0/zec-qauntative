# zec-qauntative

**ZCash Quantitative Trading Pipeline** — Version 0.0.1

A modular, production-grade quantitative machine learning pipeline designed to predict short-to-medium-term price movement direction for Zcash (`ZEC-USD`) using macroeconomic and cross-asset signals from Bitcoin (`BTC-USD`) and Ethereum (`ETH-USD`).

---

## Architecture Overview

```text
zec-qauntative/
├── config.py             # Type-hinted @dataclass objects for pipeline parameters
├── data_loader.py        # Resilient multi-provider ingestion (CMC, CCXT, yfinance) & UTC alignment
├── features.py           # Feature engineering, cross-asset metrics & ADF stationarity testing
├── target.py             # Volatility-adjusted Triple-Barrier Method labeling
├── validation.py         # Purged and Embargoed Walk-Forward Cross-Validation
├── models/
│   ├── __init__.py
│   ├── base.py           # Model base abstraction
│   ├── tabular.py        # Class-weighted Logistic Regression & LightGBM with Optuna/SHAP
│   └── sequence.py       # PyTorch GRU/LSTM with 3D temporal arrays & early stopping
├── backtest.py           # Vectorized backtesting engine with realistic crypto friction
├── main.py               # End-to-end execution pipeline
├── requirements.txt      # Project dependencies
└── README.md
```

---

## Features

- **Multi-Source Resilient Ingestion (`data_loader.py`)**:
  - Primary: CoinMarketCap API (`/v2/cryptocurrency/ohlcv/historical`) for ZEC (ID: 1437).
  - Graceful Fallback: CCXT Binance public OHLCV and `yfinance`.
  - Timestamp Alignment: Synchronized to unified UTC frequency index with bounded forward-filling ($\le 2$ intervals) and edge-dropping to avoid synthetic lookahead.

- **Stationary Feature Engineering (`features.py`)**:
  - Log Returns: $r_t = \ln(P_t / P_{t-1})$.
  - Technical Indicators: RSI-14, Stochastic Oscillator, MACD (12-26-9), EMA ratios, ATR-14, Bollinger Band width, and rolling 7d/30d annualized volatility.
  - Cross-Asset Signals: Rolling 30-day Beta, rolling correlation, and return differentials against BTC and ETH.
  - Microstructure: Rolling VWAP approximation and normalized volume ratios ($\text{Vol}_t / \text{SMA}_{20}(\text{Vol})$).
  - Automated Stationarity Check: Augmented Dickey-Fuller (ADF) test prior to model ingestion.

- **Volatility-Adjusted Target Formulation (`target.py`)**:
  - Triple-Barrier Method adapting dynamically to market regimes:
    - Upper Barrier: $P_t + 1.5 \times \text{ATR}_{14}$
    - Lower Barrier: $P_t - 1.5 \times \text{ATR}_{14}$
    - Vertical Barrier: Fixed time horizon ($t + 24$ hours).
    - Labels: `+1` (Upper hit first), `-1` (Lower hit first), `0` (Neutral / Vertical expiration).
  - Strict forward-shift validation ensuring zero lookahead bias.

- **Leakage-Proof Validation (`validation.py`)**:
  - Purged and Embargoed Walk-Forward Cross-Validation (López de Prado methodology).
  - Embargo buffer separating train and test folds to eliminate overlap leakage.

- **Multi-Model Suite (`models/`)**:
  - **Baseline**: Class-weighted Logistic Regression with L2 regularization and `RobustScaler`.
  - **Ensemble**: LightGBM with Optuna hyperparameter tuning hooks and TreeSHAP explainability.
  - **Deep Sequence**: PyTorch GRU/LSTM accepting 3D tensors `(samples, window_size, n_features)` with custom `TimeSeriesDataset`, dropout regularization, and early stopping.

- **Realistic Vectorized Backtesting (`backtest.py`)**:
  - Converts model signals into discrete positions: Long (`+1`), Cash (`0`), Short (`-1`).
  - Realistic Crypto Friction:
    - Taker fee: 10 bps (0.10%) per trade.
    - Slippage: 5 bps (0.05%) per trade.
  - Evaluates Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD), Calmar Ratio, Win Rate, and Profit Factor against Buy & Hold.

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/zec-qauntative.git
cd zec-qauntative

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables (Optional)

Configure your API keys if using CoinMarketCap or authenticated exchange endpoints:

```bash
export CMC_API_KEY="your-coinmarketcap-key"
export BINANCE_API_KEY="your-binance-key"
export BINANCE_SECRET="your-binance-secret"
export KRAKEN_API_KEY="your-kraken-key"
export KRAKEN_SECRET="your-kraken-secret"
```

---

## Usage

Run the end-to-end pipeline:

```bash
python main.py
```

---

## Versioning

Current version: **0.0.1**
