# Trading Hidden Markov

Regime-based trading system using a 7-state Gaussian Hidden Markov Model.

## Features

- **HMM Regime Detection** — Identifies Bull Run, Bear/Crash, and Neutral market regimes
- **8-Condition Voting System** — RSI, Momentum, Volatility, Volume, ADX, EMA50, EMA200, MACD
- **Interactive Dashboard** — Streamlit app with candlestick charts, equity curves, regime distribution
- **Multi-Asset** — Works with any Yahoo Finance ticker (stocks, crypto, ETFs, forex, indices)
- **Backtesting Engine** — Full simulation with trade log, win rate, max drawdown, alpha vs buy & hold

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Strategy Rules

| # | Entry Condition |
|---|-----------------|
| 1 | RSI < 90 |
| 2 | Momentum (20-bar) > 1% |
| 3 | Volatility (20-bar σ) < 6% |
| 4 | Volume > 20-bar SMA |
| 5 | ADX > 25 |
| 6 | Price > EMA 50 |
| 7 | Price > EMA 200 |
| 8 | MACD > Signal |

- **Entry:** Bull Run regime + ≥ 7/8 confirmations
- **Exit:** Regime flips to Bear/Crash
- **Cooldown:** 48h after every exit

## Architecture

- `app.py` — Streamlit dashboard
- `backtester.py` — HMM + indicators + backtest engine
- `data_loader.py` — yfinance data fetcher with disk cache

---
*For educational purposes only — not financial advice.*
