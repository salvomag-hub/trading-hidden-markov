"""
backtester.py
─────────────
HMM regime detection + technical indicators + voting system + backtest engine.

Architecture
────────────
1. build_hmm_features()   – 3-feature matrix for the HMM
2. train_hmm()            – GaussianHMM with multiple random restarts
3. add_indicators()       – RSI, Momentum, Volatility, ADX, EMAs, MACD
4. add_votes()            – vectorised 8-condition vote counter
5. run_backtest()         – sequential simulation with 2.5× leverage,
                            48-h cooldown, and Bear/Crash exit rule
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
N_STATES = 7
INITIAL_CAPITAL = 10_000.0
LEVERAGE = 1.0
COOLDOWN_HOURS = 48
MIN_VOTES = 7


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    capital_at_entry: float
    pnl: float
    return_pct: float
    exit_reason: str
    votes: int


@dataclass
class BacktestResult:
    trades: list[Trade]
    portfolio_values: pd.Series
    df: pd.DataFrame          # enriched with Regime, indicators, Votes
    bull_state: int
    bear_state: int
    total_return: float
    buy_hold_return: float
    alpha: float
    win_rate: float
    max_drawdown: float
    n_trades: int
    current_signal: str       # "LONG" | "CASH"
    current_regime: str       # "Bull Run" | "Bear/Crash" | "Neutral"


# ── Technical indicators (pure-pandas, no external TA lib) ────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return macd_line, signal_line


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder-smoothed Average Directional Index."""
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=period).mean() / (atr + 1e-10)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=period).mean() / (atr + 1e-10)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=alpha, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* enriched with all strategy indicators."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    out["RSI"] = _rsi(close)
    out["Momentum"] = close.pct_change(20)                          # 20-bar return
    out["Volatility"] = close.pct_change().rolling(20).std()        # 20-bar σ of returns
    out["Volume_SMA"] = volume.rolling(20).mean()
    out["ADX"] = _adx(high, low, close)
    out["EMA_50"] = close.ewm(span=50, min_periods=50).mean()
    out["EMA_200"] = close.ewm(span=200, min_periods=200).mean()
    out["MACD"], out["MACD_Signal"] = _macd(close)

    return out


# ── HMM engine ────────────────────────────────────────────────────────────────
def build_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 3-column feature matrix used to train the HMM:
      1. Returns           – hourly close-to-close %
      2. Range             – (High - Low) / Close
      3. Volume Volatility – rolling-20 σ of volume % changes
    """
    returns = df["Close"].pct_change()
    price_range = (df["High"] - df["Low"]) / df["Close"]
    vol_vol = df["Volume"].pct_change().rolling(20).std()

    features = pd.DataFrame(
        {"returns": returns, "range": price_range, "vol_volatility": vol_vol}
    ).dropna()

    return features


def train_hmm(
    features_df: pd.DataFrame,
    n_components: int = N_STATES,
    n_trials: int = 5,
) -> tuple:
    """
    Fit a GaussianHMM with *n_trials* random seeds; keep the best log-likelihood.

    Returns:
        (model, scaler, bull_state_index, bear_state_index)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values)

    best_score = -np.inf
    best_model: hmm.GaussianHMM | None = None

    for seed in range(n_trials):
        try:
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=200,
                tol=1e-4,
                random_state=seed,
            )
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as exc:
            logger.warning("HMM trial %d failed: %s", seed, exc)

    if best_model is None:
        raise RuntimeError("All HMM training trials failed.")

    # Identify Bull (highest mean return) and Bear (lowest mean return) states
    # Feature 0 is 'returns' in the scaled space; sign is preserved by StandardScaler
    state_return_means = best_model.means_[:, 0]
    bull_state = int(np.argmax(state_return_means))
    bear_state = int(np.argmin(state_return_means))

    logger.info(
        "HMM trained – bull_state=%d (mean=%.4f) bear_state=%d (mean=%.4f)",
        bull_state,
        state_return_means[bull_state],
        bear_state,
        state_return_means[bear_state],
    )
    return best_model, scaler, bull_state, bear_state


def label_regimes(
    df: pd.DataFrame,
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    bull_state: int,
    bear_state: int,
    features_df: pd.DataFrame,
) -> pd.Series:
    """
    Predict the HMM state for every row that has features, then reindex
    back onto *df* (back-filling the initial NaN rows).
    """
    X = scaler.transform(features_df.values)
    states = model.predict(X)

    regime_map = {bull_state: "Bull Run", bear_state: "Bear/Crash"}
    labels = pd.Series(
        [regime_map.get(int(s), "Neutral") for s in states],
        index=features_df.index,
        name="Regime",
    )

    # Reindex to full df; bfill fills early NaNs with the first known label
    return labels.reindex(df.index).bfill().fillna("Neutral")


# ── Voting system (vectorised) ────────────────────────────────────────────────
def add_votes(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series with the number of the 8 entry conditions met per row.
    NaN comparisons evaluate to False (condition not met), which is correct.
    """
    votes = (
        (df["RSI"] < 90).astype("Int8")
        + (df["Momentum"] > 0.01).astype("Int8")        # > 1 %
        + (df["Volatility"] < 0.06).astype("Int8")      # < 6 %
        + (df["Volume"] > df["Volume_SMA"]).astype("Int8")
        + (df["ADX"] > 25).astype("Int8")
        + (df["Close"] > df["EMA_50"]).astype("Int8")
        + (df["Close"] > df["EMA_200"]).astype("Int8")
        + (df["MACD"] > df["MACD_Signal"]).astype("Int8")
    )
    return votes.fillna(0).astype(int)


# ── Backtest engine ───────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> BacktestResult:
    """
    Run the full regime-based backtest simulation.

    Rules
    ─────
    • Enter LONG when:
        – current regime == "Bull Run"
        – votes >= MIN_VOTES (7 out of 8)
        – not inside the 48-hour cooldown window
    • Exit when regime flips to "Bear/Crash"
    • 48-hour hard cooldown after every exit
    • No leverage applied (1×)
    • Starting capital: $10 000
    """
    # ── Build features + train HMM ────────────────────────────────────────────
    features_df = build_hmm_features(df)
    model, scaler, bull_state, bear_state = train_hmm(features_df)

    # ── Enrich DataFrame ──────────────────────────────────────────────────────
    enriched = add_indicators(df)
    enriched["Regime"] = label_regimes(
        enriched, model, scaler, bull_state, bear_state, features_df
    )
    enriched["Votes"] = add_votes(enriched)

    # ── Sequential simulation ─────────────────────────────────────────────────
    capital = INITIAL_CAPITAL
    position: dict | None = None          # {entry_price, entry_time, capital, votes}
    cooldown_until: pd.Timestamp | None = None
    trades: list[Trade] = []
    portfolio_values: list[float] = []

    for idx, row in enriched.iterrows():
        regime = row["Regime"]

        # ── Manage open position: exit on Bear/Crash ──────────────────────────
        if position is not None and regime == "Bear/Crash":
            exit_price = float(row["Close"])
            lev_return = (exit_price / position["entry_price"] - 1.0) * LEVERAGE
            pnl = capital * lev_return
            capital = max(capital + pnl, 0.0)

            trades.append(
                Trade(
                    entry_time=position["entry_time"],
                    exit_time=idx,
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
                    capital_at_entry=position["capital"],
                    pnl=pnl,
                    return_pct=lev_return * 100.0,
                    exit_reason="Bear/Crash Regime",
                    votes=position["votes"],
                )
            )
            cooldown_until = idx + pd.Timedelta(hours=COOLDOWN_HOURS)
            position = None

        # ── Check entry conditions ────────────────────────────────────────────
        if position is None:
            in_cooldown = cooldown_until is not None and idx < cooldown_until
            if (
                not in_cooldown
                and regime == "Bull Run"
                and int(row["Votes"]) >= MIN_VOTES
            ):
                position = {
                    "entry_price": float(row["Close"]),
                    "entry_time": idx,
                    "capital": capital,
                    "votes": int(row["Votes"]),
                }

        # ── Mark-to-market portfolio value ────────────────────────────────────
        if position is not None:
            unreal = (float(row["Close"]) / position["entry_price"] - 1.0) * LEVERAGE * capital
            portfolio_values.append(capital + unreal)
        else:
            portfolio_values.append(capital)

    # ── Close any position still open at end of data ─────────────────────────
    if position is not None:
        last = enriched.iloc[-1]
        exit_price = float(last["Close"])
        lev_return = (exit_price / position["entry_price"] - 1.0) * LEVERAGE
        pnl = capital * lev_return
        capital = max(capital + pnl, 0.0)

        trades.append(
            Trade(
                entry_time=position["entry_time"],
                exit_time=enriched.index[-1],
                entry_price=position["entry_price"],
                exit_price=exit_price,
                capital_at_entry=position["capital"],
                pnl=pnl,
                return_pct=lev_return * 100.0,
                exit_reason="End of Data",
                votes=position["votes"],
            )
        )
        portfolio_values[-1] = capital

    # ── Performance metrics ───────────────────────────────────────────────────
    portfolio_series = pd.Series(portfolio_values, index=enriched.index)
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    bh_return = (
        float(enriched["Close"].iloc[-1]) / float(enriched["Close"].iloc[0]) - 1.0
    )
    alpha = total_return - bh_return
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / len(trades) if trades else 0.0

    rolling_max = portfolio_series.cummax()
    drawdowns = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    # ── Current live signal ───────────────────────────────────────────────────
    last_regime = enriched["Regime"].iloc[-1]
    last_votes = int(enriched["Votes"].iloc[-1])
    current_signal = (
        "LONG"
        if last_regime == "Bull Run" and last_votes >= MIN_VOTES
        else "CASH"
    )

    return BacktestResult(
        trades=trades,
        portfolio_values=portfolio_series,
        df=enriched,
        bull_state=bull_state,
        bear_state=bear_state,
        total_return=total_return,
        buy_hold_return=bh_return,
        alpha=alpha,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        n_trades=len(trades),
        current_signal=current_signal,
        current_regime=last_regime,
    )
