"""
data_loader.py
Fetches BTC-USD hourly OHLCV data from yfinance with disk caching.
Fetches in 59-day chunks to work around the yfinance 1h lookback limit.
"""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_EXPIRY_HOURS = 1  # Refresh cache after 1 hour


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance MultiIndex columns to simple column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_chunk(
    ticker: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Download a single ≤59-day chunk of hourly data."""
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return _flatten_columns(df)
    except Exception as exc:
        logger.warning("Chunk %s→%s failed: %s", start.date(), end.date(), exc)
        return pd.DataFrame()


def fetch_ohlcv_data(ticker: str = "BTC-USD", days: int = 365) -> pd.DataFrame:
    """
    Return a cleaned hourly OHLCV DataFrame for *ticker* covering the last
    *days* calendar days.

    Results are cached to ``cache/<ticker>_hourly_<days>d.pkl`` and reused
    for up to CACHE_EXPIRY_HOURS hours.

    Args:
        ticker: Yahoo Finance ticker symbol (default ``"BTC-USD"``).
        days:   Number of calendar days to look back (default 730).

    Returns:
        DataFrame with DatetimeIndex and columns Open, High, Low, Close, Volume.

    Raises:
        ValueError: If no data can be retrieved after all chunks.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    safe_ticker = ticker.replace("-", "_")
    cache_file = CACHE_DIR / f"{safe_ticker}_hourly_{days}d.pkl"

    # ── Serve from cache if fresh ─────────────────────────────────────────────
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < CACHE_EXPIRY_HOURS:
            logger.info(
                "Loading %s from cache (%.1f h old).", ticker, age_hours
            )
            df = pd.read_pickle(cache_file)
            df.index = pd.to_datetime(df.index)
            return df

    # ── Fetch in 59-day chunks ────────────────────────────────────────────────
    logger.info("Fetching %d days of hourly %s data …", days, ticker)
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    chunk_days = 59

    dfs: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        chunk = _fetch_chunk(ticker, chunk_start, chunk_end)
        if not chunk.empty:
            dfs.append(chunk)
        chunk_start = chunk_end

    if not dfs:
        raise ValueError(f"Could not retrieve any data for {ticker}.")

    df = pd.concat(dfs)
    df.index = pd.to_datetime(df.index)

    # Keep only the OHLCV columns we care about
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' missing from data.")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    logger.info(
        "Fetched %d candles (%s → %s).",
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )

    df.to_pickle(cache_file)
    return df
