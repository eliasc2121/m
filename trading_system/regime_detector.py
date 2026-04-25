"""
Market regime detection.
Classifies each trading day as bull / neutral / bear using rolling
returns and volatility of the market proxy (SPY).
"""

import numpy as np
import pandas as pd

from trading_system.config import (
    MARKET_PROXY, REGIME_LOOKBACK,
    BULL_RETURN_THRESH, BEAR_RETURN_THRESH,
    VOL_HIGH_THRESH, REGIME_LABELS,
    START_DATE, END_DATE,
)


def _download_proxy(start: str = START_DATE, end: str = END_DATE) -> pd.Series:
    from trading_system.data_fetcher import _check_network, download_close_series
    if _check_network():
        return download_close_series(MARKET_PROXY, start, end)
    from trading_system.synthetic_data import generate_market_proxy
    df, _ = generate_market_proxy(start, end)
    return df["Close"]


def compute_regime_series(
    proxy_close: pd.Series | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.Series:
    """
    Returns a daily Series with integer regime labels
        0 → bear
        1 → neutral
        2 → bull
    indexed by the same dates as the proxy.
    """
    if proxy_close is None:
        proxy_close = _download_proxy(start, end)

    proxy_close = proxy_close.sort_index()
    roll_ret    = proxy_close.pct_change(REGIME_LOOKBACK)
    roll_vol    = proxy_close.pct_change().rolling(REGIME_LOOKBACK).std() * np.sqrt(252)

    regime = pd.Series(1, index=proxy_close.index, name="regime")  # default: neutral

    is_bull = (roll_ret >  BULL_RETURN_THRESH) & (roll_vol < VOL_HIGH_THRESH)
    is_bear = (roll_ret <  BEAR_RETURN_THRESH) | (roll_vol > VOL_HIGH_THRESH)

    regime[is_bull] = 2
    regime[is_bear] = 0

    return regime


def get_current_regime(regime_series: pd.Series, date: pd.Timestamp) -> str:
    """Return regime label string for a given date (or the most recent prior date)."""
    idx = regime_series.index.get_indexer([date], method="ffill")[0]
    if idx < 0:
        return "neutral"
    code = int(regime_series.iloc[idx])
    return REGIME_LABELS.get(code, "neutral")
