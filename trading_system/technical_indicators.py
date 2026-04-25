"""
Technical indicators computed with pure pandas/numpy.
Covers: EMA, MACD, RSI, Bollinger Bands, ATR.
"""

import numpy as np
import pandas as pd
from trading_system.config import (
    EMA_SHORT, EMA_LONG, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    RSI_PERIOD, BB_PERIOD, BB_STD, ATR_PERIOD,
)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(close: pd.Series) -> pd.DataFrame:
    fast = ema(close, MACD_FAST)
    slow = ema(close, MACD_SLOW)
    macd_line   = fast - slow
    signal_line = ema(macd_line, MACD_SIGNAL)
    histogram   = macd_line - signal_line
    return pd.DataFrame({
        "macd":      macd_line,
        "macd_sig":  signal_line,
        "macd_hist": histogram,
    })


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi")


def compute_bollinger(close: pd.Series) -> pd.DataFrame:
    mid   = close.rolling(BB_PERIOD).mean()
    std   = close.rolling(BB_PERIOD).std()
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std
    pct_b = (close - lower) / (upper - lower + 1e-9)  # normalised position
    width = (upper - lower) / (mid + 1e-9)             # band width
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid":   mid,
        "bb_lower": lower,
        "bb_pct_b": pct_b,
        "bb_width": width,
    })


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean().rename("atr")


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with columns: Open, High, Low, Close, Volume.
    Returns enriched DataFrame with all indicator columns appended.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    df = df.copy()

    # Trend
    df["ema_short"]   = ema(c, EMA_SHORT)
    df["ema_long"]    = ema(c, EMA_LONG)
    df["ema_cross"]   = df["ema_short"] - df["ema_long"]  # positive → short > long

    # Momentum
    macd_df = compute_macd(c)
    df = pd.concat([df, macd_df], axis=1)

    # Mean-reversion
    df["rsi"] = compute_rsi(c)
    bb_df = compute_bollinger(c)
    df = pd.concat([df, bb_df], axis=1)

    # Volatility
    df["atr"]          = compute_atr(h, l, c)
    df["atr_pct"]      = df["atr"] / c            # normalised ATR
    df["daily_return"] = c.pct_change()
    df["vol_20"]       = df["daily_return"].rolling(20).std() * np.sqrt(252)

    # Volume signals
    df["vol_ratio"]    = df["Volume"] / df["Volume"].rolling(20).mean()

    # Price momentum
    df["mom_5"]  = c.pct_change(5)
    df["mom_10"] = c.pct_change(10)
    df["mom_20"] = c.pct_change(20)

    return df
