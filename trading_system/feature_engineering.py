"""
Feature engineering: combines price indicators, regime encoding, and sentiment
into the final feature matrix consumed by the XGBoost signal generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.config import (
    SP500_UNIVERSE, START_DATE, END_DATE, FORWARD_RETURN_DAYS
)
from trading_system.technical_indicators import add_all_indicators
from trading_system.regime_detector import compute_regime_series
from trading_system.data_fetcher import download_prices  # noqa: F401 – re-exported


# ── Indicator feature columns used by the model ───────────────────────────────
FEATURE_COLS = [
    "ema_cross", "ema_cross_norm",
    "macd", "macd_sig", "macd_hist",
    "rsi",
    "bb_pct_b", "bb_width",
    "atr_pct",
    "vol_20",
    "vol_ratio",
    "mom_5", "mom_10", "mom_20",
    "daily_return",
    "regime",        # encoded: 0/1/2
    "sentiment",     # [-1, 1]
]


def download_prices(
    tickers: list[str],
    start: str = START_DATE,
    end: str   = END_DATE,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for each ticker and return dict[ticker → DataFrame]."""
    from trading_system.data_fetcher import download_prices as _fetch
    return _fetch(tickers, start, end)


def build_features_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    regime_series: pd.Series,
    sentiment_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Given raw OHLCV data for one ticker, compute all features and
    attach regime + sentiment.
    """
    df = add_all_indicators(df.copy())

    # Normalise EMA cross by price level
    df["ema_cross_norm"] = df["ema_cross"] / df["Close"]

    # Attach regime (align on date index)
    df["regime"] = regime_series.reindex(df.index, method="ffill").fillna(1)

    # Attach sentiment
    if sentiment_series is not None:
        df["sentiment"] = sentiment_series.reindex(df.index, method="ffill").fillna(0)
    else:
        df["sentiment"] = 0.0

    # Forward return label: sign of 5-day ahead return → 1 (up) / 0 (down)
    df["fwd_return"] = df["Close"].pct_change(FORWARD_RETURN_DAYS).shift(-FORWARD_RETURN_DAYS)
    df["label"]      = (df["fwd_return"] > 0).astype(int)

    df["ticker"] = ticker
    return df


def build_full_dataset(
    price_data: dict[str, pd.DataFrame],
    regime_series: pd.Series,
    sentiment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Concatenate per-ticker feature DataFrames into one panel dataset.
    """
    frames = []
    for ticker, df in price_data.items():
        sent = (
            sentiment_df[ticker]
            if sentiment_df is not None and ticker in sentiment_df.columns
            else None
        )
        feat_df = build_features_for_ticker(ticker, df, regime_series, sent)
        frames.append(feat_df)

    if not frames:
        raise ValueError("No data available to build dataset.")

    panel = pd.concat(frames, axis=0).sort_index()
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["label"])
    return panel


def split_train_test(
    panel: pd.DataFrame,
    split_date: str = "2024-07-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train / test split (no look-ahead)."""
    split = pd.Timestamp(split_date)
    train = panel[panel.index < split]
    test  = panel[panel.index >= split]
    return train, test
