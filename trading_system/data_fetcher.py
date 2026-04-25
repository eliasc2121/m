"""
Data fetcher: tries FMP API first, falls back to calibrated synthetic data
when the network is unavailable.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

FMP_BASE = "https://financialmodelingprep.com/api/v3"
_API_KEY  = os.getenv("FMP_API_KEY", "")
_NET_OK: bool | None = None   # cached connectivity check


def _check_network() -> bool:
    global _NET_OK
    if _NET_OK is not None:
        return _NET_OK
    try:
        r = requests.get(
            f"{FMP_BASE}/profile/AAPL",
            params={"apikey": _API_KEY},
            timeout=5,
        )
        _NET_OK = r.status_code != 403 or "allowlist" not in r.text
    except Exception:
        _NET_OK = False
    return _NET_OK


def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> list | dict:
    params = params or {}
    params["apikey"] = _API_KEY
    url = f"{FMP_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def download_ohlcv_fmp(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = _get(f"historical-price-full/{ticker}", {"from": start, "to": end})
    if not data or "historical" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    rename = {"open": "Open", "high": "High", "low": "Low",
               "adjClose": "Close", "volume": "Volume"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "Close" not in df.columns and "close" in df.columns:
        df["Close"] = df["close"]
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[needed].dropna()


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Try FMP; fall back to synthetic data on network error."""
    if _check_network():
        try:
            df = download_ohlcv_fmp(ticker, start, end)
            if not df.empty:
                return df
        except Exception as exc:
            warnings.warn(f"FMP failed for {ticker}: {exc} — using synthetic data.")

    from trading_system.synthetic_data import generate_ohlcv
    return generate_ohlcv(ticker, start, end)


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    use_fmp = _check_network()
    if not use_fmp:
        print("  [info] No external network — using calibrated synthetic data.")
    price_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = download_ohlcv(ticker, start, end)
            if len(df) >= 60:
                price_data[ticker] = df
        except Exception as exc:
            print(f"  [error] {ticker}: {exc}")
    return price_data


def download_close_series(ticker: str, start: str, end: str) -> pd.Series:
    df = download_ohlcv(ticker, start, end)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df["Close"].rename(ticker)
