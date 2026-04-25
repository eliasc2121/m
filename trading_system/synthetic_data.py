"""
Synthetic market data generator calibrated to S&P 500 2023-2025 parameters.
Used when external APIs are unavailable. Produces correlated, regime-aware
OHLCV price series for the full universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Calibrated S&P 500 parameters (2023-2025) ─────────────────────────────────
# Annualised drift ~24%, vol ~16% for the bull phase; ~-8% / 22% bear phase
_MARKET_PARAMS = {
    "bull":    {"mu": 0.24,  "sigma": 0.16},
    "neutral": {"mu": 0.08,  "sigma": 0.18},
    "bear":    {"mu": -0.10, "sigma": 0.26},
}

# Approximate 2023-2025 regime calendar (rough approximation)
_REGIME_CALENDAR = [
    ("2023-01-01", "2023-03-15", "neutral"),
    ("2023-03-15", "2023-10-30", "bull"),
    ("2023-10-30", "2023-11-15", "bear"),
    ("2023-11-15", "2024-07-10", "bull"),
    ("2024-07-10", "2024-08-15", "bear"),
    ("2024-08-15", "2025-01-01", "bull"),
]

# Per-ticker calibration: (beta, alpha_ann, idio_vol)
_TICKER_PARAMS: dict[str, tuple[float, float, float]] = {
    "AAPL": (1.15, 0.04, 0.12), "MSFT": (1.05, 0.06, 0.10),
    "NVDA": (1.80, 0.20, 0.30), "AMZN": (1.20, 0.08, 0.18),
    "META": (1.30, 0.12, 0.22), "GOOGL":(1.10, 0.05, 0.14),
    "TSLA": (1.90, -0.02, 0.40),"BRK-B":(0.70, 0.02, 0.08),
    "UNH":  (0.60, 0.03, 0.10), "LLY":  (0.55, 0.18, 0.20),
    "JPM":  (1.10, 0.04, 0.14), "V":    (0.90, 0.06, 0.10),
    "XOM":  (0.85, 0.05, 0.15), "AVGO": (1.40, 0.10, 0.20),
    "PG":   (0.50, 0.03, 0.08), "MA":   (0.95, 0.07, 0.11),
    "COST": (0.80, 0.08, 0.12), "HD":   (0.85, 0.04, 0.13),
    "JNJ":  (0.45, 0.01, 0.09), "MRK":  (0.50, 0.05, 0.12),
    "ABBV": (0.55, 0.06, 0.14), "CVX":  (0.80, 0.03, 0.14),
    "CRM":  (1.25, 0.08, 0.22), "AMD":  (1.70, 0.15, 0.32),
    "NFLX": (1.35, 0.10, 0.25), "BAC":  (1.20, 0.03, 0.16),
    "PEP":  (0.55, 0.03, 0.09), "KO":   (0.50, 0.02, 0.08),
    "TMO":  (0.75, 0.04, 0.13), "ACN":  (0.85, 0.05, 0.12),
    "MCD":  (0.65, 0.04, 0.10), "CSCO": (0.90, 0.03, 0.13),
    "ADBE": (1.20, 0.06, 0.20), "ABT":  (0.65, 0.03, 0.11),
    "DHR":  (0.80, 0.04, 0.14), "TXN":  (1.00, 0.04, 0.14),
    "LIN":  (0.70, 0.04, 0.10), "NKE":  (0.90, 0.02, 0.15),
    "WMT":  (0.55, 0.04, 0.09), "PM":   (0.60, 0.03, 0.11),
    "NEE":  (0.65, 0.02, 0.13), "ORCL": (0.95, 0.07, 0.16),
    "RTX":  (0.75, 0.04, 0.12), "QCOM": (1.30, 0.06, 0.22),
    "HON":  (0.80, 0.03, 0.12), "IBM":  (0.80, 0.04, 0.14),
    "GE":   (1.10, 0.10, 0.18), "CAT":  (1.00, 0.05, 0.15),
    "SBUX": (0.85, 0.01, 0.16), "AMGN": (0.60, 0.04, 0.13),
    "INTU": (1.15, 0.09, 0.18), "ELV":  (0.65, 0.05, 0.12),
    "SPGI": (0.90, 0.06, 0.13), "BKNG": (1.10, 0.08, 0.18),
    "DE":   (0.90, 0.03, 0.14), "AXP":  (1.00, 0.06, 0.15),
    "GILD": (0.55, 0.04, 0.12), "PLD":  (0.85, 0.04, 0.15),
    "GS":   (1.20, 0.05, 0.18), "BLK":  (1.10, 0.06, 0.16),
    "MMC":  (0.75, 0.05, 0.11), "TJX":  (0.80, 0.06, 0.12),
    "SYK":  (0.80, 0.05, 0.13), "MDT":  (0.65, 0.01, 0.12),
    "MDLZ": (0.55, 0.03, 0.09), "ADP":  (0.80, 0.05, 0.11),
    "VRTX": (0.70, 0.10, 0.18), "ISRG": (0.85, 0.07, 0.15),
    "CI":   (0.70, 0.05, 0.13), "CB":   (0.70, 0.04, 0.11),
    "REGN": (0.65, 0.08, 0.16), "MS":   (1.25, 0.05, 0.18),
    "EOG":  (1.00, 0.04, 0.18), "SO":   (0.50, 0.02, 0.09),
    "DUK":  (0.50, 0.02, 0.09), "MO":   (0.55, 0.03, 0.11),
    "ZTS":  (0.80, 0.05, 0.14), "CL":   (0.55, 0.03, 0.09),
    "NOC":  (0.70, 0.04, 0.12), "PNC":  (1.10, 0.03, 0.16),
    "USB":  (1.05, 0.02, 0.15), "BSX":  (0.80, 0.06, 0.14),
    "ETN":  (0.95, 0.06, 0.14), "WM":   (0.70, 0.04, 0.11),
    "CME":  (0.75, 0.05, 0.12), "AON":  (0.75, 0.05, 0.11),
    "ICE":  (0.80, 0.05, 0.12), "FCX":  (1.20, 0.05, 0.25),
    "HCA":  (0.85, 0.06, 0.15), "SLB":  (1.05, 0.03, 0.20),
    "TGT":  (0.90, 0.02, 0.16), "EMR":  (0.85, 0.04, 0.13),
    "GD":   (0.70, 0.04, 0.11), "ITW":  (0.80, 0.04, 0.12),
    "MCO":  (0.95, 0.06, 0.14), "APD":  (0.75, 0.04, 0.12),
    "NSC":  (0.80, 0.03, 0.12), "KLAC": (1.40, 0.08, 0.24),
    "SPY":  (1.00, 0.00, 0.00),
}

_DEFAULT_PARAMS = (1.0, 0.04, 0.15)


def _regime_at(date: pd.Timestamp) -> str:
    for start, end, regime in _REGIME_CALENDAR:
        if pd.Timestamp(start) <= date < pd.Timestamp(end):
            return regime
    return "bull"


def _market_returns(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Daily market returns following the regime calendar."""
    n = len(dates)
    ret = np.zeros(n)
    for i, dt in enumerate(dates):
        regime = _regime_at(dt)
        p = _MARKET_PARAMS[regime]
        mu_d    = p["mu"]  / 252
        sig_d   = p["sigma"] / np.sqrt(252)
        ret[i]  = rng.normal(mu_d, sig_d)
    return ret


def generate_ohlcv(
    ticker: str,
    start: str = "2023-01-01",
    end:   str = "2025-01-01",
    seed:  int | None = None,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic OHLCV DataFrame for `ticker`.
    """
    # Deterministic seed per ticker
    if seed is None:
        seed = int(abs(hash(ticker)) % (2**31))
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end)
    n     = len(dates)

    beta, alpha_ann, idio_vol = _TICKER_PARAMS.get(ticker, _DEFAULT_PARAMS)
    idio_d = idio_vol / np.sqrt(252)

    mkt_ret = _market_returns(dates, rng)
    idio_ret = rng.normal(alpha_ann / 252, idio_d, n)
    daily_ret = beta * mkt_ret + idio_ret

    # Realistic starting price (roughly proportional to market cap rank)
    tickers_list = list(_TICKER_PARAMS.keys())
    rank = tickers_list.index(ticker) if ticker in tickers_list else 50
    s0 = max(20.0, 500.0 - rank * 3.5 + rng.normal(0, 10))

    close = s0 * np.cumprod(1 + daily_ret)

    # Build OHLC from close
    daily_range_pct = np.abs(rng.normal(0.008, 0.004, n)).clip(0.001, 0.04)
    high  = close * (1 + daily_range_pct * rng.uniform(0.4, 1.0, n))
    low   = close * (1 - daily_range_pct * rng.uniform(0.4, 1.0, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)

    avg_vol = max(1e6, 5e8 / s0)
    volume  = (avg_vol * rng.lognormal(0, 0.4, n)).astype(int)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def generate_all(
    tickers: list[str],
    start: str = "2023-01-01",
    end:   str = "2025-01-01",
) -> dict[str, pd.DataFrame]:
    return {t: generate_ohlcv(t, start, end) for t in tickers}
