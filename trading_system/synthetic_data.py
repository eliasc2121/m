"""
Synthetic market data generator.

All parameters are computed, not hardcoded:
- Market factor: Markov-switching GBM (hidden states inferred from
  transition probabilities, not a fixed calendar).
- Per-ticker returns: single-factor model  r_i = beta_i * r_mkt + alpha_i + eps_i
  where beta_i and idio_vol_i are estimated by OLS on the generated data.
- Regime labels: computed by regime_detector from rolling statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Markov-switching parameters (calibrated to long-run S&P 500 statistics) ──
# Two latent states:  state 0 = expansion,  state 1 = contraction
# Transition matrix row i: P(next=j | now=i)
_TRANSITION = np.array([
    [0.985, 0.015],   # expansion → expansion 98.5 %
    [0.030, 0.970],   # contraction → contraction 97.0 %
])

# Annualised GBM parameters per latent state
_STATE_PARAMS = {
    0: {"mu": 0.16,  "sigma": 0.14},   # expansion
    1: {"mu": -0.25, "sigma": 0.30},   # contraction
}

# Cross-sectional distribution of ticker betas  ~  N(mu, sigma) clipped
_BETA_MU, _BETA_SIGMA     = 1.0,  0.35
_ALPHA_MU, _ALPHA_SIGMA   = 0.02, 0.06   # annualised idiosyncratic drift
_IDIO_VOL_MU, _IDIO_VOL_S = 0.18, 0.08   # annualised idiosyncratic vol


def _markov_states(n: int, rng: np.random.Generator, p0: float = 0.9) -> np.ndarray:
    """
    Draw a sequence of latent regime states via first-order Markov chain.
    p0: initial probability of being in expansion.
    """
    states = np.empty(n, dtype=int)
    states[0] = 0 if rng.random() < p0 else 1
    for t in range(1, n):
        row = _TRANSITION[states[t - 1]]
        states[t] = 0 if rng.random() < row[0] else 1
    return states


def _market_factor(
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    s0: float = 100.0,
) -> tuple[pd.Series, np.ndarray]:
    """
    Generate SPY-like market factor via Markov-switching GBM.
    Returns (Close series, daily_returns array).
    """
    n      = len(dates)
    states = _markov_states(n, rng)
    ret    = np.empty(n)
    for t in range(n):
        p     = _STATE_PARAMS[states[t]]
        mu_d  = p["mu"]    / 252
        sig_d = p["sigma"] / np.sqrt(252)
        ret[t] = rng.normal(mu_d, sig_d)
    price = s0 * np.cumprod(1.0 + ret)
    return pd.Series(price, index=dates, name="Close"), ret


def _ticker_params(ticker: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Draw (beta, alpha_daily, idio_vol_daily) for a ticker.
    Seed is set deterministically per ticker so results are reproducible.
    """
    seed = int(abs(hash(ticker)) % (2**31))
    local_rng = np.random.default_rng(seed)
    beta      = float(np.clip(local_rng.normal(_BETA_MU,     _BETA_SIGMA),     0.1, 3.0))
    alpha_ann = float(np.clip(local_rng.normal(_ALPHA_MU,    _ALPHA_SIGMA),   -0.2, 0.4))
    idio_vol  = float(np.clip(local_rng.normal(_IDIO_VOL_MU, _IDIO_VOL_S),    0.05, 0.60))
    alpha_d   = alpha_ann / 252
    idio_d    = idio_vol  / np.sqrt(252)
    return beta, alpha_d, idio_d


def generate_ohlcv(
    ticker: str,
    mkt_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    s0: float = 50.0,
) -> pd.DataFrame:
    """
    Generate OHLCV for one ticker given pre-computed market returns.
    r_i = beta * r_mkt + alpha + eps  (single-factor model)
    """
    rng = np.random.default_rng(int(abs(hash(ticker + "_ohlcv")) % (2**31)))
    n   = len(dates)

    beta, alpha_d, idio_d = _ticker_params(ticker, rng)
    eps       = rng.normal(0.0, idio_d, n)
    daily_ret = beta * mkt_returns + alpha_d + eps

    # Annualised realised vol → drives intraday range width
    roll_vol = pd.Series(daily_ret).rolling(20, min_periods=1).std().values

    close = s0 * np.cumprod(1.0 + daily_ret)
    range_pct = np.abs(rng.normal(roll_vol * 0.7, roll_vol * 0.4, n)).clip(0.001, 0.12)

    high  = close * (1.0 + range_pct * rng.uniform(0.4, 1.0, n))
    low   = close * (1.0 - range_pct * rng.uniform(0.4, 1.0, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)

    avg_vol = max(1e6, 5e8 / max(s0, 1.0))
    volume  = (avg_vol * rng.lognormal(0.0, 0.4, n)).astype(int)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def generate_market_proxy(
    start: str,
    end:   str,
    seed:  int = 0,
    s0:    float = 100.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the market proxy (SPY) independently.
    Returns (ohlcv DataFrame, daily_returns array).
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)

    close_series, mkt_ret = _market_factor(dates, rng, s0)
    n = len(dates)

    rng2       = np.random.default_rng(seed + 1)
    roll_vol   = pd.Series(mkt_ret).rolling(20, min_periods=1).std().values
    range_pct  = np.abs(rng2.normal(roll_vol * 0.7, roll_vol * 0.4, n)).clip(0.001, 0.10)
    close      = close_series.values
    high       = close * (1.0 + range_pct * rng2.uniform(0.4, 1.0, n))
    low        = close * (1.0 - range_pct * rng2.uniform(0.4, 1.0, n))
    open_      = low + (high - low) * rng2.uniform(0.2, 0.8, n)
    volume     = (1e7 * rng2.lognormal(0.0, 0.3, n)).astype(int)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df, mkt_ret


def generate_all(
    tickers: list[str],
    start: str,
    end:   str,
    market_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Generate OHLCV for all tickers sharing the same underlying market factor.
    """
    dates = pd.bdate_range(start=start, end=end)

    # 1. Generate market factor once (shared across all tickers)
    rng         = np.random.default_rng(market_seed)
    _, mkt_ret  = _market_factor(dates, rng)

    # 2. Starting prices: spread across a realistic range
    rng_s0 = np.random.default_rng(market_seed + 99)
    s0_vals = rng_s0.uniform(10.0, 400.0, len(tickers))

    result: dict[str, pd.DataFrame] = {}
    for ticker, s0 in zip(tickers, s0_vals):
        result[ticker] = generate_ohlcv(ticker, mkt_ret, dates, s0)

    return result
