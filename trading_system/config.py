"""
Configuration for the Hybrid AI Trading System.
Based on: "Generating Alpha" (arXiv:2601.19504)
"""

# ── Universe ──────────────────────────────────────────────────────────────────
# Top 100 S&P 500 constituents by market cap (as of paper period)
SP500_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "V", "XOM", "AVGO", "PG", "MA", "COST", "HD", "JNJ",
    "MRK", "ABBV", "CVX", "CRM", "AMD", "NFLX", "BAC", "PEP", "KO", "TMO",
    "ACN", "MCD", "CSCO", "ADBE", "ABT", "DHR", "TXN", "LIN", "NKE", "WMT",
    "PM", "NEE", "ORCL", "RTX", "QCOM", "HON", "UPS", "IBM", "GE", "CAT",
    "SBUX", "AMGN", "INTU", "ELV", "SPGI", "BKNG", "DE", "AXP", "GILD", "PLD",
    "GS", "BLK", "MMC", "TJX", "SYK", "MDT", "MDLZ", "ADP", "VRTX", "ISRG",
    "CI", "CB", "REGN", "MS", "EOG", "SO", "DUK", "MO", "ZTS", "CL",
    "NOC", "PNC", "USB", "BSX", "ETN", "WM", "CME", "AON", "ICE", "FCX",
    "HCA", "SLB", "TGT", "EMR", "GD", "ITW", "MCO", "APD", "NSC", "KLAC",
]

MARKET_PROXY = "SPY"

# ── Backtest Period ────────────────────────────────────────────────────────────
START_DATE = "1998-01-01"
END_DATE   = "2026-04-01"
INITIAL_CAPITAL = 100_000.0

# ── Technical Indicator Parameters ────────────────────────────────────────────
EMA_SHORT   = 20
EMA_LONG    = 50
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
RSI_PERIOD  = 14
BB_PERIOD   = 20
BB_STD      = 2.0
ATR_PERIOD  = 14

# ── Regime Detection ──────────────────────────────────────────────────────────
REGIME_LOOKBACK      = 20          # rolling window (trading days)
BULL_RETURN_THRESH   =  0.02       # 20-day return > +2% → bull
BEAR_RETURN_THRESH   = -0.02       # 20-day return < -2% → bear
VOL_HIGH_THRESH      =  0.25       # annualised vol > 25% → high-vol regime
REGIME_LABELS        = {0: "bear", 1: "neutral", 2: "bull"}

# ── XGBoost Signal Generator ──────────────────────────────────────────────────
FORWARD_RETURN_DAYS  = 5           # label: sign of 5-day forward return
XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
}
MIN_TRAIN_SAMPLES = 200

# ── Portfolio / Risk Management ───────────────────────────────────────────────
MAX_POSITIONS        = 20          # max concurrent open positions
MAX_POSITION_SIZE    = 0.05        # 5 % of portfolio per stock
STOP_LOSS_PCT        = 0.05        # 5 % hard stop-loss
TAKE_PROFIT_PCT      = 0.15        # 15 % take-profit
REGIME_SCALE = {                   # exposure scalar per regime
    "bull":    1.0,
    "neutral": 0.6,
    "bear":    0.0,
}

# ── Sentiment ─────────────────────────────────────────────────────────────────
FINBERT_MODEL        = "ProsusAI/finbert"
SENTIMENT_WEIGHT     = 0.20        # blend weight in composite score
USE_MOCK_SENTIMENT   = True        # set False when news API is available
