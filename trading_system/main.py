"""
Main entry point for the Hybrid AI Trading System.

Pipeline:
  1. Download price data for S&P 500 universe.
  2. Compute market regime series (SPY proxy).
  3. Build mock sentiment scores (or real FinBERT if news available).
  4. Engineer features and build panel dataset.
  5. Train XGBoost signal generator on the training split.
  6. Run event-driven backtest on the test split.
  7. Print performance metrics and save plots.

Usage:
  python -m trading_system.main [--tickers N] [--output-dir ./results]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trading_system.config import (
    SP500_UNIVERSE, START_DATE, END_DATE, INITIAL_CAPITAL, MARKET_PROXY
)
from trading_system.data_fetcher import download_prices, download_close_series
from trading_system.feature_engineering import build_full_dataset, split_train_test
from trading_system.regime_detector import compute_regime_series
from trading_system.sentiment_analyzer import build_sentiment_series
from trading_system.signal_generator import SignalGenerator
from trading_system.backtester import Backtester
from trading_system.visualizer import (
    plot_equity_curve, plot_feature_importance, plot_trade_distribution
)


def download_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    return download_close_series(ticker, start, end)


def run(n_tickers: int = 20, output_dir: str = "./results") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tickers = SP500_UNIVERSE[:n_tickers]
    print(f"\n{'='*60}")
    print(f"  Hybrid AI Trading System  (arXiv:2601.19504)")
    print(f"  Universe : {len(tickers)} tickers  |  {START_DATE} → {END_DATE}")
    print(f"  Capital  : ${INITIAL_CAPITAL:,.0f}")
    print(f"{'='*60}\n")

    # ── 1. Price data ─────────────────────────────────────────────────────────
    print("[1/6] Downloading price data …")
    price_data = download_prices(tickers, START_DATE, END_DATE)
    available  = list(price_data.keys())
    print(f"      {len(available)}/{len(tickers)} tickers downloaded successfully.")

    if not available:
        sys.exit("No price data downloaded. Check internet connection.")

    # ── 2. Market regime ──────────────────────────────────────────────────────
    print("[2/6] Computing market regime …")
    proxy_close    = download_benchmark(MARKET_PROXY, START_DATE, END_DATE)
    regime_series  = compute_regime_series(proxy_close, START_DATE, END_DATE)
    regime_counts  = regime_series.value_counts().sort_index()
    print(f"      Bear days: {regime_counts.get(0,0)}  "
          f"Neutral: {regime_counts.get(1,0)}  "
          f"Bull: {regime_counts.get(2,0)}")

    # ── 3. Sentiment ──────────────────────────────────────────────────────────
    print("[3/6] Generating sentiment scores (mock) …")
    all_dates     = pd.bdate_range(start=START_DATE, end=END_DATE)
    sentiment_df  = build_sentiment_series(available, all_dates)
    print(f"      Shape: {sentiment_df.shape}")

    # ── 4. Feature engineering ────────────────────────────────────────────────
    print("[4/6] Engineering features …")
    panel = build_full_dataset(price_data, regime_series, sentiment_df)
    train_df, test_df = split_train_test(panel, split_date="2024-07-01")
    print(f"      Panel rows: {len(panel):,}  |  "
          f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # ── 5. Train XGBoost ──────────────────────────────────────────────────────
    print("[5/6] Training XGBoost signal generator …")
    sig_gen = SignalGenerator()
    sig_gen.fit(train_df)

    # Feature importance
    fi = sig_gen.feature_importance()
    print("      Top-5 features:")
    for feat, imp in fi.head(5).items():
        print(f"        {feat:<20s} {imp:.4f}")
    plot_feature_importance(fi, save_path=out / "feature_importance.png")

    # ── 6. Backtest ───────────────────────────────────────────────────────────
    print("[6/6] Running backtest on test period (2024-07-01 → 2025-01-01) …")
    backtester = Backtester(
        signal_gen    = sig_gen,
        price_data    = price_data,
        regime_series = regime_series,
        sentiment_df  = sentiment_df,
        panel         = test_df,
    )
    result = backtester.run(start="2024-07-01", end="2025-01-01")

    print("\n" + str(result))

    # Save trades
    if not result.trades.empty:
        result.trades.to_csv(out / "trades.csv", index=False)
        print(f"\n  Trades saved → {out/'trades.csv'}")

    # Save equity curve
    result.equity_curve.to_csv(out / "equity_curve.csv", header=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    # Download benchmarks for comparison
    benchmarks = {}
    for bm in ["SPY", "QQQ"]:
        try:
            bm_prices = download_benchmark(bm, "2024-07-01", "2025-01-01")
            benchmarks[bm] = bm_prices
        except Exception:
            pass

    plot_equity_curve(
        result.equity_curve,
        benchmarks    = benchmarks,
        regime_series = regime_series,
        save_path     = out / "equity_curve.png",
    )
    plot_trade_distribution(result.trades, save_path=out / "trade_distribution.png")

    print(f"\nAll outputs saved to: {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid AI Trading System (arXiv:2601.19504)"
    )
    parser.add_argument(
        "--tickers", type=int, default=20,
        help="Number of S&P 500 tickers to include (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to save plots and CSVs (default: ./results)",
    )
    args = parser.parse_args()
    run(n_tickers=args.tickers, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
