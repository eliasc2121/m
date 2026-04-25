"""
Event-driven backtesting engine.

Daily loop over the test period:
  1. Detect market regime.
  2. For each ticker, generate a composite signal (ML + sentiment).
  3. Rank candidates; open new positions up to MAX_POSITIONS.
  4. Enforce stop-loss / take-profit on open positions.
  5. Log daily portfolio value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from trading_system.config import (
    INITIAL_CAPITAL, MAX_POSITIONS, MAX_POSITION_SIZE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, REGIME_SCALE, REGIME_LABELS,
)
from trading_system.regime_detector import get_current_regime
from trading_system.signal_generator import SignalGenerator
from trading_system.feature_engineering import FEATURE_COLS


@dataclass
class Position:
    ticker:      str
    entry_price: float
    shares:      int
    entry_date:  pd.Timestamp
    stop_loss:   float = field(init=False)
    take_profit: float = field(init=False)

    def __post_init__(self):
        self.stop_loss   = self.entry_price * (1 - STOP_LOSS_PCT)
        self.take_profit = self.entry_price * (1 + TAKE_PROFIT_PCT)

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    def current_value(self, price: float) -> float:
        return price * self.shares

    def pnl(self, price: float) -> float:
        return self.current_value(price) - self.cost_basis

    def should_close(self, price: float) -> bool:
        return price <= self.stop_loss or price >= self.take_profit


@dataclass
class BacktestResult:
    equity_curve:    pd.Series
    trades:          pd.DataFrame
    metrics:         dict

    def __str__(self) -> str:
        m = self.metrics
        lines = [
            "=" * 50,
            "  BACKTEST RESULTS",
            "=" * 50,
            f"  Period            : {m['start_date']} → {m['end_date']}",
            f"  Initial Capital   : ${m['initial_capital']:>12,.2f}",
            f"  Final Value       : ${m['final_value']:>12,.2f}",
            f"  Total Return      : {m['total_return_pct']:>+.2f}%",
            f"  CAGR              : {m['cagr_pct']:>+.2f}%",
            f"  Sharpe Ratio      : {m['sharpe']:.3f}",
            f"  Max Drawdown      : {m['max_drawdown_pct']:.2f}%",
            f"  Win Rate          : {m['win_rate_pct']:.1f}%",
            f"  Total Trades      : {m['total_trades']}",
            "=" * 50,
        ]
        return "\n".join(lines)


class Backtester:
    def __init__(
        self,
        signal_gen:    SignalGenerator,
        price_data:    dict[str, pd.DataFrame],
        regime_series: pd.Series,
        sentiment_df:  pd.DataFrame | None = None,
        panel:         pd.DataFrame | None = None,
    ):
        self.signal_gen    = signal_gen
        self.price_data    = price_data
        self.regime_series = regime_series
        self.sentiment_df  = sentiment_df
        self.panel         = panel   # pre-built feature panel for fast lookup

        self.cash:      float           = INITIAL_CAPITAL
        self.positions: dict[str, Position] = {}
        self.equity_log: list[tuple]    = []
        self.trade_log:  list[dict]     = []

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, start: str, end: str) -> BacktestResult:
        dates = pd.bdate_range(start=start, end=end)
        self.cash      = INITIAL_CAPITAL
        self.positions = {}
        self.equity_log.clear()
        self.trade_log.clear()

        tickers = list(self.price_data.keys())

        for dt in dates:
            current_prices = self._get_prices(tickers, dt)
            if not current_prices:
                continue

            regime_str  = get_current_regime(self.regime_series, dt)
            regime_code = [k for k, v in REGIME_LABELS.items() if v == regime_str][0]
            scale       = REGIME_SCALE.get(regime_str, 0.6)

            # 1. Close positions that hit SL/TP
            self._close_triggered(current_prices, dt)

            # 2. Compute signals for all tickers
            signals = self._compute_signals(tickers, dt, regime_code)

            # 3. Open new positions if we have capacity & regime allows
            if scale > 0:
                self._open_positions(signals, current_prices, dt, scale)

            # 4. Log equity
            portfolio_value = self._portfolio_value(current_prices)
            self.equity_log.append((dt, portfolio_value))

        equity_curve = pd.Series(
            {dt: val for dt, val in self.equity_log}, name="equity"
        )
        trades_df    = pd.DataFrame(self.trade_log)
        metrics      = self._compute_metrics(equity_curve, trades_df, start, end)
        return BacktestResult(equity_curve, trades_df, metrics)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_prices(self, tickers: list[str], dt: pd.Timestamp) -> dict[str, float]:
        prices = {}
        for t in tickers:
            df = self.price_data.get(t)
            if df is None:
                continue
            idx = df.index.get_indexer([dt], method="ffill")[0]
            if idx >= 0:
                prices[t] = float(df["Close"].iloc[idx])
        return prices

    def _compute_signals(
        self,
        tickers: list[str],
        dt: pd.Timestamp,
        regime_code: int,
    ) -> dict[str, float]:
        if self.panel is None:
            return {}

        day_data = self.panel.xs(dt, drop_level=False) if dt in self.panel.index else None
        if day_data is None or day_data.empty:
            return {}

        signals = {}
        for _, row in day_data.iterrows():
            ticker = row.get("ticker")
            if ticker is None:
                continue
            sent = 0.0
            if self.sentiment_df is not None and ticker in self.sentiment_df.columns:
                idx = self.sentiment_df.index.get_indexer([dt], method="ffill")[0]
                if idx >= 0:
                    sent = float(self.sentiment_df[ticker].iloc[idx])
            try:
                sig = self.signal_gen.generate_signal(row, sent)
                signals[ticker] = sig
            except Exception:
                pass

        return signals

    def _open_positions(
        self,
        signals: dict[str, float],
        prices: dict[str, float],
        dt: pd.Timestamp,
        scale: float,
    ) -> None:
        slots = MAX_POSITIONS - len(self.positions)
        if slots <= 0:
            return

        # Rank by signal strength (descending); only buy signals (>0)
        ranked = sorted(
            [(t, s) for t, s in signals.items() if s > 0 and t not in self.positions],
            key=lambda x: x[1],
            reverse=True,
        )[:slots]

        portfolio_val = self._portfolio_value(prices)
        for ticker, sig in ranked:
            price = prices.get(ticker)
            if price is None or price <= 0:
                continue

            # Position size = scale × MAX_POSITION_SIZE × confidence
            confidence   = min(abs(sig), 1.0)
            alloc_frac   = scale * MAX_POSITION_SIZE * confidence
            alloc_cash   = portfolio_val * alloc_frac
            alloc_cash   = min(alloc_cash, self.cash)

            shares = int(alloc_cash // price)
            if shares <= 0:
                continue

            cost = shares * price
            self.cash -= cost
            self.positions[ticker] = Position(ticker, price, shares, dt)

    def _close_triggered(
        self,
        prices: dict[str, float],
        dt: pd.Timestamp,
    ) -> None:
        to_close = [
            t for t, pos in self.positions.items()
            if t in prices and pos.should_close(prices[t])
        ]
        for ticker in to_close:
            self._close_position(ticker, prices[ticker], dt, reason="sl_tp")

    def _close_position(
        self,
        ticker: str,
        price: float,
        dt: pd.Timestamp,
        reason: str = "signal",
    ) -> None:
        pos = self.positions.pop(ticker)
        proceeds = pos.current_value(price)
        self.cash += proceeds
        self.trade_log.append({
            "ticker":     ticker,
            "entry_date": pos.entry_date,
            "exit_date":  dt,
            "entry_price":pos.entry_price,
            "exit_price": price,
            "shares":     pos.shares,
            "pnl":        pos.pnl(price),
            "reason":     reason,
        })

    def _portfolio_value(self, prices: dict[str, float]) -> float:
        pos_value = sum(
            pos.current_value(prices.get(t, pos.entry_price))
            for t, pos in self.positions.items()
        )
        return self.cash + pos_value

    # ── Performance metrics ───────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(
        equity: pd.Series,
        trades: pd.DataFrame,
        start: str,
        end: str,
    ) -> dict:
        if equity.empty:
            return {}

        ret = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_years      = max(len(equity) / 252, 1e-6)
        cagr         = (1 + total_return) ** (1 / n_years) - 1

        sharpe = 0.0
        if ret.std() > 0:
            sharpe = (ret.mean() / ret.std()) * np.sqrt(252)

        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd   = float(drawdown.min())

        win_rate = 0.0
        if not trades.empty and "pnl" in trades.columns:
            win_rate = (trades["pnl"] > 0).mean()

        return {
            "start_date":        start,
            "end_date":          end,
            "initial_capital":   INITIAL_CAPITAL,
            "final_value":       float(equity.iloc[-1]),
            "total_return_pct":  total_return * 100,
            "cagr_pct":          cagr * 100,
            "sharpe":            sharpe,
            "max_drawdown_pct":  max_dd * 100,
            "win_rate_pct":      win_rate * 100,
            "total_trades":      len(trades),
        }
