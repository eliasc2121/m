"""
Visualisation utilities: equity curve, drawdown, regime bands, feature importance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from trading_system.config import INITIAL_CAPITAL, REGIME_LABELS


_REGIME_COLORS = {0: "#ffcccc", 1: "#ffffcc", 2: "#ccffcc"}   # bear/neutral/bull


def plot_equity_curve(
    equity: pd.Series,
    benchmarks: dict[str, pd.Series] | None = None,
    regime_series: pd.Series | None = None,
    save_path: str | Path | None = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("Hybrid AI Trading System – Backtest Results", fontsize=14, fontweight="bold")

    ax_eq, ax_dd, ax_reg = axes

    # ── Equity / benchmark panel ──────────────────────────────────────────────
    _shade_regimes(ax_eq, equity.index, regime_series)

    norm_eq = equity / INITIAL_CAPITAL * 100
    ax_eq.plot(norm_eq.index, norm_eq.values, color="#1f77b4", lw=1.8, label="Strategy")

    if benchmarks:
        colors = ["#ff7f0e", "#2ca02c", "#d62728"]
        for (name, bm_series), col in zip(benchmarks.items(), colors):
            bm_aligned = bm_series.reindex(equity.index, method="ffill").dropna()
            if not bm_aligned.empty:
                norm_bm = bm_aligned / bm_aligned.iloc[0] * 100
                ax_eq.plot(norm_bm.index, norm_bm.values, lw=1.2,
                           linestyle="--", color=col, label=name)

    ax_eq.axhline(100, color="grey", lw=0.7, linestyle=":")
    ax_eq.set_ylabel("Portfolio Value (base=100)")
    ax_eq.legend(loc="upper left", fontsize=9)
    ax_eq.grid(True, alpha=0.3)

    # ── Drawdown panel ────────────────────────────────────────────────────────
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100
    ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                       color="#d62728", alpha=0.6)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.3)

    # ── Regime panel ──────────────────────────────────────────────────────────
    if regime_series is not None:
        reg_aligned = regime_series.reindex(equity.index, method="ffill").fillna(1)
        ax_reg.step(reg_aligned.index, reg_aligned.values, where="post", color="#555")
        ax_reg.set_yticks([0, 1, 2])
        ax_reg.set_yticklabels(["Bear", "Neutral", "Bull"], fontsize=8)
        ax_reg.set_ylabel("Regime")
        ax_reg.grid(True, alpha=0.3)
    else:
        ax_reg.set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 15,
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    top = importance.head(top_n).sort_values()
    ax.barh(top.index, top.values, color="#1f77b4")
    ax.set_title("XGBoost Feature Importance (top features)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_trade_distribution(
    trades: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    if trades.empty or "pnl" not in trades.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1, ax2 = axes
    ax1.hist(trades["pnl"], bins=40, color="#1f77b4", edgecolor="white")
    ax1.axvline(0, color="red", lw=1.2, linestyle="--")
    ax1.set_title("P&L Distribution per Trade")
    ax1.set_xlabel("P&L ($)")

    if "exit_date" in trades.columns and "entry_date" in trades.columns:
        holds = (
            pd.to_datetime(trades["exit_date"]) -
            pd.to_datetime(trades["entry_date"])
        ).dt.days
        ax2.hist(holds, bins=30, color="#ff7f0e", edgecolor="white")
        ax2.set_title("Holding Period Distribution")
        ax2.set_xlabel("Days Held")

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shade_regimes(ax, index, regime_series):
    if regime_series is None:
        return
    reg = regime_series.reindex(index, method="ffill").fillna(1).astype(int)
    prev_val  = reg.iloc[0]
    prev_date = index[0]
    for dt, val in reg.items():
        if val != prev_val:
            ax.axvspan(prev_date, dt, color=_REGIME_COLORS.get(prev_val, "white"), alpha=0.25)
            prev_val, prev_date = val, dt
    ax.axvspan(prev_date, index[-1], color=_REGIME_COLORS.get(prev_val, "white"), alpha=0.25)


def _save_or_show(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
