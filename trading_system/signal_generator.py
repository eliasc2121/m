"""
XGBoost-based trading signal generator.

Trains a binary classifier (up/down over 5 trading days) on historical
features and returns a continuous probability score used for position sizing.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("xgboost is required: pip install xgboost") from exc

from trading_system.config import XGB_PARAMS, MIN_TRAIN_SAMPLES, SENTIMENT_WEIGHT
from trading_system.feature_engineering import FEATURE_COLS
from trading_system.sentiment_analyzer import blend_sentiment


class SignalGenerator:
    """
    Wraps an XGBoost classifier that predicts 5-day forward return direction.
    predict_proba(:, 1) → probability of an up-move, used as raw signal [0, 1].
    """

    def __init__(self):
        self.model   = XGBClassifier(**XGB_PARAMS)
        self.scaler  = StandardScaler()
        self.trained = False

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> None:
        if len(train_df) < MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_TRAIN_SAMPLES} samples, got {len(train_df)}."
            )

        X = train_df[FEATURE_COLS].values
        y = train_df["label"].values.astype(int)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True
        print(f"[SignalGenerator] Trained on {len(train_df):,} samples.")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_proba(self, feature_row: pd.Series | pd.DataFrame) -> np.ndarray:
        """Return P(up) for each row. Shape: (n,)."""
        if isinstance(feature_row, pd.Series):
            feature_row = feature_row.to_frame().T
        X = feature_row[FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def generate_signal(
        self,
        feature_row: pd.Series,
        sentiment_score: float = 0.0,
    ) -> float:
        """
        Returns a blended composite signal in [-1, 1].
            > 0 → buy candidate
            ≤ 0 → no position / flat
        """
        if not self.trained:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")

        p_up    = float(self.predict_proba(feature_row)[0])
        ml_sig  = (p_up - 0.5) * 2          # map [0,1] → [-1, 1]
        return blend_sentiment(ml_sig, sentiment_score, SENTIMENT_WEIGHT)

    # ── Feature importance ───────────────────────────────────────────────────

    def feature_importance(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=FEATURE_COLS).sort_values(ascending=False)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model   = state["model"]
        self.scaler  = state["scaler"]
        self.trained = True
