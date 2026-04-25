"""
Sentiment analysis using FinBERT (ProsusAI/finbert).
Falls back to a seeded pseudo-random mock when USE_MOCK_SENTIMENT=True
or when the model cannot be loaded (no GPU / network).
"""

from __future__ import annotations

import hashlib
import warnings
from typing import List

import numpy as np
import pandas as pd

from trading_system.config import (
    FINBERT_MODEL, SENTIMENT_WEIGHT, USE_MOCK_SENTIMENT
)


# ── FinBERT loader (lazy, cached) ─────────────────────────────────────────────

_pipeline = None


def _load_finbert():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            max_length=512,
            truncation=True,
        )
        return _pipeline
    except Exception as exc:
        warnings.warn(f"FinBERT could not be loaded ({exc}). Using mock sentiment.")
        return None


# ── Scoring helpers ────────────────────────────────────────────────────────────

_LABEL_SCORE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def _score_from_finbert(texts: List[str]) -> List[float]:
    pipe = _load_finbert()
    if pipe is None:
        return _mock_scores(texts)
    results = pipe(texts)
    scores = []
    for r in results:
        label = r["label"].lower()
        conf  = r["score"]
        scores.append(_LABEL_SCORE.get(label, 0.0) * conf)
    return scores


def _mock_scores(texts: List[str]) -> List[float]:
    """
    Deterministic mock: hashes each text and maps to [-1, 1].
    Produces reproducible results without any network or model access.
    """
    scores = []
    for t in texts:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        # Map to [-0.5, 0.5] — subtle, not noisy
        scores.append((h % 1000) / 1000.0 - 0.5)
    return scores


def score_headlines(headlines: List[str]) -> float:
    """Return a single composite sentiment score in [-1, 1]."""
    if not headlines:
        return 0.0
    if USE_MOCK_SENTIMENT:
        raw = _mock_scores(headlines)
    else:
        raw = _score_from_finbert(headlines)
    return float(np.mean(raw))


# ── Build per-ticker daily sentiment DataFrame ─────────────────────────────────

def build_sentiment_series(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    news_map: dict | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame[ticker × date] → sentiment score in [-1, 1].

    news_map: dict[ticker → dict[date_str → List[str]]]
        If None, mock sentiment is generated per (ticker, date).
    """
    sent_dict: dict[str, pd.Series] = {}

    for ticker in tickers:
        scores = []
        for dt in dates:
            date_str = str(dt.date())
            if news_map and ticker in news_map and date_str in news_map[ticker]:
                headlines = news_map[ticker][date_str]
                score = score_headlines(headlines)
            else:
                # Deterministic mock keyed by ticker + date
                score = _mock_scores([f"{ticker}_{date_str}"])[0]
            scores.append(score)
        sent_dict[ticker] = pd.Series(scores, index=dates, name=ticker)

    return pd.DataFrame(sent_dict)


def blend_sentiment(
    signal_score: float,
    sentiment_score: float,
    weight: float = SENTIMENT_WEIGHT,
) -> float:
    """Blend ML signal [−1,1] with sentiment [−1,1]."""
    return (1 - weight) * signal_score + weight * sentiment_score
