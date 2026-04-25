"""Perpetual funding-rate signal.

Extreme positive funding = overcrowded longs paying shorts → contrarian short bias.
Extreme negative funding = overcrowded shorts paying longs → contrarian long bias.
Near-zero = neutral.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import funding_apr, squash, zscore
from . import SignalScore

# Threshold on 8h funding rate where the crowd is clearly one-sided (heuristic).
CROWDED_APR = 0.30   # 30% APR is already very rich for perpetual
EXTREME_APR = 0.80


def compute(funding_df: pd.DataFrame) -> SignalScore:
    """funding_df: rows with 'ts' and 'funding_rate' (8h decimal, e.g. 0.0001 = 0.01%)."""
    if funding_df is None or funding_df.empty or "funding_rate" not in funding_df.columns:
        return SignalScore("funding", 0.0, 0.0, "no funding data")

    df = funding_df.dropna(subset=["funding_rate"]).sort_values("ts")
    if df.empty:
        return SignalScore("funding", 0.0, 0.0, "empty after dropna")

    latest = df["funding_rate"].iloc[-1]
    latest_apr = funding_apr(latest)

    # Contrarian: positive funding → negative score.
    base = -squash(latest_apr / CROWDED_APR, scale=1.0)

    # Confidence rises with sample size and with distance from zero.
    conf = min(1.0, len(df) / 168.0) * min(1.0, abs(latest_apr) / CROWDED_APR + 0.2)

    # Z-score flag for regime context.
    z = zscore(df["funding_rate"], window=min(168, max(20, len(df) // 3))).iloc[-1]
    tag = "neutral"
    if latest_apr > EXTREME_APR:
        tag = "extreme long crowding"
    elif latest_apr > CROWDED_APR:
        tag = "long crowding"
    elif latest_apr < -EXTREME_APR:
        tag = "extreme short crowding"
    elif latest_apr < -CROWDED_APR:
        tag = "short crowding"

    rationale = (
        f"latest 8h funding={latest:+.5f} ({latest_apr:+.2%} APR), "
        f"z={z:+.2f} vs recent window — {tag}; contrarian score={base:+.3f}"
    )
    return SignalScore(
        name="funding",
        score=base,
        confidence=conf,
        rationale=rationale,
        details={"latest_apr": float(latest_apr), "zscore": float(z) if pd.notna(z) else None},
    )
