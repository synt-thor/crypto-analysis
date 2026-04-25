"""Open-interest + price signal.

Direction of OI change combined with price direction:
  price up + OI up     → new longs opening        →  +long bias
  price up + OI down   → shorts covering (fragile) →  mild long, low confidence
  price down + OI up   → new shorts opening       →  +short bias
  price down + OI down → longs liquidating        →  mild short, fading
"""

from __future__ import annotations

import pandas as pd

from ..indicators import clip_score
from . import SignalScore


def compute(ticker_history: pd.DataFrame) -> SignalScore:
    """ticker_history: rows with 'ts', 'mark_price', 'open_interest' (time series)."""
    if ticker_history is None or ticker_history.empty:
        return SignalScore("oi", 0.0, 0.0, "no ticker history")

    required = {"mark_price", "open_interest"}
    if not required.issubset(ticker_history.columns):
        return SignalScore("oi", 0.0, 0.0, "ticker_history missing columns")

    df = ticker_history.sort_values("ts").tail(48).reset_index(drop=True)
    if len(df) < 4:
        return SignalScore("oi", 0.0, 0.2, "insufficient OI samples")

    p0, p1 = df["mark_price"].iloc[0], df["mark_price"].iloc[-1]
    oi0, oi1 = df["open_interest"].iloc[0], df["open_interest"].iloc[-1]
    if p0 <= 0 or oi0 <= 0:
        return SignalScore("oi", 0.0, 0.2, "invalid base values")

    dp = (p1 - p0) / p0
    doi = (oi1 - oi0) / oi0

    if dp > 0 and doi > 0:
        score = +min(0.8, 2 * min(dp, 0.05) + 2 * min(doi, 0.05) * 4)
        tag = "fresh longs building"
    elif dp < 0 and doi > 0:
        score = -min(0.8, 2 * min(-dp, 0.05) + 2 * min(doi, 0.05) * 4)
        tag = "fresh shorts building"
    elif dp > 0 and doi < 0:
        score = +0.15
        tag = "short squeeze / covering (fragile)"
    else:  # dp < 0, doi < 0
        score = -0.15
        tag = "long liquidation fading"

    score = clip_score(score)
    conf = min(1.0, len(df) / 48.0)
    rationale = (
        f"price Δ={dp:+.2%} over {len(df)} samples, OI Δ={doi:+.2%} — {tag}"
    )
    return SignalScore(
        name="oi",
        score=score,
        confidence=conf,
        rationale=rationale,
        details={"dp": float(dp), "doi": float(doi)},
    )
