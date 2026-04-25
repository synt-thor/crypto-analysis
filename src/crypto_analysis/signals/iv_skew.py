"""Implied-volatility / realised-volatility spread signal.

When option IV >> recent realised vol, traders are paying up for protection →
indicates hedging pressure / fear → mild contrarian long bias.
When IV collapses below RV, complacency → mild short bias.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import clip_score, squash
from . import SignalScore


def compute(rv_series: pd.DataFrame, atm_iv: float | None) -> SignalScore:
    """rv_series: deribit historical_volatility output (cols: ts, rv).
    atm_iv: latest ATM implied vol percent (e.g. 55.0 for 55%). None if missing.
    """
    if rv_series is None or rv_series.empty or "rv" not in rv_series.columns:
        return SignalScore("iv_skew", 0.0, 0.0, "no RV data")
    if atm_iv is None or atm_iv <= 0:
        return SignalScore("iv_skew", 0.0, 0.2, "no ATM IV available")

    rv = float(rv_series["rv"].dropna().iloc[-1])
    spread = atm_iv - rv
    # Positive spread: IV > RV → fear → contrarian long.
    # Squash 20 pp spread to a full score.
    score = clip_score(squash(spread / 20.0))
    conf = 0.5
    tag = "fear premium (hedging bid)" if spread > 5 else (
        "complacency" if spread < -5 else "balanced vol"
    )
    rationale = f"ATM IV={atm_iv:.1f}%, RV={rv:.1f}% → spread={spread:+.1f}pp — {tag}"
    return SignalScore(
        name="iv_skew",
        score=score,
        confidence=conf,
        rationale=rationale,
        details={"iv": float(atm_iv), "rv": rv, "spread": spread},
    )
