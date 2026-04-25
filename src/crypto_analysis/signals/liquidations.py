"""Liquidation-flow signal from Deribit recent trades.

When 'liquidation' is set on a trade, the taker side indicates which side is
being wrecked:

    taker direction = sell   →  long being liquidated   (forced sell)
    taker direction = buy    →  short being liquidated  (forced buy)

Heavy long liquidations often mark short-term capitulation lows → contrarian
LONG bias. Heavy short liquidations often mark squeeze highs → contrarian
SHORT bias. The signal is capped because Deribit's trade window is small
compared to CEX perp-dominant venues.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import clip_score, squash
from . import SignalScore

MAX_LIQ_SCORE = 0.6


def compute(liqs: pd.DataFrame) -> SignalScore:
    if liqs is None or liqs.empty or "direction" not in liqs.columns:
        return SignalScore("liquidations", 0.0, 0.0, "no liquidation trades in window")

    df = liqs.copy()
    # Amount is USD notional on Deribit futures.
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce").fillna(0.0)
    long_liq = float(df.loc[df["direction"] == "sell", "amount"].sum())
    short_liq = float(df.loc[df["direction"] == "buy", "amount"].sum())
    total = long_liq + short_liq
    if total <= 0:
        return SignalScore("liquidations", 0.0, 0.1, "liquidation notional is zero")

    # Contrarian: more long liquidations → positive score (buy the capitulation).
    imb = (long_liq - short_liq) / total  # ∈ [-1, 1]
    score = clip_score(squash(imb * 1.2) * MAX_LIQ_SCORE, -MAX_LIQ_SCORE, MAX_LIQ_SCORE)
    conf = 0.35 + 0.25 * min(1.0, len(df) / 50.0)  # more data → higher conf

    tag = (
        "long liquidations dominate (capitulation)" if imb > 0.2 else
        "short liquidations dominate (squeeze)" if imb < -0.2 else
        "balanced liquidation flow"
    )
    rationale = (
        f"n={len(df)} liq trades — long_liq=${long_liq:,.0f}, short_liq=${short_liq:,.0f}, "
        f"imb={imb:+.2%} — {tag}"
    )
    return SignalScore(
        name="liquidations",
        score=score,
        confidence=conf,
        rationale=rationale,
        details={"long_liq": long_liq, "short_liq": short_liq, "imbalance": imb, "n": len(df)},
    )
