"""Top-N order book depth imbalance on BTC-PERPETUAL.

Short-term signal only (minutes–hours). Susceptible to spoofing so it carries
moderate confidence and is bounded to ±0.6 score magnitude.
"""

from __future__ import annotations

from typing import Any

from ..indicators import clip_score
from . import SignalScore

MAX_OB_SCORE = 0.6


def compute(order_book: dict[str, Any] | None, levels: int = 10) -> SignalScore:
    if not order_book or "bids" not in order_book or "asks" not in order_book:
        return SignalScore("orderbook", 0.0, 0.0, "no orderbook")

    bids = order_book["bids"][:levels]
    asks = order_book["asks"][:levels]
    if not bids or not asks:
        return SignalScore("orderbook", 0.0, 0.0, "empty book")

    bid_size = sum(float(size) for _, size in bids)
    ask_size = sum(float(size) for _, size in asks)
    total = bid_size + ask_size
    if total <= 0:
        return SignalScore("orderbook", 0.0, 0.0, "zero depth")

    imbalance = (bid_size - ask_size) / total
    score = clip_score(imbalance * MAX_OB_SCORE, -MAX_OB_SCORE, MAX_OB_SCORE)
    conf = 0.45
    tag = "bids dominating" if imbalance > 0.1 else (
        "asks dominating" if imbalance < -0.1 else "book balanced"
    )
    rationale = (
        f"top-{levels} bids={bid_size:,.0f}, asks={ask_size:,.0f}, "
        f"imbalance={imbalance:+.2%} — {tag}"
    )
    return SignalScore(
        name="orderbook",
        score=score,
        confidence=conf,
        rationale=rationale,
        details={"imbalance": imbalance, "bid_size": bid_size, "ask_size": ask_size},
    )
