"""Spot-perp premium (Deribit perp mark vs Binance spot).

Perp consistently above spot → speculative long pressure → mild short bias.
Perp below spot → stress / deleveraging → mild long bias.
Uses very short-term snapshot, not term-structure.
"""

from __future__ import annotations

from ..indicators import clip_score, squash
from . import SignalScore


def compute(perp_mark: float | None, spot_price: float | None) -> SignalScore:
    if not perp_mark or not spot_price or spot_price <= 0:
        return SignalScore("spot_futures", 0.0, 0.0, "missing spot or perp")
    premium = (perp_mark - spot_price) / spot_price
    # 20 bps premium is already notable for perp vs spot.
    score = -clip_score(squash(premium / 0.002))
    conf = 0.55
    tag = (
        "perp rich over spot" if premium > 0.001
        else "perp discount to spot" if premium < -0.001
        else "perp ≈ spot"
    )
    rationale = f"perp={perp_mark:.2f}, spot={spot_price:.2f}, premium={premium:+.3%} — {tag}"
    return SignalScore(
        name="spot_futures",
        score=score,
        confidence=conf,
        rationale=rationale,
        details={"premium": premium},
    )
