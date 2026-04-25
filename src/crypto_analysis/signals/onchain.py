"""On-chain macro backdrop.

Short-term trading is weakly affected by on-chain, but extreme network stress
(mempool blowouts, large fee spikes, hashrate drops) flags environment shifts.
"""

from __future__ import annotations

from typing import Any

from ..indicators import clip_score, squash
from . import SignalScore


def compute(
    mempool: dict[str, Any] | None,
    fees: dict[str, Any] | None,
    hashrate_3d: dict[str, Any] | None,
) -> SignalScore:
    if not (mempool or fees or hashrate_3d):
        return SignalScore("onchain", 0.0, 0.0, "no on-chain data")

    score = 0.0
    bits = []

    if fees:
        fastest = fees.get("fastestFee") or fees.get("fast_fee") or 0
        if fastest:
            # Rough heuristic: extreme fees (>200 sat/vB) usually accompany
            # euphoria (short bias) or ordinals-driven churn (neutral).
            contribution = -squash((fastest - 60) / 120.0) * 0.3
            score += contribution
            bits.append(f"fastestFee={fastest} sat/vB")

    if mempool:
        count = mempool.get("count", 0)
        vsize = mempool.get("vsize", 0)
        # Very deep backlog tends to pair with speculative froth.
        if count > 200_000:
            score -= 0.1
        bits.append(f"mempool={count:,} tx")
        if vsize:
            bits.append(f"vsize={vsize/1_000_000:.1f} MvB")

    if hashrate_3d:
        cur = hashrate_3d.get("currentHashrate") or 0
        diff_change = hashrate_3d.get("currentDifficulty") or 0
        if cur:
            bits.append(f"hashrate≈{cur/1e18:.1f} EH/s")
        if diff_change:
            # Rising difficulty historically coincides with bull regime momentum.
            score += 0.05

    score = clip_score(score)
    conf = 0.35  # on-chain is weak for short timeframe
    return SignalScore(
        name="onchain",
        score=score,
        confidence=conf,
        rationale="; ".join(bits) if bits else "sparse on-chain inputs",
    )
