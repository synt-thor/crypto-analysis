"""Macro numeric signal: DXY / SPY / VIX / 10Y trend vs BTC.

Heuristics (historically stable but regime-dependent):
  DXY up      →  BTC down pressure (inverse)
  SPY up      →  BTC up (risk-on co-move)
  VIX up      →  BTC down (risk-off)
  TNX up      →  BTC down (tightening liquidity)
Combined into single score weighted by |trend|.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import clip_score, squash
from . import SignalScore

# Sign of each macro's historical correlation to BTC.
CORR_SIGN = {"DXY": -1, "SPY": +1, "QQQ": +1, "VIX": -1, "TNX": -1, "GOLD": 0}


def _momentum(closes: pd.Series, fast: int = 5, slow: int = 20) -> float:
    if len(closes) < slow:
        return 0.0
    fast_ma = closes.tail(fast).mean()
    slow_ma = closes.tail(slow).mean()
    if slow_ma == 0:
        return 0.0
    return (fast_ma - slow_ma) / slow_ma


def compute(macro_panel: pd.DataFrame) -> SignalScore:
    if macro_panel is None or macro_panel.empty:
        return SignalScore("macro", 0.0, 0.0, "no macro data")

    scores = []
    bits = []
    for name, sign in CORR_SIGN.items():
        sub = macro_panel[macro_panel["name"] == name].sort_values("ts")
        if sub.empty:
            continue
        mom = _momentum(sub["close"])
        # Normalise 2% fast-vs-slow gap to a unit score, apply correlation sign.
        contrib = sign * squash(mom / 0.02)
        scores.append(contrib)
        bits.append(f"{name} mom={mom:+.2%}→{contrib:+.2f}")

    if not scores:
        return SignalScore("macro", 0.0, 0.0, "macro panel empty after filter")
    avg = sum(scores) / len(scores)
    return SignalScore(
        name="macro",
        score=clip_score(avg),
        confidence=min(1.0, 0.3 + 0.1 * len(scores)),
        rationale="; ".join(bits),
        details={"components": bits},
    )
