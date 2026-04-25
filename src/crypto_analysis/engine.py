"""Signal fusion engine.

Combines individual SignalScore objects into a single directional score in
[-1, +1] with a confidence estimate. Weights are literature/expert defaults
and can be overridden at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .signals import SignalScore

# Defaults sized for a SHORT-TERM (hours–days) BTC perp trader.
# News is intentionally low because news→number conversion is noisy.
DEFAULT_WEIGHTS: dict[str, float] = {
    "funding":       0.16,
    "spot_futures":  0.12,
    "option_skew":   0.11,
    "basis":         0.09,
    "oi":            0.09,
    "macro":         0.08,
    "iv_skew":       0.07,
    "gex":           0.06,
    "liquidations":  0.06,
    "orderbook":     0.05,
    "onchain":       0.06,
    "news":          0.05,
}

# Short-term (hours): price microstructure + forced flow dominates.
WEIGHTS_ST: dict[str, float] = {
    "orderbook":     0.18,
    "liquidations":  0.16,
    "spot_futures":  0.16,
    "funding":       0.14,
    "oi":            0.12,
    "option_skew":   0.08,
    "iv_skew":       0.06,
    "gex":           0.04,
    "basis":         0.03,
    "macro":         0.02,
    "onchain":       0.01,
    "news":          0.00,
}

# Medium-term (days): positioning & derivatives structure.
WEIGHTS_MT: dict[str, float] = {
    "funding":       0.17,
    "option_skew":   0.15,
    "basis":         0.13,
    "spot_futures":  0.11,
    "oi":            0.10,
    "gex":           0.09,
    "iv_skew":       0.08,
    "liquidations":  0.05,
    "macro":         0.07,
    "orderbook":     0.02,
    "onchain":       0.02,
    "news":          0.01,
}

# Long-term (weeks): macro + on-chain + deep basis.
WEIGHTS_LT: dict[str, float] = {
    "macro":         0.24,
    "onchain":       0.18,
    "basis":         0.14,
    "iv_skew":       0.10,
    "option_skew":   0.08,
    "funding":       0.08,
    "gex":           0.06,
    "news":          0.05,
    "spot_futures":  0.04,
    "oi":            0.02,
    "liquidations":  0.00,
    "orderbook":     0.01,
}

TIMEFRAME_WEIGHTS = {"ST": WEIGHTS_ST, "MT": WEIGHTS_MT, "LT": WEIGHTS_LT}


@dataclass
class EngineResult:
    score: float                  # final weighted score in [-1, +1]
    confidence: float             # 0..1
    contributions: list[dict] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)


def fuse(
    signals: list[SignalScore],
    weights: dict[str, float] | None = None,
) -> EngineResult:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    # Normalise only over signals we actually have (skip zero-confidence ones).
    usable = [s for s in signals if s.confidence > 0 and s.name in w]
    if not usable:
        return EngineResult(0.0, 0.0, [], w)

    # Effective weight = base weight * confidence, then renormalise.
    raw = [(s, w[s.name] * s.confidence) for s in usable]
    total_w = sum(wt for _, wt in raw)
    if total_w == 0:
        return EngineResult(0.0, 0.0, [], w)

    score = sum(s.score * wt / total_w for s, wt in raw)
    # Confidence: average confidence weighted by base weight mass present.
    present_base_mass = sum(w[s.name] for s in usable)
    conf = sum(s.confidence * w[s.name] for s in usable) / max(present_base_mass, 1e-9)

    contribs = [
        {
            "name": s.name,
            "score": s.score,
            "confidence": s.confidence,
            "base_weight": w[s.name],
            "effective_weight": wt / total_w,
            "contribution": s.score * wt / total_w,
            "rationale": s.rationale,
        }
        for s, wt in raw
    ]
    contribs.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return EngineResult(score=score, confidence=conf, contributions=contribs, weights=w)
