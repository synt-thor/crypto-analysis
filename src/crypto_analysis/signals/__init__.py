"""Directional signals for BTC futures.

Every signal module exposes a `compute(...)` function returning a `SignalScore`
with fields (name, score in [-1, +1], confidence in [0, 1], rationale).
"""

from dataclasses import dataclass, field


@dataclass
class SignalScore:
    name: str
    score: float            # -1.0 strong short ... +1.0 strong long
    confidence: float       # 0.0 .. 1.0 (data sufficiency / robustness)
    rationale: str
    details: dict = field(default_factory=dict)
