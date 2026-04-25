"""News & geopolitics signal — structured wrapper around a manually-supplied brief.

We deliberately do NOT call any search API from inside this module: news→number
is lossy and the signal needs human review. The workflow is:

1. Operator (or a WebSearch-enabled agent in a notebook cell) gathers recent
   headlines and hands in a `NewsBrief` dict:
      {
        "window_hours": 24,
        "events": [
           {"headline": "...", "bias": "long|short|neutral", "weight": 0.0-1.0,
            "rationale": "..."}
        ],
        "macro_calendar": ["FOMC 2026-05-01", ...]
      }
2. This module maps that structured brief to a SignalScore.
3. Because news is noisy, the signal is capped at ±0.6 and confidence at 0.5.
"""

from __future__ import annotations

from typing import Any

from ..indicators import clip_score
from . import SignalScore

MAX_NEWS_SCORE = 0.6
MAX_NEWS_CONF = 0.5

_BIAS_SIGN = {"long": +1, "bull": +1, "short": -1, "bear": -1, "neutral": 0}


def compute(brief: dict[str, Any] | None) -> SignalScore:
    if not brief or not brief.get("events"):
        return SignalScore("news", 0.0, 0.0, "no news brief supplied")

    total_weight = 0.0
    numerator = 0.0
    reasons = []
    for ev in brief["events"]:
        sign = _BIAS_SIGN.get(str(ev.get("bias", "neutral")).lower(), 0)
        w = float(ev.get("weight", 0.5))
        w = max(0.0, min(1.0, w))
        numerator += sign * w
        total_weight += w
        if sign != 0:
            reasons.append(f"{'+' if sign>0 else '-'}{w:.2f}: {ev.get('headline','?')[:80]}")

    if total_weight == 0:
        return SignalScore("news", 0.0, 0.1, "events carry no weight")

    raw = numerator / total_weight  # in [-1, +1]
    score = clip_score(raw * MAX_NEWS_SCORE, -MAX_NEWS_SCORE, MAX_NEWS_SCORE)
    conf = min(MAX_NEWS_CONF, 0.2 + 0.05 * len(brief["events"]))

    cal = brief.get("macro_calendar") or []
    if cal:
        reasons.append(f"calendar: {', '.join(cal[:3])}")

    rationale = " | ".join(reasons) if reasons else "news neutral"
    return SignalScore(
        name="news",
        score=score,
        confidence=conf,
        rationale=rationale,
        details={"raw": raw, "n_events": len(brief["events"])},
    )
