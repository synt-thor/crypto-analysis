"""Map engine score → LONG / SHORT / NEUTRAL with confidence-gated thresholds."""

from __future__ import annotations

from dataclasses import dataclass

from .engine import EngineResult, TIMEFRAME_WEIGHTS, fuse
from .signals import SignalScore

LONG_THRESHOLD = 0.15
SHORT_THRESHOLD = -0.15
MIN_CONFIDENCE = 0.25


@dataclass
class Decision:
    verdict: str          # "LONG" | "SHORT" | "NEUTRAL"
    score: float
    confidence: float
    reason: str


def decide(result: EngineResult) -> Decision:
    if result.confidence < MIN_CONFIDENCE:
        return Decision(
            "NEUTRAL",
            result.score,
            result.confidence,
            f"confidence {result.confidence:.2f} below {MIN_CONFIDENCE} — stand aside",
        )
    if result.score >= LONG_THRESHOLD:
        return Decision(
            "LONG",
            result.score,
            result.confidence,
            f"score {result.score:+.2f} ≥ {LONG_THRESHOLD} with conf {result.confidence:.2f}",
        )
    if result.score <= SHORT_THRESHOLD:
        return Decision(
            "SHORT",
            result.score,
            result.confidence,
            f"score {result.score:+.2f} ≤ {SHORT_THRESHOLD} with conf {result.confidence:.2f}",
        )
    return Decision(
        "NEUTRAL",
        result.score,
        result.confidence,
        f"score {result.score:+.2f} inside ±{LONG_THRESHOLD} band",
    )


@dataclass
class MultiTimeframeDecision:
    st: Decision
    mt: Decision
    lt: Decision
    st_result: EngineResult
    mt_result: EngineResult
    lt_result: EngineResult


def decide_multi(signals: list[SignalScore]) -> MultiTimeframeDecision:
    st_res = fuse(signals, weights=TIMEFRAME_WEIGHTS["ST"])
    mt_res = fuse(signals, weights=TIMEFRAME_WEIGHTS["MT"])
    lt_res = fuse(signals, weights=TIMEFRAME_WEIGHTS["LT"])
    return MultiTimeframeDecision(
        st=decide(st_res),
        mt=decide(mt_res),
        lt=decide(lt_res),
        st_result=st_res,
        mt_result=mt_res,
        lt_result=lt_res,
    )


def format_multi_report(multi: MultiTimeframeDecision) -> str:
    lines = ["Multi-timeframe decision", "=" * 54]
    for label, dec, res in [
        ("ST (hours)", multi.st, multi.st_result),
        ("MT (days) ", multi.mt, multi.mt_result),
        ("LT (weeks)", multi.lt, multi.lt_result),
    ]:
        lines.append(
            f"{label}: {dec.verdict:<7} score={dec.score:+.3f}  conf={dec.confidence:.2f}"
        )
        top3 = res.contributions[:3]
        for c in top3:
            lines.append(
                f"   • {c['name']:<13} {c['contribution']:+.3f}  ({c['rationale'][:70]})"
            )
        lines.append("")
    lines.append(
        "Disclaimer: probabilistic aid only. Past edges do not guarantee future "
        "performance. Always apply risk management independent of this output."
    )
    return "\n".join(lines)


def format_report(result: EngineResult, decision: Decision) -> str:
    lines = [
        f"Verdict: {decision.verdict}  (score={decision.score:+.3f}, confidence={decision.confidence:.2f})",
        f"Reason:  {decision.reason}",
        "",
        "Signal contributions (ordered by magnitude):",
    ]
    for c in result.contributions:
        lines.append(
            f"  • {c['name']:<12} score={c['score']:+.2f}  conf={c['confidence']:.2f}  "
            f"w_eff={c['effective_weight']:.2f}  contrib={c['contribution']:+.3f}"
        )
        lines.append(f"      └─ {c['rationale']}")
    lines.append("")
    lines.append(
        "Disclaimer: probabilistic aid only. Past edges do not guarantee future "
        "performance. Always apply risk management independent of this output."
    )
    return "\n".join(lines)
