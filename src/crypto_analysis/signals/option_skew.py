"""25-delta risk reversal on the nearest tradeable BTC expiry.

RR25 = IV(25Δ call) − IV(25Δ put)
  positive → calls richer than puts → bullish positioning
  negative → puts richer than calls → hedging demand / bearish sentiment

We pick the nearest expiry with ≥ 7 days to expiry (front-week skew can be
noisy on low-OI series).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..greeks import GreeksInputs, delta as bs_delta
from ..indicators import clip_score, squash
from . import SignalScore

TARGET_DELTA = 0.25


def _annualise_days(d: float) -> float:
    return d / 365.0


def compute(options: pd.DataFrame, spot_price: float) -> SignalScore:
    if options is None or options.empty or spot_price <= 0:
        return SignalScore("option_skew", 0.0, 0.0, "no options or spot")

    df = options.dropna(subset=["expiry", "strike", "mark_iv", "option_type"]).copy()
    df = df[df["mark_iv"] > 0]
    if df.empty:
        return SignalScore("option_skew", 0.0, 0.0, "no valid options")

    now = datetime.now(tz=timezone.utc)
    df["dte_days"] = (df["expiry"] - now).dt.total_seconds() / 86400.0
    tradeable = df[df["dte_days"] >= 7].sort_values("dte_days")
    if tradeable.empty:
        return SignalScore("option_skew", 0.0, 0.0, "no expiry with dte>=7d")

    # Pick the single nearest expiry with ≥7d and at least 5 strikes on each side.
    for exp, grp in tradeable.groupby("expiry", sort=True):
        calls = grp[grp["option_type"] == "C"]
        puts = grp[grp["option_type"] == "P"]
        if len(calls) < 3 or len(puts) < 3:
            continue

        tau = _annualise_days(grp["dte_days"].iloc[0])
        # Deribit mark_iv is already in percent (e.g. 55.0 → 55%).
        def _d(row, is_call):
            return bs_delta(GreeksInputs(
                spot=spot_price, strike=float(row["strike"]), tau_years=tau,
                iv=float(row["mark_iv"]) / 100.0, is_call=is_call,
            ))

        calls = calls.assign(d=calls.apply(lambda r: _d(r, True), axis=1))
        puts = puts.assign(d=puts.apply(lambda r: _d(r, False), axis=1))

        # Closest to ±0.25 delta.
        call25 = calls.iloc[(calls["d"] - TARGET_DELTA).abs().argsort().iloc[0]]
        put25 = puts.iloc[(puts["d"] + TARGET_DELTA).abs().argsort().iloc[0]]

        iv_c = float(call25["mark_iv"])
        iv_p = float(put25["mark_iv"])
        rr = iv_c - iv_p     # in vol points (pp)

        # Normalise: 5 pp RR is already a strong reading for BTC.
        score = clip_score(squash(rr / 5.0))
        tag = (
            "calls richer (bullish skew)" if rr > 1 else
            "puts richer (hedging/bearish skew)" if rr < -1 else
            "flat skew"
        )
        rationale = (
            f"expiry {exp.date()} dte={grp['dte_days'].iloc[0]:.1f}d — "
            f"call25Δ IV={iv_c:.1f}% @K={call25['strike']:.0f}, "
            f"put25Δ IV={iv_p:.1f}% @K={put25['strike']:.0f}, "
            f"RR={rr:+.1f}pp — {tag}"
        )
        return SignalScore(
            name="option_skew",
            score=score,
            confidence=0.55,
            rationale=rationale,
            details={"rr_pp": rr, "iv_call25": iv_c, "iv_put25": iv_p,
                     "expiry": str(exp.date())},
        )

    return SignalScore("option_skew", 0.0, 0.1, "no expiry had enough strikes both sides")
