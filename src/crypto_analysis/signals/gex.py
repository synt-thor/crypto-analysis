"""Dealer gamma exposure proxy from Deribit option open interest.

We do not know dealer inventory directly. The common convention is to sum
gamma-weighted OI separately for calls and puts and treat the normalised
difference as gamma-weighted sentiment:

    net = (call_gamma_oi − put_gamma_oi) / (call_gamma_oi + put_gamma_oi)

    net > 0 : call-heavy positioning — bullish lean
    net < 0 : put-heavy positioning — defensive / bearish lean

This is NOT a clean "market gamma" regime indicator; it is a bias proxy.
Confidence is kept moderate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..greeks import GreeksInputs, gamma as bs_gamma
from ..indicators import clip_score
from . import SignalScore


def compute(options: pd.DataFrame, spot_price: float, dte_max: int = 60) -> SignalScore:
    if options is None or options.empty or spot_price <= 0:
        return SignalScore("gex", 0.0, 0.0, "no option data or spot")

    df = options.dropna(subset=["expiry", "strike", "mark_iv", "option_type", "open_interest"]).copy()
    df = df[(df["mark_iv"] > 0) & (df["open_interest"] > 0)]
    if df.empty:
        return SignalScore("gex", 0.0, 0.0, "no options with IV & OI")

    now = datetime.now(tz=timezone.utc)
    df["dte_days"] = (df["expiry"] - now).dt.total_seconds() / 86400.0
    df = df[(df["dte_days"] > 0) & (df["dte_days"] <= dte_max)]
    if df.empty:
        return SignalScore("gex", 0.0, 0.1, f"no expiries within {dte_max}d")

    def _gamma(row):
        return bs_gamma(GreeksInputs(
            spot=spot_price, strike=float(row["strike"]),
            tau_years=row["dte_days"] / 365.0,
            iv=float(row["mark_iv"]) / 100.0,
            is_call=row["option_type"] == "C",
        ))

    df["gamma"] = df.apply(_gamma, axis=1)
    df["gamma_oi"] = df["gamma"] * df["open_interest"]

    call_g = float(df.loc[df["option_type"] == "C", "gamma_oi"].sum())
    put_g = float(df.loc[df["option_type"] == "P", "gamma_oi"].sum())
    tot = call_g + put_g
    if tot <= 0:
        return SignalScore("gex", 0.0, 0.1, "gamma-weighted OI is zero")

    net = (call_g - put_g) / tot
    score = clip_score(net * 0.7)  # cap because dealer direction is assumption
    conf = 0.4

    tag = (
        "call-heavy positioning" if net > 0.15 else
        "put-heavy positioning" if net < -0.15 else
        "balanced"
    )
    rationale = (
        f"call γ-OI={call_g:,.3f}, put γ-OI={put_g:,.3f}, "
        f"net={net:+.2%} — {tag} (≤{dte_max}d expiries)"
    )
    return SignalScore(
        name="gex",
        score=score,
        confidence=conf,
        rationale=rationale,
        details={"call_gamma_oi": call_g, "put_gamma_oi": put_g, "net": net},
    )
