"""Futures term-structure (annualised basis) signal.

Very high basis (contango) → speculative long premium → short bias as mean-reversion.
Deep backwardation → panic / supply stress → long bias.
Normal 5–15% APR contango → neutral / mild long bias (carry).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..indicators import annualized_basis, clip_score
from . import SignalScore

# Heuristic APR thresholds for annualised basis.
NEUTRAL_LOW = 0.00
NEUTRAL_HIGH = 0.15
RICH = 0.25
EXTREME_RICH = 0.50


def compute(book_summary: pd.DataFrame, spot_price: float) -> SignalScore:
    """book_summary: DataFrame returned by deribit.book_summary_by_currency(kind='future')."""
    if book_summary is None or book_summary.empty or spot_price <= 0:
        return SignalScore("basis", 0.0, 0.0, "missing futures book or spot")

    df = book_summary.copy()
    # Only dated futures, skip perpetual.
    df = df[df["instrument_name"].str.contains("-PERPETUAL") == False]
    if df.empty:
        return SignalScore("basis", 0.0, 0.0, "no dated futures")

    now = datetime.now(tz=timezone.utc)
    rows = []
    for _, r in df.iterrows():
        mid = r.get("mid_price") or r.get("mark_price") or r.get("last")
        if mid is None or pd.isna(mid):
            continue
        # Parse expiry from instrument name, e.g. BTC-28MAR25
        try:
            tail = r["instrument_name"].split("-")[1]
            expiry = datetime.strptime(tail, "%d%b%y").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        dte = (expiry - now).total_seconds() / 86400.0
        # Skip near-expiry futures — annualisation blows small premia into
        # unrepresentative APRs. 7d is the smallest horizon that typically
        # carries meaningful term premium.
        if dte < 7.0:
            continue
        apr = annualized_basis(float(mid), spot_price, dte)
        rows.append({"instrument": r["instrument_name"], "dte": dte, "apr": apr})

    if not rows:
        return SignalScore("basis", 0.0, 0.0, "no parsable dated futures")

    bdf = pd.DataFrame(rows).sort_values("dte")
    front = bdf.iloc[0]

    apr = float(front["apr"])
    # Map APR to a score. Richness → short bias.
    if apr >= EXTREME_RICH:
        score = -0.9
        tag = "extreme contango"
    elif apr >= RICH:
        score = -0.5
        tag = "rich contango"
    elif apr > NEUTRAL_HIGH:
        score = -0.2
        tag = "above-normal contango"
    elif apr >= NEUTRAL_LOW:
        score = 0.1
        tag = "normal contango (mild carry)"
    elif apr >= -NEUTRAL_HIGH:
        score = 0.3
        tag = "flat / mild backwardation"
    else:
        score = 0.8
        tag = "deep backwardation (stress)"

    score = clip_score(score)
    conf = 0.6 if len(rows) >= 2 else 0.4
    rationale = (
        f"front future {front['instrument']} @ {front['dte']:.1f}d → {apr:+.2%} APR basis — {tag}"
    )
    return SignalScore(
        name="basis",
        score=score,
        confidence=conf,
        rationale=rationale,
        details={"front_apr": apr, "points": rows},
    )
