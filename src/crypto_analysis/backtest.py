"""Minimal vectorised backtest for the fusion signal.

Design:
  - Inputs are historical hourly tables: OHLCV (perp), funding history, basis
    history, macro panel. On-chain/news are not backtestable from free sources
    at the moment, so they are excluded from historical runs (weights
    redistributed automatically by engine.fuse when signals are missing).
  - For each timestamp we rebuild a minimal signal set (funding, basis, oi,
    spot_futures, macro) using data known AT THAT TIME.
  - Position: sign(decision) held for `hold_hours`, then re-evaluated.
  - Output: DataFrame of timestamps, decisions, forward returns, and
    aggregate stats (hit rate, avg return, sharpe, equity curve).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decision import decide
from .engine import fuse
from .signals import SignalScore
from .signals.basis import compute as basis_compute
from .signals.funding import compute as funding_compute
from .signals.oi import compute as oi_compute
from .signals.spot_futures import compute as spot_futures_compute


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    hit_rate: float
    avg_fwd_return: float
    sharpe: float
    equity_curve: pd.Series


def _funding_window(funding_df: pd.DataFrame, ts) -> pd.DataFrame:
    return funding_df[funding_df["ts"] <= ts].tail(168)  # last ~7 days


def _oi_window(ticker_hist: pd.DataFrame, ts) -> pd.DataFrame:
    return ticker_hist[ticker_hist["ts"] <= ts].tail(48)


def run(
    perp_ohlcv: pd.DataFrame,           # cols: ts, close (perp hourly)
    spot_ohlcv: pd.DataFrame,           # cols: ts, close (spot hourly)
    funding_df: pd.DataFrame,           # cols: ts, funding_rate
    ticker_hist: pd.DataFrame | None,   # cols: ts, mark_price, open_interest
    hold_hours: int = 8,
    step_hours: int = 4,
) -> BacktestResult:
    perp = perp_ohlcv.sort_values("ts").reset_index(drop=True)
    spot = spot_ohlcv.set_index("ts")["close"].sort_index()
    trades = []

    for i in range(100, len(perp) - hold_hours, step_hours):
        row = perp.iloc[i]
        ts = row["ts"]
        perp_close = float(row["close"])
        try:
            spot_close = float(spot.asof(ts))
        except Exception:
            continue
        if not np.isfinite(spot_close):
            continue

        signals: list[SignalScore] = []
        signals.append(funding_compute(_funding_window(funding_df, ts)))
        signals.append(spot_futures_compute(perp_close, spot_close))
        if ticker_hist is not None and not ticker_hist.empty:
            signals.append(oi_compute(_oi_window(ticker_hist, ts)))
        # basis needs a book summary; skip in backtest (no historical book).
        _ = basis_compute  # referenced so imports are purposeful

        eng = fuse(signals)
        dec = decide(eng)

        fwd_price = float(perp.iloc[i + hold_hours]["close"])
        fwd_ret = (fwd_price - perp_close) / perp_close
        direction = {"LONG": +1, "SHORT": -1, "NEUTRAL": 0}[dec.verdict]
        pnl = direction * fwd_ret

        trades.append({
            "ts": ts,
            "verdict": dec.verdict,
            "score": dec.score,
            "confidence": dec.confidence,
            "fwd_return": fwd_ret,
            "direction": direction,
            "pnl": pnl,
        })

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(tdf, float("nan"), float("nan"), float("nan"), empty)

    active = tdf[tdf["direction"] != 0]
    hit = float((active["pnl"] > 0).mean()) if not active.empty else float("nan")
    avg = float(active["pnl"].mean()) if not active.empty else float("nan")
    std = float(active["pnl"].std(ddof=0)) if not active.empty else float("nan")
    sharpe = float(avg / std * np.sqrt(252 * 24 / max(1, hold_hours))) if std and std > 0 else float("nan")
    eq = (1 + tdf["pnl"].fillna(0)).cumprod()
    eq.index = tdf["ts"]
    return BacktestResult(tdf, hit, avg, sharpe, eq)
