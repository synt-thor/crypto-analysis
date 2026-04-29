"""Backtest with optional fees, slippage, stop-loss, and take-profit.

Design:
  - Inputs are historical bar tables: OHLCV (perp + spot), funding history.
  - For each timestamp we rebuild the backtestable subset of signals
    (funding, spot_futures, oi when available) using data known AT THAT TIME.
  - Position: sign(decision) held until either max-hold reached, stop-loss
    hit, or take-profit hit. Walk forward bar-by-bar checking high/low.
  - Round-trip cost = 2 × (fee_bps + slippage_bps) charged once per active
    trade (NEUTRAL verdicts cost nothing).
  - Output: DataFrame of trades with realised PnL, plus aggregate stats and
    a multiplier-based equity curve.

`bars_per_hour` lets the same code drive 1m / 5m / 15m / 1h backtests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decision import decide
from .engine import fuse
from .signals import SignalScore
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
    # New aggregate fields
    total_return: float = 0.0          # final_equity - 1.0
    max_drawdown: float = 0.0          # min equity / running peak - 1
    win_count: int = 0
    loss_count: int = 0
    neutral_count: int = 0
    sl_hits: int = 0
    tp_hits: int = 0
    timeout_exits: int = 0


def _funding_window(funding_df: pd.DataFrame, ts) -> pd.DataFrame:
    return funding_df[funding_df["ts"] <= ts].tail(168)


def _oi_window(ticker_hist: pd.DataFrame, ts) -> pd.DataFrame:
    return ticker_hist[ticker_hist["ts"] <= ts].tail(48)


def _simulate_trade(
    perp: pd.DataFrame,
    entry_idx: int,
    direction: int,
    sl_pct: float | None,
    tp_pct: float | None,
    max_hold_bars: int,
) -> tuple[float, str, int]:
    """Walk forward bar-by-bar until SL / TP / max-hold.

    Returns (gross_pnl_fraction, exit_reason, hold_bars_used).
    `gross_pnl_fraction` excludes fees — caller subtracts cost.
    """
    if direction == 0:
        return 0.0, "neutral", 0

    entry_price = float(perp.iloc[entry_idx]["close"])
    if entry_price <= 0:
        return 0.0, "invalid_entry", 0

    sl_price = (entry_price * (1.0 - sl_pct * direction)) if sl_pct else None
    tp_price = (entry_price * (1.0 + tp_pct * direction)) if tp_pct else None

    end_idx = min(entry_idx + max_hold_bars, len(perp) - 1)
    for j in range(entry_idx + 1, end_idx + 1):
        bar = perp.iloc[j]
        high = float(bar.get("high", bar["close"]))
        low = float(bar.get("low", bar["close"]))

        if direction > 0:  # LONG
            if sl_price is not None and low <= sl_price:
                return -sl_pct, "stop_loss", j - entry_idx
            if tp_price is not None and high >= tp_price:
                return tp_pct, "take_profit", j - entry_idx
        else:  # SHORT
            if sl_price is not None and high >= sl_price:
                return -sl_pct, "stop_loss", j - entry_idx
            if tp_price is not None and low <= tp_price:
                return tp_pct, "take_profit", j - entry_idx

    # Max-hold: exit at close of last bar in window.
    exit_price = float(perp.iloc[end_idx]["close"])
    pnl = direction * (exit_price - entry_price) / entry_price
    return pnl, "timeout", end_idx - entry_idx


def run(
    perp_ohlcv: pd.DataFrame,
    spot_ohlcv: pd.DataFrame,
    funding_df: pd.DataFrame,
    ticker_hist: pd.DataFrame | None,
    hold_hours: float = 8,
    step_hours: float = 4,
    bars_per_hour: int = 1,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> BacktestResult:
    """Run backtest.

    bars_per_hour:
        1   = 1h bars (default, matches old behaviour)
        4   = 15m bars
        12  = 5m  bars
        60  = 1m  bars

    fee_bps + slippage_bps: round-trip cost = 2 × (fee + slippage) bps.
    Deribit perp taker fee is ~5 bps so default mimics taker-only execution.

    stop_loss_pct / take_profit_pct: e.g. 0.005 = 0.5%. None disables.
    """
    perp = perp_ohlcv.sort_values("ts").reset_index(drop=True)
    spot = spot_ohlcv.set_index("ts")["close"].sort_index()

    hold_bars = max(1, int(hold_hours * bars_per_hour))
    step_bars = max(1, int(step_hours * bars_per_hour))
    warmup_bars = max(100, int(24 * bars_per_hour))  # at least 1 day of warmup

    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10000.0

    trades = []
    for i in range(warmup_bars, len(perp) - hold_bars, step_bars):
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
        # basis / option_skew / iv_skew / gex / orderbook / liquidations / macro / news
        # are not backtested here — engine.fuse re-normalises around missing signals.

        eng = fuse(signals)
        dec = decide(eng)
        direction = {"LONG": +1, "SHORT": -1, "NEUTRAL": 0}[dec.verdict]

        gross_pnl, exit_reason, bars_held = _simulate_trade(
            perp, i, direction,
            sl_pct=stop_loss_pct, tp_pct=take_profit_pct,
            max_hold_bars=hold_bars,
        )

        net_pnl = gross_pnl - (round_trip_cost if direction != 0 else 0.0)

        trades.append({
            "ts": ts,
            "verdict": dec.verdict,
            "score": dec.score,
            "confidence": dec.confidence,
            "direction": direction,
            "gross_pnl": gross_pnl,
            "fees_paid": (round_trip_cost if direction != 0 else 0.0),
            "pnl": net_pnl,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
        })

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(tdf, float("nan"), float("nan"), float("nan"), empty)

    active = tdf[tdf["direction"] != 0]
    hit = float((active["pnl"] > 0).mean()) if not active.empty else float("nan")
    avg = float(active["pnl"].mean()) if not active.empty else float("nan")
    std = float(active["pnl"].std(ddof=0)) if not active.empty else float("nan")
    bars_per_year = 252 * 24 * bars_per_hour
    sharpe = (
        float(avg / std * np.sqrt(bars_per_year / max(1, hold_bars)))
        if std and std > 0 else float("nan")
    )

    eq = (1.0 + tdf["pnl"].fillna(0)).cumprod()
    eq.index = tdf["ts"]
    total_return = float(eq.iloc[-1] - 1.0) if not eq.empty else 0.0

    # Max drawdown on equity curve.
    running_peak = eq.cummax()
    drawdown = (eq / running_peak) - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    win = int((active["pnl"] > 0).sum()) if not active.empty else 0
    lose = int((active["pnl"] < 0).sum()) if not active.empty else 0
    neutral = int((tdf["direction"] == 0).sum())
    sl = int((tdf["exit_reason"] == "stop_loss").sum())
    tp = int((tdf["exit_reason"] == "take_profit").sum())
    timeout = int((tdf["exit_reason"] == "timeout").sum())

    return BacktestResult(
        trades=tdf,
        hit_rate=hit,
        avg_fwd_return=avg,
        sharpe=sharpe,
        equity_curve=eq,
        total_return=total_return,
        max_drawdown=max_dd,
        win_count=win,
        loss_count=lose,
        neutral_count=neutral,
        sl_hits=sl,
        tp_hits=tp,
        timeout_exits=timeout,
    )
