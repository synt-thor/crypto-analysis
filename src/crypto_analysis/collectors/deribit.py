"""Deribit public REST wrappers.

All endpoints are unauthenticated public JSON-RPC methods exposed via GET.
Reference: https://docs.deribit.com/
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import DERIBIT_REST
from ..http import get_json


def _call(method: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{DERIBIT_REST}/public/{method}"
    payload = get_json(url, params=params or {})
    if "error" in payload:
        raise RuntimeError(f"Deribit error on {method}: {payload['error']}")
    return payload["result"]


def get_instruments(currency: str = "BTC", kind: str = "future", expired: bool = False) -> pd.DataFrame:
    result = _call("get_instruments", {"currency": currency, "kind": kind, "expired": str(expired).lower()})
    return pd.DataFrame(result)


def ticker(instrument_name: str) -> dict[str, Any]:
    return _call("ticker", {"instrument_name": instrument_name})


def book_summary_by_currency(currency: str = "BTC", kind: str = "future") -> pd.DataFrame:
    result = _call("get_book_summary_by_currency", {"currency": currency, "kind": kind})
    return pd.DataFrame(result)


def funding_rate_history(
    instrument_name: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    result = _call(
        "get_funding_rate_history",
        {
            "instrument_name": instrument_name,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
        },
    )
    df = pd.DataFrame(result)
    if not df.empty and "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def tradingview_chart_data(
    instrument_name: str, start_ms: int, end_ms: int, resolution: str = "60"
) -> pd.DataFrame:
    result = _call(
        "get_tradingview_chart_data",
        {
            "instrument_name": instrument_name,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": resolution,
        },
    )
    if result.get("status") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(result["ticks"], unit="ms", utc=True),
            "open": result["open"],
            "high": result["high"],
            "low": result["low"],
            "close": result["close"],
            "volume": result["volume"],
            "cost": result["cost"],
        }
    )
    df["instrument"] = instrument_name
    df["resolution"] = resolution
    return df


def tradingview_chart_data_chunked(
    instrument_name: str,
    start_ms: int,
    end_ms: int,
    resolution: str = "60",
    chunk_bars: int = 4000,
) -> pd.DataFrame:
    """Chunked OHLCV fetch for arbitrary time windows + fine resolutions.

    Deribit's tradingview endpoint returns up to ~5000 bars per call. For
    1-minute backtests over weeks this isn't enough — we walk forward in
    chunks of `chunk_bars` and concatenate.
    """
    res_to_minutes = {
        "1": 1, "3": 3, "5": 5, "10": 10, "15": 15, "30": 30,
        "60": 60, "120": 120, "180": 180, "360": 360, "720": 720, "1D": 1440,
    }
    minutes_per_bar = res_to_minutes.get(resolution, 60)
    chunk_ms = chunk_bars * minutes_per_bar * 60 * 1000

    cursor = start_ms
    frames: list[pd.DataFrame] = []
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        df = tradingview_chart_data(instrument_name, cursor, chunk_end, resolution)
        if not df.empty:
            frames.append(df)
        cursor = chunk_end
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def historical_volatility(currency: str = "BTC") -> pd.DataFrame:
    result = _call("get_historical_volatility", {"currency": currency})
    df = pd.DataFrame(result, columns=["timestamp_ms", "rv"])
    df["ts"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df


def last_trades_by_currency(currency: str = "BTC", count: int = 100, kind: str | None = None) -> pd.DataFrame:
    params: dict[str, Any] = {"currency": currency, "count": count}
    if kind:
        params["kind"] = kind
    result = _call("get_last_trades_by_currency", params)
    return pd.DataFrame(result.get("trades", []))


def last_liquidations(currency: str = "BTC", count: int = 1000) -> pd.DataFrame:
    """Recent trades flagged as liquidations on Deribit futures.

    Returns only rows where the Deribit 'liquidation' field is present.
    """
    trades = last_trades_by_currency(currency, count=count, kind="future")
    if trades.empty or "liquidation" not in trades.columns:
        return pd.DataFrame()
    liqs = trades[trades["liquidation"].notna() & (trades["liquidation"] != "")].copy()
    if liqs.empty:
        return liqs
    liqs["ts"] = pd.to_datetime(liqs["timestamp"], unit="ms", utc=True)
    return liqs


def order_book(instrument_name: str, depth: int = 20) -> dict[str, Any]:
    """Top-N bids/asks. Deribit default depth is 20."""
    return _call("get_order_book", {"instrument_name": instrument_name, "depth": depth})


def index_price(index_name: str = "btc_usd") -> float:
    """Deribit's BTC index price (volume-weighted across major spots).

    Reliable spot reference that works from any IP — no geo-blocking.
    """
    result = _call("get_index_price", {"index_name": index_name})
    return float(result.get("index_price") or 0.0)


def option_book_summary(currency: str = "BTC") -> pd.DataFrame:
    """All option chains with IVs. One row per option instrument.

    Columns of interest: instrument_name, mark_iv, bid_iv, ask_iv, mid_price,
    underlying_price, open_interest, mark_price, volume.
    Strike + expiry are parsed from instrument_name (e.g. BTC-27JUN25-60000-C).
    """
    result = _call("get_book_summary_by_currency", {"currency": currency, "kind": "option"})
    df = pd.DataFrame(result)
    if df.empty:
        return df

    # Parse strike and expiry from instrument name.
    parts = df["instrument_name"].str.split("-", expand=True)
    # Expected: [BTC, DDMMMYY, STRIKE, C|P]
    if parts.shape[1] < 4:
        return df
    df["expiry_str"] = parts[1]
    df["strike"] = pd.to_numeric(parts[2], errors="coerce")
    df["option_type"] = parts[3].str.upper()
    df["expiry"] = pd.to_datetime(df["expiry_str"], format="%d%b%y", utc=True, errors="coerce")
    return df
