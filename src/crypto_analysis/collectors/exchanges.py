"""Public spot data from Binance / Coinbase + Binance perpetual funding."""

from __future__ import annotations

import pandas as pd

from ..config import BINANCE_SPOT_REST, COINBASE_REST
from ..http import get_json

BINANCE_FAPI = "https://fapi.binance.com"


def binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms
    rows = get_json(f"{BINANCE_SPOT_REST}/api/v3/klines", params)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col])
    df["symbol"] = symbol
    return df[["ts", "symbol", "open", "high", "low", "close", "volume", "quote_volume"]]


def binance_perp_funding(symbol: str = "BTCUSDT", limit: int = 500) -> pd.DataFrame:
    rows = get_json(
        f"{BINANCE_FAPI}/fapi/v1/fundingRate",
        {"symbol": symbol, "limit": limit},
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    return df


def coinbase_candles(
    product_id: str = "BTC-USD", granularity: int = 3600
) -> pd.DataFrame:
    rows = get_json(
        f"{COINBASE_REST}/products/{product_id}/candles",
        {"granularity": granularity},
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["product_id"] = product_id
    return df.sort_values("ts").reset_index(drop=True)
