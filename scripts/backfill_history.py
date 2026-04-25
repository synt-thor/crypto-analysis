"""Backfill historical OHLCV, funding, and macro panel for backtesting."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from crypto_analysis.collectors import deribit, exchanges
from crypto_analysis.collectors.macro import macro_panel
from crypto_analysis.config import PERPETUAL
from crypto_analysis.storage import write_parquet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--resolution", default="60", help="Deribit TV resolution (e.g. 60 = 1h, 1D = daily)")
    args = ap.parse_args()

    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    perp_ohlcv = deribit.tradingview_chart_data(PERPETUAL, start_ms, end_ms, resolution=args.resolution)
    write_parquet("deribit_ohlcv", perp_ohlcv, partition_cols=["instrument", "resolution"])
    print(f"deribit perp ohlcv rows: {len(perp_ohlcv)}")

    funding = deribit.funding_rate_history(PERPETUAL, start_ms, end_ms)
    if not funding.empty:
        funding["funding_rate"] = funding.get("interest_1h", funding.get("funding_rate"))
        write_parquet("deribit_funding", funding)
    print(f"deribit funding rows: {len(funding)}")

    binance = exchanges.binance_klines("BTCUSDT", "1h", start_ms=start_ms, end_ms=end_ms, limit=1000)
    write_parquet("binance_spot_btc_1h", binance)
    print(f"binance spot 1h rows: {len(binance)}")

    mp = macro_panel(days=args.days)
    if not mp.empty:
        write_parquet("macro_panel", mp, partition_cols=["name"])
    print(f"macro panel rows: {len(mp)}")


if __name__ == "__main__":
    main()
