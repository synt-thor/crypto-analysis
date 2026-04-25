"""Collect a single-point snapshot of all live public inputs and persist them."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from crypto_analysis.collectors import deribit, exchanges, onchain
from crypto_analysis.config import PERPETUAL
from crypto_analysis.storage import write_parquet, write_raw


def main() -> None:
    now = datetime.now(tz=timezone.utc)
    start_ms = int((now - timedelta(days=7)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    book = deribit.book_summary_by_currency("BTC", "future")
    write_raw("deribit", "book_summary", book.to_dict(orient="records"))
    write_parquet("deribit_book_summary", book)

    perp_ticker = deribit.ticker(PERPETUAL)
    write_raw("deribit", f"ticker_{PERPETUAL}", perp_ticker)

    funding = deribit.funding_rate_history(PERPETUAL, start_ms, end_ms)
    if not funding.empty:
        funding["funding_rate"] = funding.get("interest_1h", funding.get("funding_rate"))
        write_parquet("deribit_funding", funding)

    rv = deribit.historical_volatility("BTC")
    write_parquet("deribit_historical_volatility", rv)

    binance_spot = exchanges.binance_klines("BTCUSDT", "5m", limit=24)
    write_parquet("binance_spot_btc_5m", binance_spot)

    mempool = onchain.mempool_snapshot()
    fees = onchain.fees_recommended()
    hr = onchain.hashrate_3d()
    diff = onchain.difficulty_adjustment()
    write_raw("onchain", "snapshot", {
        "mempool": mempool, "fees": fees, "hashrate_3d": hr, "difficulty": diff,
    })
    onchain_df = pd.DataFrame([{
        "ts": now,
        "mempool_count": mempool.get("count"),
        "mempool_vsize": mempool.get("vsize"),
        "fastest_fee": fees.get("fastestFee"),
        # hashrate / difficulty are exa-scale and overflow int64 — store as float.
        "hashrate": float(hr.get("currentHashrate") or 0.0),
        "difficulty": float(hr.get("currentDifficulty") or 0.0),
    }])
    write_parquet("onchain_snapshot", onchain_df)

    print(f"snapshot @ {now.isoformat()} — futures={len(book)} rows, funding={len(funding)} rows")


if __name__ == "__main__":
    main()
