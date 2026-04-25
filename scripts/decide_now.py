"""Compute a live LONG/SHORT/NEUTRAL decision using all public inputs.

News brief is optional — pass --news-json path/to/brief.json if available.
See src/crypto_analysis/signals/news.py for the expected schema.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from crypto_analysis.collectors import deribit, exchanges, onchain
from crypto_analysis.collectors.macro import macro_panel
from crypto_analysis.config import PERPETUAL
from crypto_analysis.decision import decide, decide_multi, format_multi_report, format_report
from crypto_analysis.engine import fuse
from crypto_analysis.signals import SignalScore
from crypto_analysis.signals.basis import compute as basis_compute
from crypto_analysis.signals.funding import compute as funding_compute
from crypto_analysis.signals.gex import compute as gex_compute
from crypto_analysis.signals.iv_skew import compute as iv_skew_compute
from crypto_analysis.signals.liquidations import compute as liquidations_compute
from crypto_analysis.signals.macro import compute as macro_compute
from crypto_analysis.signals.news import compute as news_compute
from crypto_analysis.signals.oi import compute as oi_compute
from crypto_analysis.signals.onchain import compute as onchain_compute
from crypto_analysis.signals.option_skew import compute as option_skew_compute
from crypto_analysis.signals.orderbook import compute as orderbook_compute
from crypto_analysis.signals.spot_futures import compute as spot_futures_compute


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news-json", type=Path, default=None)
    args = ap.parse_args()

    now = datetime.now(tz=timezone.utc)
    start_ms = int((now - timedelta(days=7)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    # --- Live data pulls --------------------------------------------------
    book = deribit.book_summary_by_currency("BTC", "future")
    perp_ticker = deribit.ticker(PERPETUAL)
    perp_mark = float(perp_ticker.get("mark_price") or perp_ticker.get("last_price") or 0)
    rv = deribit.historical_volatility("BTC")
    funding = deribit.funding_rate_history(PERPETUAL, start_ms, end_ms)
    perp_ob = deribit.order_book(PERPETUAL, depth=20)
    options = deribit.option_book_summary("BTC")
    liq_trades = deribit.last_liquidations("BTC", count=1000)
    if not funding.empty:
        funding["funding_rate"] = funding.get("interest_1h", funding.get("funding_rate"))

    binance_spot = exchanges.binance_klines("BTCUSDT", "1m", limit=5)
    spot_price = float(binance_spot["close"].iloc[-1]) if not binance_spot.empty else 0.0

    mempool = onchain.mempool_snapshot()
    fees = onchain.fees_recommended()
    hr = onchain.hashrate_3d()
    mp = macro_panel(days=45)

    # --- News brief (optional) -------------------------------------------
    news_brief = None
    if args.news_json and args.news_json.exists():
        news_brief = json.loads(args.news_json.read_text())

    # --- Build ticker history from the single live point -----------------
    ticker_hist = pd.DataFrame([{
        "ts": now,
        "mark_price": perp_mark,
        "open_interest": perp_ticker.get("open_interest", 0),
    }])

    # Get ATM IV from an option if possible: pick near-expiry ATM call.
    atm_iv_val = None
    opts = deribit.get_instruments("BTC", "option")
    if not opts.empty:
        opts["expiration_ts"] = pd.to_datetime(opts["expiration_timestamp"], unit="ms", utc=True)
        soon = opts[opts["expiration_ts"] > now].sort_values("expiration_ts").head(30)
        if not soon.empty and spot_price:
            soon = soon.assign(abs_delta=(soon["strike"] - spot_price).abs())
            near_atm = soon.sort_values(["expiration_ts", "abs_delta"]).iloc[0]
            try:
                atm_ticker = deribit.ticker(near_atm["instrument_name"])
                atm_iv_val = float(atm_ticker.get("mark_iv") or 0) or None
            except Exception:
                atm_iv_val = None

    # --- Compute signals --------------------------------------------------
    signals: list[SignalScore] = [
        funding_compute(funding),
        basis_compute(book, spot_price),
        oi_compute(ticker_hist),        # single-point: low confidence by design
        iv_skew_compute(rv, atm_iv_val),
        spot_futures_compute(perp_mark, spot_price),
        orderbook_compute(perp_ob, levels=10),
        option_skew_compute(options, spot_price),
        gex_compute(options, spot_price, dte_max=60),
        liquidations_compute(liq_trades),
        onchain_compute(mempool, fees, hr),
        macro_compute(mp),
        news_compute(news_brief),
    ]

    result = fuse(signals)
    d = decide(result)
    print(format_report(result, d))
    print()
    print(format_multi_report(decide_multi(signals)))


if __name__ == "__main__":
    main()
