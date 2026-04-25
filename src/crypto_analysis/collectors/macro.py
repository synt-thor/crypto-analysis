"""Macro quotes (DXY, SPY, 10Y yield, VIX, Gold) via yfinance — free, no key."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",   # US Dollar Index
    "SPY": "SPY",        # S&P 500 ETF
    "QQQ": "QQQ",        # Nasdaq 100 ETF
    "VIX": "^VIX",       # Fear index
    "TNX": "^TNX",       # 10-year treasury yield x10
    "GOLD": "GC=F",      # Gold futures
}


def history(ticker: str, days: int = 90) -> pd.DataFrame:
    """Lazy-import yfinance so the rest of the package stays import-safe."""
    import yfinance as yf

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    df = yf.download(
        ticker, start=start.date(), end=end.date(),
        progress=False, auto_adjust=False,
    )
    if df.empty:
        return df
    df = df.reset_index().rename(columns=str.lower)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["ts"] = pd.to_datetime(df["date"], utc=True)
    df["ticker"] = ticker
    return df[["ts", "ticker", "open", "high", "low", "close", "volume"]]


def macro_panel(days: int = 90) -> pd.DataFrame:
    frames = []
    for name, t in MACRO_TICKERS.items():
        try:
            df = history(t, days=days)
        except Exception:
            continue
        if df.empty:
            continue
        df["name"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
