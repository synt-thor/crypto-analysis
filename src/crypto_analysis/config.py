"""Endpoints, symbols, and defaults shared across collectors."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PARQUET = PROJECT_ROOT / "data" / "parquet"

DERIBIT_REST = "https://www.deribit.com/api/v2"
MEMPOOL_REST = "https://mempool.space/api"
BINANCE_SPOT_REST = "https://api.binance.com"
COINBASE_REST = "https://api.exchange.coinbase.com"

DEFAULT_CURRENCY = "BTC"
PERPETUAL = "BTC-PERPETUAL"

DEFAULT_OHLCV_RESOLUTION = "60"
DEFAULT_BACKFILL_DAYS = 180

HTTP_TIMEOUT_S = 20.0
HTTP_MAX_RETRIES = 4
