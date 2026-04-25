"""Raw JSON dumps + Parquet read/write + DuckDB view helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_PARQUET, DATA_RAW


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def write_raw(source: str, endpoint: str, payload: Any) -> Path:
    now = _now_utc()
    target = (
        DATA_RAW
        / source
        / f"{now:%Y}"
        / f"{now:%m}"
        / f"{now:%d}"
        / f"{endpoint}_{now:%H%M%S}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, default=str))
    return target


def write_parquet(
    table: str,
    df: pd.DataFrame,
    partition_cols: list[str] | None = None,
) -> Path:
    if df.empty:
        return DATA_PARQUET / table
    target_dir = DATA_PARQUET / table
    target_dir.mkdir(parents=True, exist_ok=True)
    if partition_cols:
        df.to_parquet(
            target_dir,
            partition_cols=partition_cols,
            index=False,
            engine="pyarrow",
        )
        return target_dir
    file_path = target_dir / f"part_{_now_utc():%Y%m%dT%H%M%S}.parquet"
    df.to_parquet(file_path, index=False, engine="pyarrow")
    return file_path


def read_parquet(table: str) -> pd.DataFrame:
    target = DATA_PARQUET / table
    if not target.exists():
        return pd.DataFrame()
    return pd.read_parquet(target, engine="pyarrow")


def duckdb_conn():
    import duckdb

    con = duckdb.connect(database=":memory:")
    for child in DATA_PARQUET.glob("*"):
        if child.is_dir():
            path = str(child / "**" / "*.parquet")
            con.execute(
                f"CREATE OR REPLACE VIEW {child.name} AS SELECT * FROM read_parquet('{path}')"
            )
    return con
