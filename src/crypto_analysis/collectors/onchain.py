"""On-chain snapshots from mempool.space (no key required)."""

from __future__ import annotations

from typing import Any

from ..config import MEMPOOL_REST
from ..http import get_json


def mempool_snapshot() -> dict[str, Any]:
    return get_json(f"{MEMPOOL_REST}/mempool")


def hashrate_3d() -> dict[str, Any]:
    return get_json(f"{MEMPOOL_REST}/v1/mining/hashrate/3d")


def hashrate_1y() -> dict[str, Any]:
    return get_json(f"{MEMPOOL_REST}/v1/mining/hashrate/1y")


def fees_recommended() -> dict[str, Any]:
    return get_json(f"{MEMPOOL_REST}/v1/fees/recommended")


def difficulty_adjustment() -> dict[str, Any]:
    return get_json(f"{MEMPOOL_REST}/v1/difficulty-adjustment")


def blocks_tip_height() -> int:
    return int(get_json(f"{MEMPOOL_REST}/blocks/tip/height"))
