"""Derived calculations used across signals and notebooks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annualized_basis(futures_price: float, spot_price: float, days_to_expiry: float) -> float:
    """Simple annualized basis for a dated future.

    Returns NaN for perpetuals (days_to_expiry <= 0).
    """
    if days_to_expiry <= 0 or spot_price <= 0:
        return float("nan")
    return ((futures_price / spot_price) - 1.0) * (365.0 / days_to_expiry)


def funding_apr(funding_rate_8h: float) -> float:
    """Convert 8h funding rate to annualised percentage."""
    return funding_rate_8h * 3.0 * 365.0


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    rolling = series.rolling(window, min_periods=max(5, window // 3))
    mu = rolling.mean()
    sigma = rolling.std(ddof=0)
    return (series - mu) / sigma.replace(0, np.nan)


def clip_score(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if math.isnan(x):
        return 0.0
    return max(lo, min(hi, x))


def squash(x: float, scale: float = 1.0) -> float:
    """tanh squash for unbounded inputs into [-1, 1]."""
    if math.isnan(x):
        return 0.0
    return math.tanh(x / scale)
