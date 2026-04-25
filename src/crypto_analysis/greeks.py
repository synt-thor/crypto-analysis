"""Black-Scholes Greeks for BTC options.

We assume zero risk-free rate for crypto denominated in BTC — Deribit options
are inverse-quoted in BTC so the standard r=0 approximation is close enough
for directional signal use (not for mark-making).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GreeksInputs:
    spot: float          # underlying price (USD)
    strike: float        # strike (USD)
    tau_years: float     # time to expiry in years
    iv: float            # implied vol as decimal (0.55 == 55%)
    is_call: bool


def _d1(inp: GreeksInputs) -> float:
    if inp.tau_years <= 0 or inp.iv <= 0 or inp.spot <= 0 or inp.strike <= 0:
        return 0.0
    return (
        math.log(inp.spot / inp.strike) + 0.5 * inp.iv * inp.iv * inp.tau_years
    ) / (inp.iv * math.sqrt(inp.tau_years))


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def delta(inp: GreeksInputs) -> float:
    if inp.tau_years <= 0 or inp.iv <= 0:
        return 0.0
    d1 = _d1(inp)
    return _cdf(d1) if inp.is_call else (_cdf(d1) - 1.0)


def gamma(inp: GreeksInputs) -> float:
    """Standard BS gamma (per unit of underlying)."""
    if inp.tau_years <= 0 or inp.iv <= 0 or inp.spot <= 0:
        return 0.0
    d1 = _d1(inp)
    return _phi(d1) / (inp.spot * inp.iv * math.sqrt(inp.tau_years))
