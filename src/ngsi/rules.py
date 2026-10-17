"""47 CFR 25.261 spectrum-sharing rules.

* **25.261(c)** - band-splitting: where the two NGSO systems cannot agree, each is
  entitled to half the available spectrum (in practice the band is divided so
  each system gets 50% of the assignable bandwidth).
* **25.261(d)** - degraded-throughput protection: a system must limit the
  short-term increase in the victim link's equivalent noise temperature to
  ``Delta T/T`` of 3% (long-term, 0.4% of the time) and 6% (short-term).

Constants are defined here so experiments never hard-code rule numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import DTT_LONG_TERM, LONG_TERM_EXCEEDANCE_FRACTION
from .constants import DTT_THRESHOLD as DTT_SHORT_TERM

# --- 25.261(c) band-splitting ----------------------------------------------
BAND_SPLIT_FRACTION: float = 0.5

__all__ = [
    "DTT_LONG_TERM",
    "DTT_SHORT_TERM",
    "LONG_TERM_EXCEEDANCE_FRACTION",
    "BAND_SPLIT_FRACTION",
    "BandSplit",
    "band_split",
    "exceeds_long_term",
    "exceeds_short_term",
]


@dataclass(frozen=True)
class BandSplit:
    """Outcome of applying the 25.261(c) band split to a total bandwidth."""

    total_mhz: float
    per_system_mhz: float
    fraction: float


def band_split(total_bandwidth_mhz: float, fraction: float = BAND_SPLIT_FRACTION) -> BandSplit:
    """Split ``total_bandwidth_mhz`` between two systems (default 50/50)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    if total_bandwidth_mhz <= 0.0:
        raise ValueError("total_bandwidth_mhz must be positive")
    per = total_bandwidth_mhz * fraction
    return BandSplit(total_mhz=total_bandwidth_mhz, per_system_mhz=per, fraction=fraction)


def exceeds_long_term(dtt: float) -> bool:
    """True when an event exceeds the 3% long-term Delta T/T criterion."""
    return float(dtt) > DTT_LONG_TERM


def exceeds_short_term(dtt: float) -> bool:
    """True when an event exceeds the 6% short-term Delta T/T criterion."""
    return float(dtt) > DTT_SHORT_TERM
