"""47 CFR 25.261 spectrum-sharing rules.

* **25.261(c)** - default procedure: absent coordination, whenever ``Delta T/T``
  at a receiver exceeds **6 percent** because of another system's emissions in a
  commonly authorized band, the band is divided ``1/n`` among the ``n`` affected
  networks of the same processing round, and the affected stations may use only
  their selected ``1/n`` **while the 6 percent threshold is exceeded**.  The 6
  percent ``Delta T/T`` value therefore *defines* the in-line event that triggers
  band splitting.
* **25.261(d)** - protection of earlier-round systems: a compatibility showing
  must demonstrate, via a degraded-throughput methodology, no more than **3
  percent** time-weighted average degraded throughput and no more than **0.4
  percent** absolute change in link availability, for links with at least 99
  percent baseline availability at a C/N of 0 dB (FCC 24-117).

Constants are defined here so experiments never hard-code rule numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    DTT_LONG_TERM,
    LONG_TERM_EXCEEDANCE_FRACTION,
    THROUGHPUT_DEGRADE_MAX,
    UNAVAILABILITY_INCREASE_MAX,
)
from .constants import DTT_THRESHOLD as DTT_SHORT_TERM

# --- 25.261(c) band-splitting ----------------------------------------------
BAND_SPLIT_FRACTION: float = 0.5

__all__ = [
    "DTT_LONG_TERM",
    "DTT_SHORT_TERM",
    "LONG_TERM_EXCEEDANCE_FRACTION",
    "THROUGHPUT_DEGRADE_MAX",
    "UNAVAILABILITY_INCREASE_MAX",
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
