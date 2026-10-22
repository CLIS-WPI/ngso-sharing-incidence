"""Authorized frequency bands and the co-frequency overlap matrix.

Every system declares its authorized frequency ranges per *link type* (see
``configs/bands.yaml``).  In-line interference and 47 CFR 25.261 band-splitting
only apply to a pair of systems that are co-frequency in the **same link type
and direction**; :func:`overlap` decides this and returns the shared band and its
centre frequency (needed to replace a single per-system ``freq_ghz``).

All units are GHz.  Ranges are half-open ``[low, high)`` but thresholds are
floating point so "touching" bands (``high_a == low_b``) do **not** count as
overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError

LINK_TYPES: tuple[str, ...] = (
    "user_downlink",
    "user_uplink",
    "gateway_downlink",
    "gateway_uplink",
)

Range = tuple[float, float]


@dataclass(frozen=True)
class BandSet:
    """Authorized frequency ranges for one system in one link type."""

    ranges_ghz: tuple[Range, ...]
    source: str


def _validate_ranges(raw: Any) -> tuple[Range, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"band ranges must be a non-empty list: {raw!r}")
    out: list[Range] = []
    for entry in raw:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise ConfigError(f"each band range must be [low, high]: {entry!r}")
        lo, hi = float(entry[0]), float(entry[1])
        if not lo < hi:
            raise ConfigError(f"band range low must be < high: {entry!r}")
        out.append((lo, hi))
    return tuple(sorted(out))


@dataclass(frozen=True)
class SystemBands:
    """All authorized bands for one system, keyed by link type."""

    name: str
    links: dict[str, BandSet]

    def ranges(self, link_type: str) -> tuple[Range, ...]:
        if link_type not in self.links:
            raise KeyError(f"{self.name} has no bands for link type {link_type!r}")
        return self.links[link_type].ranges_ghz


@dataclass(frozen=True)
class Overlap:
    """Co-frequency overlap of two systems in one link type."""

    link_type: str
    segments_ghz: tuple[Range, ...]

    @property
    def overlaps(self) -> bool:
        return len(self.segments_ghz) > 0

    @property
    def total_bandwidth_ghz(self) -> float:
        return float(sum(hi - lo for lo, hi in self.segments_ghz))

    @property
    def widest_segment_ghz(self) -> Range | None:
        if not self.segments_ghz:
            return None
        return max(self.segments_ghz, key=lambda s: s[1] - s[0])

    @property
    def center_ghz(self) -> float | None:
        """Centre of the widest shared segment (per-pair shared-band frequency)."""
        seg = self.widest_segment_ghz
        if seg is None:
            return None
        return float(0.5 * (seg[0] + seg[1]))


def intersect(a: tuple[Range, ...], b: tuple[Range, ...]) -> tuple[Range, ...]:
    """Intersections of two sorted, disjoint range lists (half-open)."""
    segs: list[Range] = []
    for la, ha in a:
        for lb, hb in b:
            lo, hi = max(la, lb), min(ha, hb)
            if lo < hi:
                segs.append((lo, hi))
    return tuple(sorted(segs))


def overlap(a: SystemBands, b: SystemBands, link_type: str) -> Overlap:
    """Co-frequency overlap of systems ``a`` and ``b`` in ``link_type``."""
    return Overlap(link_type, intersect(a.ranges(link_type), b.ranges(link_type)))


def pair_overlaps(a: SystemBands, b: SystemBands) -> dict[str, Overlap]:
    """All link-type overlaps for an ordered system pair."""
    return {lt: overlap(a, b, lt) for lt in LINK_TYPES}


def user_downlink_overlap(a: SystemBands, b: SystemBands) -> Overlap:
    """The user-downlink overlap (the link type used by the latitude station grid)."""
    return overlap(a, b, "user_downlink")


def co_frequency_pairs(
    names: list[str], bands: dict[str, SystemBands], link_type: str
) -> list[tuple[str, str]]:
    """Ordered ``(a, b)`` pairs of *different* systems co-frequency in ``link_type``."""
    out: list[tuple[str, str]] = []
    for a in names:
        for b in names:
            if a == b:
                continue
            if overlap(bands[a], bands[b], link_type).overlaps:
                out.append((a, b))
    return out


def load_bands(path: str | Path) -> dict[str, SystemBands]:
    """Load ``bands.yaml`` keyed by system name."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"bands config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("bands.yaml must contain a mapping at top level")
    entries = data.get("systems")
    if not isinstance(entries, dict) or not entries:
        raise ConfigError("bands.yaml must define a non-empty 'systems' mapping")
    result: dict[str, SystemBands] = {}
    for name, links in entries.items():
        if not isinstance(links, dict):
            raise ConfigError(f"bands for {name!r} must be a mapping: {links!r}")
        for lt in links:
            if lt not in LINK_TYPES:
                raise ConfigError(f"unknown link type {lt!r} for {name!r}")
        parsed: dict[str, BandSet] = {}
        for lt, entry in links.items():
            if not isinstance(entry, dict) or "source" not in entry:
                raise ConfigError(f"{name}/{lt} missing required 'source'")
            parsed[lt] = BandSet(_validate_ranges(entry.get("ranges_ghz")), str(entry["source"]))
        result[str(name)] = SystemBands(str(name), parsed)
    return result
