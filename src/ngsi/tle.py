"""As-deployed constellation elements from public TLE snapshots (Phase 3b).

Phase 3a models each operator as an ideal Walker-Delta pattern with a single
random phasing per shell.  For the as-deployed scenario we instead take a real
:term:`TLE` snapshot (CelesTrak), convert every satellite's mean elements to the
same element set the analytic J2 propagator consumes, and group the satellites
into shells by inclination/altitude.

The production path stays the fast batched two-body+J2 propagator
(:func:`ngsi.propagation.propagate_ecef`); SGP4 is *not* run per satellite here --
it remains the validation reference (:func:`ngsi.propagation.propagate_sgp4`,
``tests/test_propagation.py``).  TLE mean motion is treated as the osculating
``n`` so ``a = (mu / n^2)^(1/3)`` is consistent with the model's ``n``.

No number read from a TLE is an input to any physical model: the snapshot is
downloaded and its URL/date recorded in ``data/raw/README.md`` (AGENTS.md rule 1).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import Beam, Shell
from .constellation import MU_EARTH_KM3_S2, WalkerConstellation
from .frames import utc_to_jd
from .propagation import J2

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
USER_AGENT = "ngsi-research/0.1 (+https://github.com/CLIS-WPI/ngso-sharing-incidence)"
TWO_PI = 2.0 * np.pi


def celestrak_url(group: str, fmt: str = "tle") -> str:
    """CelesTrak GP query URL for one group (``starlink``, ``oneweb``, ...)."""
    return f"{CELESTRAK_GP_URL}?GROUP={group}&FORMAT={fmt}"


@dataclass(frozen=True)
class TleRecord:
    """One satellite: identity, epoch and the six TLE mean elements."""

    name: str
    line1: str
    line2: str
    epoch_jd: float
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_day: float

    @property
    def mean_motion_rad_s(self) -> float:
        return float(self.mean_motion_rev_day * TWO_PI / 86400.0)

    @property
    def semi_major_axis_km(self) -> float:
        n = self.mean_motion_rad_s
        return float((MU_EARTH_KM3_S2 / n**2) ** (1.0 / 3.0))

    @property
    def argument_of_latitude_rad(self) -> float:
        return float(np.deg2rad(self.arg_perigee_deg + self.mean_anomaly_deg))


def _tle_epoch_jd(line1: str) -> float:
    """Julian date (UTC) of a TLE epoch field ``YYDDD.DDDDDDDD`` (cols 19-32)."""
    field = line1[18:32].strip()
    yy = int(field[:2])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(field[2:])
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)
    return utc_to_jd(dt)


def parse_tle(text: str) -> list[TleRecord]:
    """Parse a TLE block into :class:`TleRecord` objects (blank lines ignored)."""
    lines = [ln.rstrip("\n\r") for ln in text.splitlines() if ln.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(f"TLE block has {len(lines)} non-blank lines (not a multiple of 3)")
    records: list[TleRecord] = []
    for i in range(0, len(lines), 3):
        name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
        if not l1.startswith("1 ") or not l2.startswith("2 "):
            raise ValueError(f"bad TLE at line {i + 1}: {l1!r} / {l2!r}")
        records.append(
            TleRecord(
                name=name,
                line1=l1,
                line2=l2,
                epoch_jd=_tle_epoch_jd(l1),
                inclination_deg=float(l2[8:16]),
                raan_deg=float(l2[17:25]),
                eccentricity=float("0." + l2[26:33].strip()),
                arg_perigee_deg=float(l2[34:42]),
                mean_anomaly_deg=float(l2[43:51]),
                mean_motion_rev_day=float(l2[52:63]),
            )
        )
    return records


def fetch_group(
    group: str,
    cache_dir: str | Path,
    *,
    force: bool = False,
    timeout: float = 60.0,
) -> str:
    """Download a CelesTrak group (cached in ``cache_dir/<group>.tle``); return text.

    Returns the cached file verbatim when present unless ``force`` is set, so a
    run is reproducible against a pinned snapshot once downloaded.
    """
    cache = Path(cache_dir) / f"{group}.tle"
    if cache.exists() and not force:
        return cache.read_text(encoding="utf-8")
    req = urllib.request.Request(celestrak_url(group), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed host)
        text = resp.read().decode("utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def _secular_rates(record: TleRecord, earth_radius_km: float) -> tuple[float, float]:
    """J2 secular rates ``(raan_dot, u_dot)`` [rad/s] for one record (circular model)."""
    a = record.semi_major_axis_km
    n = record.mean_motion_rad_s
    inc = np.deg2rad(record.inclination_deg)
    factor = J2 * n * (earth_radius_km / a) ** 2
    raan_dot = -1.5 * factor * np.cos(inc)
    u_dot = n + 0.75 * factor * (5.0 * np.cos(inc) ** 2 - 1.0)
    return float(raan_dot), float(u_dot)


def advance_to_epoch(
    record: TleRecord, epoch_jd: float, earth_radius_km: float
) -> tuple[float, float]:
    """RAAN and argument of latitude (rad) of ``record`` at ``epoch_jd``.

    Uses the same J2 secular rates as :func:`ngsi.propagation.propagate_ecef`, so
    a common simulation epoch can be used even though each TLE has its own epoch.
    """
    dt_s = (epoch_jd - record.epoch_jd) * 86400.0
    raan_dot, u_dot = _secular_rates(record, earth_radius_km)
    raan = np.deg2rad(record.raan_deg) + raan_dot * dt_s
    u = record.argument_of_latitude_rad + u_dot * dt_s
    return float(raan % TWO_PI), float(u % TWO_PI)


def common_epoch_jd(records: list[TleRecord]) -> float:
    """Simulation epoch: the latest TLE epoch in the snapshot (all are advanced to it)."""
    if not records:
        raise ValueError("no records")
    return float(max(r.epoch_jd for r in records))


def elements_from_records(
    records: list[TleRecord],
    epoch_jd: float,
    earth_radius_km: float,
    plane_index: NDArray[np.int64] | None = None,
) -> WalkerConstellation:
    """Build a flat :class:`WalkerConstellation` for ``records`` at ``epoch_jd``."""
    n = len(records)
    a = np.empty(n)
    e = np.empty(n)
    inc = np.empty(n)
    raan = np.empty(n)
    argp = np.empty(n)
    m = np.empty(n)
    mm = np.empty(n)
    for i, rec in enumerate(records):
        ra, u = advance_to_epoch(rec, epoch_jd, earth_radius_km)
        a[i] = rec.semi_major_axis_km
        e[i] = rec.eccentricity
        inc[i] = np.deg2rad(rec.inclination_deg)
        raan[i] = ra
        argp[i] = np.deg2rad(rec.arg_perigee_deg)
        m[i] = (u - argp[i]) % TWO_PI
        mm[i] = rec.mean_motion_rad_s
    if plane_index is None:
        plane_index = np.zeros(n, dtype=np.int64)
    return WalkerConstellation(
        semi_major_axis_km=a,
        eccentricity=e,
        inclination_rad=inc,
        raan_rad=raan,
        arg_perigee_rad=argp,
        mean_anomaly_rad=m,
        mean_motion_rad_s=mm,
        plane_index=np.asarray(plane_index, dtype=np.int64),
        slot_index=np.arange(n, dtype=np.int64),
    )


def cluster_records(
    records: list[TleRecord],
    incl_tol_deg: float = 2.0,
    alt_tol_km: float = 40.0,
) -> list[list[int]]:
    """Greedy clustering of records into near-identical (inclination, altitude) shells.

    Returns a list of index lists; each cluster is a shell of the as-deployed
    constellation.  Order is deterministic (first-seen anchor).
    """
    clusters: list[list[int]] = []
    anchors: list[tuple[float, float]] = []
    for i, rec in enumerate(records):
        a = rec.semi_major_axis_km
        inc = rec.inclination_deg
        placed = False
        for ci, (inc0, a0) in enumerate(anchors):
            if abs(inc - inc0) <= incl_tol_deg and abs(a - a0) <= alt_tol_km:
                clusters[ci].append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            anchors.append((inc, a))
    return clusters


def shells_from_clusters(
    records: list[TleRecord],
    clusters: list[list[int]],
    beam: Beam,
    source: str,
    earth_radius_km: float,
) -> tuple[list[Shell], NDArray[np.int64]]:
    """Turn clusters into config :class:`Shell` objects and a per-satellite shell index.

    ``planes``/``sats_per_plane`` are derived from the observed RAAN distribution
    (greedy 5 deg) and are descriptive only; the elements (not these counts) drive
    propagation.  ``beam`` is inherited from the modelled constellation because no
    filing states a per-generation beam.
    """
    shell_index = np.empty(len(records), dtype=np.int64)
    shells: list[Shell] = []
    for ci, idx in enumerate(clusters):
        incs = np.array([records[i].inclination_deg for i in idx])
        alts = np.array([records[i].semi_major_axis_km - earth_radius_km for i in idx])
        raans = np.array([records[i].raan_deg for i in idx])
        n_planes = _count_planes(raans)
        spp = max(1, int(round(len(idx) / n_planes)))
        for i in idx:
            shell_index[i] = ci
        shells.append(
            Shell(
                name=f"tle-{ci:02d}",
                altitude_km=float(np.mean(alts)),
                inclination_deg=float(np.mean(incs)),
                planes=n_planes,
                sats_per_plane=spp,
                phasing_f=0,
                beam=beam,
                source=source,
            )
        )
    return shells, shell_index


def filter_operational(
    records: list[TleRecord],
    earth_radius_km: float,
    min_altitude_km: float,
    max_altitude_km: float,
) -> tuple[list[TleRecord], list[TleRecord], list[TleRecord]]:
    """Split records into (kept, raising/deorbit, above-band) by circular altitude.

    The band ``[min_altitude_km, max_altitude_km]`` is the operator's operating
    shell altitude window (source tagged ASSUMED in the experiment config).
    Satellites below it are orbit-raising or deorbiting; satellites above it are
    off-shell.  Neither is part of the operational interference environment.
    """
    kept: list[TleRecord] = []
    low: list[TleRecord] = []
    high: list[TleRecord] = []
    for r in records:
        alt = r.semi_major_axis_km - earth_radius_km
        if alt < min_altitude_km:
            low.append(r)
        elif alt > max_altitude_km:
            high.append(r)
        else:
            kept.append(r)
    return kept, low, high


def _count_planes(raan_deg: NDArray[np.float64], tol_deg: float = 5.0) -> int:
    """Number of distinct RAAN planes (greedy binning on the RAAN circle)."""
    anchors: list[float] = []
    for r in np.sort(raan_deg % 360.0):
        for i, a in enumerate(anchors):
            d = abs(r - a) % 360.0
            if min(d, 360.0 - d) <= tol_deg:
                anchors[i] = 0.5 * (a + r)
                break
        else:
            anchors.append(float(r))
    return max(1, len(anchors))
