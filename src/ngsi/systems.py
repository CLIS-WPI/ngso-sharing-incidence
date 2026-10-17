"""Assemble a system's shells into one Walker element set and apply phasing.

A *system* is all shells of one operator (e.g. Starlink).  In-line events are
only computed **between different systems**; shells of the same system never
interfere with each other.  A *phasing* is a random RAAN offset and a random
mean-anomaly (argument-of-latitude) offset drawn per shell, which rotates the
whole Walker pattern without changing its geometry (AGENTS.md rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .config import Constellation, Shell
from .constellation import WalkerConstellation, walker_delta

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class SystemModel:
    """One operator's full constellation as a single flat element set."""

    name: str
    elements: WalkerConstellation
    shell_index: NDArray[np.int64]
    shells: tuple[Shell, ...]

    @property
    def n_sats(self) -> int:
        return len(self.elements)

    @property
    def n_shells(self) -> int:
        return len(self.shells)


def _concatenate(elements: list[WalkerConstellation]) -> WalkerConstellation:
    return WalkerConstellation(
        semi_major_axis_km=np.concatenate([e.semi_major_axis_km for e in elements]),
        eccentricity=np.concatenate([e.eccentricity for e in elements]),
        inclination_rad=np.concatenate([e.inclination_rad for e in elements]),
        raan_rad=np.concatenate([e.raan_rad for e in elements]),
        arg_perigee_rad=np.concatenate([e.arg_perigee_rad for e in elements]),
        mean_anomaly_rad=np.concatenate([e.mean_anomaly_rad for e in elements]),
        mean_motion_rad_s=np.concatenate([e.mean_motion_rad_s for e in elements]),
        plane_index=np.concatenate([e.plane_index for e in elements]),
        slot_index=np.concatenate([e.slot_index for e in elements]),
    )


def build_system(
    system: Constellation,
    rng: np.random.Generator | None = None,
    earth_radius_km: float = 6371.0,
) -> SystemModel:
    """Build a :class:`SystemModel`, applying one random phasing per shell if ``rng`` given."""
    els: list[WalkerConstellation] = []
    idx: list[NDArray[np.int64]] = []
    for i, shell in enumerate(system.shells):
        el = walker_delta(
            altitude_km=shell.altitude_km,
            inclination_deg=shell.inclination_deg,
            planes=shell.planes,
            sats_per_plane=shell.sats_per_plane,
            phasing_f=shell.phasing_f,
            earth_radius_km=earth_radius_km,
        )
        if rng is not None:
            draan = float(rng.uniform(0.0, TWO_PI))
            dma = float(rng.uniform(0.0, TWO_PI))
            el = replace(
                el,
                raan_rad=(el.raan_rad + draan) % TWO_PI,
                mean_anomaly_rad=(el.mean_anomaly_rad + dma) % TWO_PI,
            )
        els.append(el)
        idx.append(np.full(len(el), i, dtype=np.int64))
    return SystemModel(
        name=system.name,
        elements=_concatenate(els),
        shell_index=np.concatenate(idx),
        shells=tuple(system.shells),
    )
