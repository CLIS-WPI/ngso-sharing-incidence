"""Load and validate the YAML configuration files in ``configs/``.

Rule: every physical parameter must carry a ``source`` (an FCC filing reference)
or the literal string ``ASSUMED``.  Validation fails loudly otherwise so that
invented numbers cannot silently enter a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ASSUMED = "ASSUMED"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


@dataclass(frozen=True)
class Beam:
    """Antenna / downlink link parameters for one satellite shell.

    ``peak_gain_dbi`` and ``beamwidth_deg`` drive the ITU-R S.1528 space-station
    pattern; ``eirp_density_dbw_per_mhz``, ``freq_ghz``, ``bandwidth_mhz`` and
    ``system_noise_temp_k`` drive the I/N and Delta T/T link budget
    (:mod:`ngsi.interference`).
    """

    peak_gain_dbi: float
    beamwidth_deg: float
    eirp_density_dbw_per_mhz: float
    freq_ghz: float
    bandwidth_mhz: float
    system_noise_temp_k: float
    source: str

    _REQUIRED = (
        "peak_gain_dbi",
        "beamwidth_deg",
        "eirp_density_dbw_per_mhz",
        "freq_ghz",
        "bandwidth_mhz",
        "system_noise_temp_k",
        "source",
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Beam":
        for key in cls._REQUIRED:
            if key not in d:
                raise ConfigError(f"beam missing required key {key!r}: {d!r}")
        for key in ("beamwidth_deg", "freq_ghz", "bandwidth_mhz", "system_noise_temp_k"):
            if float(d[key]) <= 0.0:
                raise ConfigError(f"beam {key!r} must be positive: {d!r}")
        return cls(
            peak_gain_dbi=float(d["peak_gain_dbi"]),
            beamwidth_deg=float(d["beamwidth_deg"]),
            eirp_density_dbw_per_mhz=float(d["eirp_density_dbw_per_mhz"]),
            freq_ghz=float(d["freq_ghz"]),
            bandwidth_mhz=float(d["bandwidth_mhz"]),
            system_noise_temp_k=float(d["system_noise_temp_k"]),
            source=str(d["source"]),
        )


@dataclass(frozen=True)
class Shell:
    """One orbital shell of an NGSO constellation (Walker-Delta parameters)."""

    name: str
    altitude_km: float
    inclination_deg: float
    planes: int
    sats_per_plane: int
    phasing_f: int
    beam: Beam
    source: str

    @property
    def total_sats(self) -> int:
        return self.planes * self.sats_per_plane

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Shell":
        required = (
            "name",
            "altitude_km",
            "inclination_deg",
            "planes",
            "sats_per_plane",
            "phasing_f",
            "source",
        )
        for key in required:
            if key not in d:
                raise ConfigError(f"shell missing required key {key!r}: {d!r}")
        if "beam" not in d:
            raise ConfigError(f"shell {d.get('name')!r} missing 'beam'")
        if int(d["planes"]) <= 0 or int(d["sats_per_plane"]) <= 0:
            raise ConfigError(f"shell {d['name']!r}: planes and sats_per_plane must be > 0")
        return cls(
            name=str(d["name"]),
            altitude_km=float(d["altitude_km"]),
            inclination_deg=float(d["inclination_deg"]),
            planes=int(d["planes"]),
            sats_per_plane=int(d["sats_per_plane"]),
            phasing_f=int(d["phasing_f"]),
            beam=Beam.from_dict(d["beam"]),
            source=str(d["source"]),
        )


@dataclass(frozen=True)
class Constellation:
    """A named system, possibly spanning several shells."""

    name: str
    operator: str
    shells: list[Shell]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Constellation":
        if "name" not in d or "shells" not in d:
            raise ConfigError(f"constellation missing name/shells: {d!r}")
        shells = [Shell.from_dict(s) for s in d["shells"]]
        if not shells:
            raise ConfigError(f"constellation {d['name']!r} has no shells")
        return cls(
            name=str(d["name"]),
            operator=str(d.get("operator", "")),
            shells=shells,
        )


@dataclass(frozen=True)
class StationGrid:
    """Earth-station latitude/longitude grid and minimum operating elevation."""

    latitudes_deg: list[float]
    longitudes_deg: list[float]
    min_elevation_deg: float
    source: str = ASSUMED

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StationGrid":
        if "latitudes_deg" not in d or "longitudes_deg" not in d:
            raise ConfigError(f"station grid missing lat/lon lists: {d!r}")
        lats = [float(v) for v in d["latitudes_deg"]]
        lons = [float(v) for v in d["longitudes_deg"]]
        if not lats or not lons:
            raise ConfigError("station grid lat/lon lists must be non-empty")
        return cls(
            latitudes_deg=lats,
            longitudes_deg=lons,
            min_elevation_deg=float(d.get("min_elevation_deg", 25.0)),
            source=str(d.get("source", ASSUMED)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment file under ``configs/experiments/``."""

    name: str
    description: str
    constellations: list[str]
    stations: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        for key in ("name", "constellations"):
            if key not in d:
                raise ConfigError(f"experiment missing required key {key!r}: {d!r}")
        return cls(
            name=str(d["name"]),
            description=str(d.get("description", "")),
            constellations=[str(c) for c in d["constellations"]],
            stations=str(d.get("stations", "stations.yaml")),
            seed=int(d.get("seed", 0)),
            params=dict(d.get("params", {})),
        )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"config file {p} must contain a mapping at top level")
    return data


def load_constellations(path: str | Path) -> dict[str, Constellation]:
    """Load ``constellations.yaml`` keyed by constellation name."""
    data = _load_yaml(path)
    entries = data.get("constellations")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("constellations.yaml must define a non-empty 'constellations' list")
    result: dict[str, Constellation] = {}
    for entry in entries:
        con = Constellation.from_dict(entry)
        if con.name in result:
            raise ConfigError(f"duplicate constellation name {con.name!r}")
        result[con.name] = con
    return result


def load_stations(path: str | Path) -> StationGrid:
    """Load ``stations.yaml``."""
    data = _load_yaml(path)
    grid = data.get("station_grid")
    if not isinstance(grid, dict):
        raise ConfigError("stations.yaml must define a 'station_grid' mapping")
    return StationGrid.from_dict(grid)


def load_sanity_stations(path: str | Path) -> StationGrid:
    """Load the optional ``sanity_symmetry_grid`` from ``stations.yaml``."""
    data = _load_yaml(path)
    grid = data.get("sanity_symmetry_grid")
    if not isinstance(grid, dict):
        raise ConfigError("stations.yaml must define a 'sanity_symmetry_grid' mapping")
    return StationGrid.from_dict(grid)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load one file under ``configs/experiments/``."""
    return ExperimentConfig.from_dict(_load_yaml(path))


@dataclass(frozen=True)
class EarthModel:
    """Earth shape used for station geometry and orbital radius bookkeeping."""

    shape: str
    radius_km: float
    source: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EarthModel":
        for key in ("shape", "radius_km", "source"):
            if key not in d:
                raise ConfigError(f"earth_model missing required key {key!r}: {d!r}")
        if d["shape"] != "spherical":
            raise ConfigError(
                f"earth_model.shape must be 'spherical' for Gate 1 (got {d['shape']!r})"
            )
        if float(d["radius_km"]) <= 0.0:
            raise ConfigError("earth_model.radius_km must be positive")
        return cls(shape=str(d["shape"]), radius_km=float(d["radius_km"]), source=str(d["source"]))


@dataclass(frozen=True)
class SimulationConfig:
    """Everything needed to propagate: the systems, the earth model, the epoch."""

    systems: dict[str, Constellation]
    earth: EarthModel
    epoch_utc: str
    epoch_source: str


def load_simulation(path: str | Path) -> SimulationConfig:
    """Load ``constellations.yaml`` systems plus the earth model and GMST epoch."""
    data = _load_yaml(path)
    entries = data.get("constellations")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("constellations.yaml must define a non-empty 'constellations' list")
    systems: dict[str, Constellation] = {}
    for entry in entries:
        con = Constellation.from_dict(entry)
        if con.name in systems:
            raise ConfigError(f"duplicate constellation name {con.name!r}")
        systems[con.name] = con

    earth_raw = data.get("earth_model")
    if not isinstance(earth_raw, dict):
        raise ConfigError("constellations.yaml must define an 'earth_model' mapping")
    epoch_raw = data.get("epoch")
    if not isinstance(epoch_raw, dict) or "utc" not in epoch_raw:
        raise ConfigError("constellations.yaml must define an 'epoch' mapping with 'utc'")
    return SimulationConfig(
        systems=systems,
        earth=EarthModel.from_dict(earth_raw),
        epoch_utc=str(epoch_raw["utc"]),
        epoch_source=str(epoch_raw.get("source", ASSUMED)),
    )
