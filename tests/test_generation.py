"""Starlink Gen1 / Gen2 selection (Walker) and classification (TLE) (AGENTS.md rule 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ngsi.config import load_simulation
from ngsi.runner import select_starlink_generation
from ngsi.tle import TleRecord, classify_generation

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
R = 6371.0


def test_gen2_shells_follow_fcc_22_91() -> None:
    sim = load_simulation(CONFIGS / "constellations.yaml")
    g2 = sim.systems["STARLINK_GEN2"]
    # FCC 22-91 paras. 1, 19: 525/53, 530/43, 535/33 km/deg; 28 planes (para. 7).
    assert [(s.altitude_km, s.inclination_deg) for s in g2.shells] == [
        (525.0, 53.0), (530.0, 43.0), (535.0, 33.0)]
    assert all(s.planes == 28 for s in g2.shells)
    # The equal split of the 7,500 cap: 3 x 28 x 89 = 7476 <= 7500.
    assert sum(s.total_sats for s in g2.shells) == 7476


def test_select_generation() -> None:
    sim = load_simulation(CONFIGS / "constellations.yaml")
    n1 = sum(s.total_sats for s in sim.systems["STARLINK"].shells)
    n2 = sum(s.total_sats for s in sim.systems["STARLINK_GEN2"].shells)
    for gen, n in (("gen1", n1), ("gen2", n2), ("all", n1 + n2)):
        sel = select_starlink_generation(sim, gen)
        assert sel.systems["STARLINK"].name == "STARLINK"
        assert sum(s.total_sats for s in sel.systems["STARLINK"].shells) == n
    assert select_starlink_generation(sim, None) is sim
    with pytest.raises(ValueError):
        select_starlink_generation(sim, "gen3")


def _rec(inc: float, alt: float) -> TleRecord:
    # Mean motion for a circular orbit at ``alt`` (only a, i are used here).
    import math

    from ngsi.tle import MU_EARTH_KM3_S2

    a = R + alt
    n_rev_day = math.sqrt(MU_EARTH_KM3_S2 / a ** 3) * 86400.0 / (2.0 * math.pi)
    return TleRecord(name="X", line1="", line2="", epoch_jd=2460676.5, inclination_deg=inc,
                     raan_deg=0.0, eccentricity=0.0, arg_perigee_deg=0.0,
                     mean_anomaly_deg=0.0, mean_motion_rev_day=n_rev_day)


def test_classify_generation_first_match() -> None:
    rule = [
        {"generation": "gen2", "inclination_deg": 43.0, "inclination_tol_deg": 1.0,
         "altitude_min_km": 0.0, "altitude_max_km": 1e9},
        {"generation": "gen2", "inclination_deg": 53.0, "inclination_tol_deg": 0.5,
         "altitude_min_km": 0.0, "altitude_max_km": 535.0},
        {"generation": "gen1", "inclination_deg": 53.0, "inclination_tol_deg": 0.5,
         "altitude_min_km": 535.0, "altitude_max_km": 1e9},
    ]
    recs = [_rec(43.0, 560.0), _rec(53.0, 525.0), _rec(53.05, 550.0), _rec(70.0, 570.0)]
    assert classify_generation(recs, R, rule) == ["gen2", "gen2", "gen1", "unclassified"]
