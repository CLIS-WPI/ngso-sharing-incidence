"""TLE parsing and as-deployed element conversion.

A real CelesTrak TLE (STARLINK-1008, 2026 day 290) is used as the hand-checkable
fixture; the epoch is cross-checked against ``utc_to_jd`` of the calendar date.
"""

from __future__ import annotations

import numpy as np
import pytest

from ngsi.config import Beam
from ngsi.frames import utc_to_jd
from ngsi.propagation import propagate_ecef
from ngsi.tle import (
    TleRecord,
    advance_to_epoch,
    celestrak_url,
    cluster_records,
    common_epoch_jd,
    elements_from_records,
    filter_operational,
    parse_tle,
    shells_from_clusters,
)

L1 = "1 44714U 19074B   26290.76446686  .00047328  00000+0  43433-3 0  9990"
L2 = "2 44714  53.1444 193.2866 0002884 130.0102 230.1168 15.67276838383342"
TLE_TEXT = f"STARLINK-1008\n{L1}\n{L2}\n"


def test_celestrak_url():
    assert celestrak_url("starlink") == (
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    )


def test_parse_fields():
    rec = parse_tle(TLE_TEXT)[0]
    assert rec.name == "STARLINK-1008"
    assert rec.inclination_deg == pytest.approx(53.1444)
    assert rec.raan_deg == pytest.approx(193.2866)
    assert rec.eccentricity == pytest.approx(0.0002884)
    assert rec.arg_perigee_deg == pytest.approx(130.0102)
    assert rec.mean_anomaly_deg == pytest.approx(230.1168)
    assert rec.mean_motion_rev_day == pytest.approx(15.67276838)


def test_epoch_jd_matches_calendar():
    rec = parse_tle(TLE_TEXT)[0]
    from datetime import datetime, timedelta, timezone

    jd0 = utc_to_jd(datetime(2026, 1, 1, tzinfo=timezone.utc))
    expected = jd0 + (289.0 + 0.76446686)  # day-of-year 290.76446686
    assert rec.epoch_jd == pytest.approx(expected, abs=1e-6)


def test_semi_major_axis_consistent_with_mean_motion():
    rec = parse_tle(TLE_TEXT)[0]
    n = rec.mean_motion_rad_s
    assert rec.semi_major_axis_km == pytest.approx((398600.4418 / n**2) ** (1 / 3))
    assert 6600.0 < rec.semi_major_axis_km < 7200.0


def test_elements_argument_of_latitude_and_raan():
    rec = parse_tle(TLE_TEXT)[0]
    els = elements_from_records([rec], rec.epoch_jd, 6371.0)
    assert els.semi_major_axis_km[0] == pytest.approx(rec.semi_major_axis_km)
    assert els.raan_rad[0] == pytest.approx(np.deg2rad(rec.raan_deg), abs=1e-9)
    u = els.arg_perigee_rad[0] + els.mean_anomaly_rad[0]
    assert u % (2 * np.pi) == pytest.approx(rec.argument_of_latitude_rad % (2 * np.pi), abs=1e-9)


def test_advance_equals_propagating_forward():
    """Advancing elements to epoch+d must equal propagating d seconds from epoch."""
    rec = parse_tle(TLE_TEXT)[0]
    dt_s = 3600.0
    els0 = elements_from_records([rec], rec.epoch_jd, 6371.0)
    p0 = propagate_ecef(els0, np.array([dt_s]), rec.epoch_jd, 6371.0, device="cpu")

    jd1 = rec.epoch_jd + dt_s / 86400.0
    els1 = elements_from_records([rec], jd1, 6371.0)
    p1 = propagate_ecef(els1, np.array([0.0]), jd1, 6371.0, device="cpu")

    assert np.allclose(p0[0, 0], p1[0, 0], atol=1e-6)


def test_cluster_and_shells():
    rng = np.random.default_rng(0)
    recs = []
    for inc, base in ((53.0, 550.0), (53.2, 545.0), (70.0, 570.0)):
        for k in range(4):
            r = TleRecord(
                name=f"S{inc}-{k}", line1=L1, line2=L2, epoch_jd=2461000.0,
                inclination_deg=inc + rng.uniform(-0.05, 0.05),
                raan_deg=(k * 30.0) % 360.0, eccentricity=1e-4,
                arg_perigee_deg=0.0, mean_anomaly_deg=float(rng.uniform(0, 360)),
                mean_motion_rev_day=15.5,
            )
            r = _override_altitude(r, base)
            recs.append(r)
    clusters = cluster_records(recs, incl_tol_deg=2.0, alt_tol_km=40.0)
    assert len(clusters) == 2  # 53.0+53.2 merge; 70 separate
    beam = Beam(44.0, 1.1, -8.3, 12.0, 240.0, 150.0, "test")
    shells, idx = shells_from_clusters(recs, clusters, beam, "test", 6371.0)
    assert len(shells) == 2
    assert idx.shape == (12,)
    assert set(idx.tolist()) == {0, 1}
    assert common_epoch_jd(recs) == 2461000.0


def _override_altitude(rec: TleRecord, altitude_km: float) -> TleRecord:
    """Return ``rec`` with mean motion set for an ``altitude_km`` circular orbit."""
    a = 6371.0 + altitude_km
    n = np.sqrt(398600.4418 / a**3)
    return TleRecord(
        name=rec.name, line1=rec.line1, line2=rec.line2, epoch_jd=rec.epoch_jd,
        inclination_deg=rec.inclination_deg, raan_deg=rec.raan_deg,
        eccentricity=rec.eccentricity, arg_perigee_deg=rec.arg_perigee_deg,
        mean_anomaly_deg=rec.mean_anomaly_deg, mean_motion_rev_day=n * 86400.0 / (2 * np.pi),
    )


def test_build_as_deployed_system_from_records():
    from ngsi.config import Constellation
    from ngsi.systems import build_as_deployed_system

    recs = []
    for k in range(6):
        recs.append(
            TleRecord(
                name=f"OW-{k}", line1=L1, line2=L2, epoch_jd=2461000.0,
                inclination_deg=87.9, raan_deg=(k * 60.0) % 360.0, eccentricity=2e-4,
                arg_perigee_deg=0.0, mean_anomaly_deg=(k * 45.0) % 360.0,
                mean_motion_rev_day=13.16,
            )
        )
    con = Constellation(
        name="ONEWEB", operator="test",
        shells=[_oneweb_shell()],
    )
    model = build_as_deployed_system(con, recs, earth_radius_km=6371.0)
    assert model.n_sats == 6
    assert model.n_shells == 1
    assert model.elements.semi_major_axis_km.shape == (6,)
    assert model.shells[0].beam.peak_gain_dbi == 30.0
    assert "CelesTrak" in model.shells[0].source


def _oneweb_shell():
    from ngsi.config import Shell

    return Shell(
        name="s1", altitude_km=1200.0, inclination_deg=87.9, planes=18, sats_per_plane=36,
        phasing_f=1, beam=Beam(30.0, 3.0, -20.0, 12.0, 125.0, 200.0, "test"), source="test",
    )


def test_filter_operational_hand_checkable():
    """250 km -> raise, 550 km -> kept, 1200 km -> off-shell high (window 450-600)."""
    recs = [
        _override_altitude(_rec_at(0.0), 250.0),
        _override_altitude(_rec_at(10.0), 550.0),
        _override_altitude(_rec_at(20.0), 1200.0),
    ]
    kept, low, high = filter_operational(recs, 6371.0, 450.0, 600.0)
    assert [r.name for r in kept] == ["R10.0"]
    assert [r.name for r in low] == ["R0.0"]
    assert [r.name for r in high] == ["R20.0"]
    assert len(kept) + len(low) + len(high) == len(recs)


def _rec_at(raan_deg: float) -> TleRecord:
    return TleRecord(
        name=f"R{raan_deg}", line1=L1, line2=L2, epoch_jd=2461000.0,
        inclination_deg=53.0, raan_deg=raan_deg, eccentricity=1e-4,
        arg_perigee_deg=0.0, mean_anomaly_deg=0.0, mean_motion_rev_day=15.5,
    )


def test_build_as_deployed_respects_explicit_epoch():
    """A later epoch_jd shifts the mean anomaly by the J2 secular rate, not zero."""
    from ngsi.config import Constellation
    from ngsi.systems import build_as_deployed_system

    recs = [_rec_at(float(k)) for k in range(6)]
    con = Constellation(name="ONEWEB", operator="t", shells=[_oneweb_shell()])
    same = build_as_deployed_system(con, recs, 6371.0, epoch_jd=2461000.0)
    later = build_as_deployed_system(con, recs, 6371.0, epoch_jd=2461001.0)
    assert np.allclose(same.elements.mean_anomaly_rad,
                       np.deg2rad([r.mean_anomaly_deg for r in recs]))
    assert not np.allclose(same.elements.mean_anomaly_rad,
                           later.elements.mean_anomaly_rad)


def test_advance_to_epoch_zero_when_same():
    rec = parse_tle(TLE_TEXT)[0]
    raan, u = advance_to_epoch(rec, rec.epoch_jd, 6371.0)
    assert raan == pytest.approx(np.deg2rad(rec.raan_deg), abs=1e-12)
    assert u == pytest.approx(rec.argument_of_latitude_rad % (2 * np.pi), abs=1e-12)
