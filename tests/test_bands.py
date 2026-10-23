"""Co-frequency overlap logic and the shipped band config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ngsi.bands import LINK_TYPES, intersect, load_bands, overlap, user_downlink_overlap
from ngsi.config import ConfigError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def bands():
    return load_bands(CONFIGS / "bands.yaml")


def test_intersect_hand_values():
    assert intersect([(10.7, 12.7)], [(11.0, 13.0)]) == ((11.0, 12.7),)
    assert intersect([(1.0, 2.0)], [(3.0, 4.0)]) == ()


def test_touching_bands_do_not_overlap():
    assert intersect([(1.0, 2.0)], [(2.0, 3.0)]) == ()


def test_multi_segment_intersection_and_centre():
    a = [(17.8, 18.6), (18.8, 19.3), (19.7, 20.2)]
    b = [(17.7, 18.6), (18.8, 20.2)]
    segs = intersect(a, b)
    assert segs == ((17.8, 18.6), (18.8, 19.3), (19.7, 20.2))
    # widest segment is 18.8-19.3 (0.5) vs 17.8-18.6 (0.8) -> 17.8-18.6 wins
    widest = max(segs, key=lambda s: s[1] - s[0])
    assert widest == (17.8, 18.6)
    assert 0.5 * (widest[0] + widest[1]) == pytest.approx(18.2)


def test_shipped_user_downlink_pairs(bands):
    sl, ku, ow = bands["STARLINK"], bands["KUIPER"], bands["ONEWEB"]
    # Starlink and OneWeb share the Ku user downlink.
    ov = user_downlink_overlap(sl, ow)
    assert ov.overlaps
    assert ov.segments_ghz == ((10.7, 12.7),)
    assert ov.center_ghz == pytest.approx(11.7)
    # Kuiper's user downlink is Ka -> no co-frequency user-downlink pair.
    assert not user_downlink_overlap(sl, ku).overlaps
    assert not user_downlink_overlap(ow, ku).overlaps


def test_shipped_gateway_overlaps(bands):
    sl, ku, ow = bands["STARLINK"], bands["KUIPER"], bands["ONEWEB"]
    assert overlap(sl, ku, "gateway_downlink").segments_ghz == (
        (17.8, 18.6), (18.8, 19.3), (19.7, 20.2),
    )
    assert overlap(sl, ku, "gateway_uplink").segments_ghz == ((27.5, 29.1), (29.5, 30.0))
    assert overlap(sl, ow, "gateway_downlink").segments_ghz == ((17.8, 18.6), (18.8, 19.3))
    assert overlap(sl, ow, "user_uplink").segments_ghz == ((14.0, 14.5),)


def test_all_systems_declare_every_link_type_with_source(bands):
    for name, sb in bands.items():
        for lt in LINK_TYPES:
            assert lt in sb.links, (name, lt)
            assert sb.links[lt].source.strip(), (name, lt)


def test_unknown_link_type_rejected(tmp_path):
    bad = {"systems": {"X": {"downlink": {"ranges_ghz": [[1, 2]], "source": "s"}}}}
    p = tmp_path / "bands.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match="unknown link type"):
        load_bands(p)


def test_missing_source_rejected(tmp_path):
    bad = {"systems": {"X": {"user_downlink": {"ranges_ghz": [[1, 2]]}}}}
    p = tmp_path / "bands.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match="source"):
        load_bands(p)
