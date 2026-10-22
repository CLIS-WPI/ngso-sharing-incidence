"""Beam/link schema and sourcing (AGENTS.md rule 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ngsi.config import Beam, ConfigError, load_constellations

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

_LINK_FIELDS = ("freq_ghz", "bandwidth_mhz", "system_noise_temp_k")


def _beam_dict(**overrides):
    d = {
        "peak_gain_dbi": 44.0,
        "beamwidth_deg": 1.1,
        "eirp_density_dbw_per_mhz": -8.3,
        "freq_ghz": 12.0,
        "bandwidth_mhz": 240.0,
        "system_noise_temp_k": 150.0,
        "source": "test",
    }
    d.update(overrides)
    return d


def test_beam_round_trips_all_fields():
    beam = Beam.from_dict(_beam_dict())
    assert beam.freq_ghz == 12.0
    assert beam.bandwidth_mhz == 240.0
    assert beam.system_noise_temp_k == 150.0
    assert beam.source == "test"


@pytest.mark.parametrize("missing", _LINK_FIELDS)
def test_beam_link_field_is_required(missing):
    d = _beam_dict()
    d.pop(missing)
    with pytest.raises(ConfigError, match=missing):
        Beam.from_dict(d)


@pytest.mark.parametrize("bad", ("beamwidth_deg", "freq_ghz", "bandwidth_mhz", "system_noise_temp_k"))
def test_beam_positive_fields_rejected(bad):
    with pytest.raises(ConfigError, match=bad):
        Beam.from_dict(_beam_dict(**{bad: 0.0}))


def test_shipped_beams_declare_the_new_fields_and_a_source():
    cons = load_constellations(CONFIGS / "constellations.yaml")
    for con in cons.values():
        for shell in con.shells:
            beam = shell.beam
            for field in _LINK_FIELDS:
                assert getattr(beam, field) > 0.0, (shell.name, field)
            assert beam.source.strip(), shell.name


def test_flight_provenance_of_sourced_beams():
    """STARLINK/KUIPER beams cite filings; ONEWEB EIRP cites Del Portillo 2019."""
    cons = load_constellations(CONFIGS / "constellations.yaml")
    for name in ("STARLINK", "KUIPER"):
        for shell in cons[name].shells:
            assert "ASSUMED" != shell.beam.source.strip(), shell.name
            assert "FCC" in shell.beam.source or "fcc" in shell.beam.source, shell.name
    for shell in cons["ONEWEB"].shells:
        src = shell.beam.source
        assert "Del Portillo" in src, shell.name  # EIRP density is sourced
        assert "ASSUMED" in src, shell.name       # gain / beamwidth remain ASSUMED


def test_beam_steering_models():
    """STARLINK is PFD-constant (filed); KUIPER/ONEWEB are ASSUMED constant-EIRP."""
    cons = load_constellations(CONFIGS / "constellations.yaml")
    for shell in cons["STARLINK"].shells:
        assert shell.beam.beam_steering == "pfd_constant", shell.name
        assert shell.beam.pfd_anchor_elevation_deg == pytest.approx(25.0)
    for name in ("KUIPER", "ONEWEB"):
        for shell in cons[name].shells:
            assert shell.beam.beam_steering == "constant_eirp", shell.name
            assert shell.beam.pfd_anchor_elevation_deg is None, shell.name


def test_pfd_constant_requires_anchor():
    base = {
        "peak_gain_dbi": 44.0, "beamwidth_deg": 1.1,
        "eirp_density_dbw_per_mhz": 15.68, "freq_ghz": 12.0,
        "bandwidth_mhz": 240.0, "system_noise_temp_k": 150.0, "source": "x",
    }
    with pytest.raises(ValueError):
        Beam.from_dict({**base, "beam_steering": "pfd_constant"})
    with pytest.raises(ValueError):
        Beam.from_dict({**base, "beam_steering": "pfd_constant",
                        "pfd_anchor_elevation_deg": 25.0})
    ok = Beam.from_dict({**base, "beam_steering": "pfd_constant",
                         "pfd_anchor_elevation_deg": 25.0, "pfd_reference_altitude_km": 550.0})
    assert ok.beam_steering == "pfd_constant"
    assert ok.pfd_reference_altitude_km == 550.0
