"""theta* from Eq. (5) and the PFD0 of the configured Starlink beam (Phase 4e).

Eq. (5) of the paper: with a constant surface PFD0 the victim's I/N toward a
satellite seen at off-axis angle theta is

    I/N = PFD0 + G(theta) + 10 log10(lambda^2 / 4 pi) - N,

so the event threshold I/N = 10 log10(delta) is crossed at the gain
G* = 10 log10(delta) + N - PFD0 - 10 log10(lambda^2/4pi), and on the S.1428 /
S.465 sidelobe envelope G(theta) = 32 - 25 log10(theta) at
theta* = 10^((32 - G*) / 25).  Under constant PFD this does not depend on the
shell altitude.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from ngsi.analytic import beam_pfd0_dbw_per_m2_per_mhz, equivalent_angle_deg
from ngsi.config import load_simulation
from ngsi.constants import DTT_THRESHOLD
from ngsi.link_budget import noise_power_dbw_per_mhz, terminal_peak_gain_dbi

REPO = Path(__file__).resolve().parents[1]
K_BOLTZMANN = 1.380649e-23
C_LIGHT = 299792458.0


def _theta_star_eq5(pfd0: float, t_sys_k: float, freq_ghz: float, delta: float) -> float:
    noise = 10.0 * math.log10(K_BOLTZMANN * t_sys_k * 1e6)          # dBW/MHz
    lam = C_LIGHT / (freq_ghz * 1e9)
    area = 10.0 * math.log10(lam**2 / (4.0 * math.pi))               # dB m^2
    g_star = 10.0 * math.log10(delta) + noise - pfd0 - area           # dBi
    return 10.0 ** ((32.0 - g_star) / 25.0)


def test_theta_star_hand_value() -> None:
    # PFD0 = -116, T = 350.1 K, f = 11.7 GHz, delta = 0.06:
    # N = -143.16 dBW/MHz, 10log(lambda^2/4pi) = -42.82 dB, G* = 3.44 dBi -> 13.88 deg.
    assert _theta_star_eq5(-116.0, 350.1, 11.7, 0.06) == pytest.approx(13.88, abs=0.005)


def test_configured_pfd0_is_the_limit_for_every_shell() -> None:
    sim = load_simulation(REPO / "configs" / "constellations.yaml")
    for name in ("STARLINK", "STARLINK_GEN2"):
        for sh in sim.systems[name].shells:
            assert beam_pfd0_dbw_per_m2_per_mhz(sh.beam, sim.earth.radius_km) == \
                pytest.approx(-116.0, abs=5e-4), (name, sh.name, sh.altitude_km)


def test_library_theta_star_matches_eq5() -> None:
    sim = load_simulation(REPO / "configs" / "constellations.yaml")
    beam = sim.systems["STARLINK_GEN2"].shells[0].beam
    t_sys = sim.systems["ONEWEB"].shells[0].beam.system_noise_temp_k
    pfd0 = beam_pfd0_dbw_per_m2_per_mhz(beam, sim.earth.radius_km)
    peak = terminal_peak_gain_dbi(0.45, 11.7, 0.6)
    lib = equivalent_angle_deg(peak, 0.45, 11.7, pfd0, noise_power_dbw_per_mhz(t_sys))
    assert lib == pytest.approx(_theta_star_eq5(pfd0, t_sys, 11.7, DTT_THRESHOLD), abs=0.01)


def test_theta_star_macro_matches_eq5() -> None:
    txt = (REPO / "paper" / "numbers.tex").read_text(encoding="utf-8")
    m = re.search(r"\\newcommand\{\\AnalyticThetaStarDeg\}\{([^}]*)\}", txt)
    assert m, "AnalyticThetaStarDeg missing from paper/numbers.tex"
    sim = load_simulation(REPO / "configs" / "constellations.yaml")
    t_sys = sim.systems["ONEWEB"].shells[0].beam.system_noise_temp_k
    expected = _theta_star_eq5(-116.0, t_sys, 11.7, DTT_THRESHOLD)
    assert float(m.group(1)) == pytest.approx(expected, abs=0.006)
