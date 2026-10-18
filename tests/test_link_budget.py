"""Hand-checkable values for the link-budget / Delta T/T helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ngsi.antennas import s1428_gain
from ngsi.link_budget import (
    aggregate_dtt_from_i_over_n_db,
    cn_db,
    cn_over_n_plus_i_db,
    dtt_from_i_over_n_db,
    i_over_n_db,
    noise_power_dbw_per_mhz,
    pattern_gain_batch,
    shannon_throughput,
    terminal_peak_gain_dbi,
)


def test_terminal_peak_gain() -> None:
    # 0.6 m at 11.7 GHz, eta=0.6 -> eta (pi D/lambda)^2 = 0.6*(pi*23.42)^2.
    d_over_lambda = 0.6 * 11.7e9 / 299792458.0
    expected = 10.0 * np.log10(0.6 * (np.pi * d_over_lambda) ** 2)
    assert terminal_peak_gain_dbi(0.6, 11.7, 0.6) == pytest.approx(expected)
    assert terminal_peak_gain_dbi(0.6, 11.7, 0.6) == pytest.approx(35.11, abs=0.02)


def test_noise_power_per_mhz() -> None:
    # k*150*1e6 = 2.071e-15 W -> -146.84 dBW.
    assert noise_power_dbw_per_mhz(150.0) == pytest.approx(-146.84, abs=0.01)


def test_i_over_n_single_entry() -> None:
    val = i_over_n_db(-8.3, 34.0, 173.8, 150.0)
    assert val == pytest.approx(-8.3 + 34.0 - 173.8 + 146.84, abs=0.01)
    assert val == pytest.approx(-1.26, abs=0.02)


def test_dtt_and_aggregation() -> None:
    assert dtt_from_i_over_n_db(0.0) == pytest.approx(1.0)
    # two entries at exactly the -12.2 dB criterion add to 2*0.06.
    agg = aggregate_dtt_from_i_over_n_db(-12.2, -12.2)
    assert agg == pytest.approx(2.0 * 10.0 ** (-1.22), rel=1e-9)


def test_cn_and_cn_over_n_plus_i() -> None:
    # C/N equals EIRP + Gpeak - FSPL - noise.
    assert cn_db(-8.3, 35.0, 170.0, 150.0) == pytest.approx(
        -8.3 + 35.0 - 170.0 - noise_power_dbw_per_mhz(150.0)
    )
    # C/(N+I) = C/N - 10log10(1+DTT); DTT=0.06 -> -0.253 dB.
    assert cn_over_n_plus_i_db(10.0, 0.06) == pytest.approx(10.0 - 0.253, abs=0.002)


def test_shannon_throughput_with_margin() -> None:
    # SNR 10 dB, 3 dB margin -> B log2(1 + 10/1.995).
    r = shannon_throughput(10.0, 1e6, 3.0)
    assert r == pytest.approx(1e6 * np.log2(1.0 + 10.0 / 10.0 ** 0.3), rel=1e-9)


def test_pattern_gain_batch_matches_scalar() -> None:
    peak, d, f = 35.0, 0.6, 11.7
    psi = torch.tensor([0.0, 1.0, 5.0, 20.0], dtype=torch.float64)
    got = pattern_gain_batch(psi, peak, d, f, kind="S1428")
    assert got[0].item() == pytest.approx(peak)  # boresight
    for i, p in enumerate([0.0, 1.0, 5.0, 20.0]):
        assert got[i].item() == pytest.approx(s1428_gain(p, peak, d, f), abs=1e-9)
