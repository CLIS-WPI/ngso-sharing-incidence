"""Single-source Delta T/T constant and its I/N equivalent (AGENTS.md rule 6)."""

from __future__ import annotations

import pytest

from ngsi import events, rules
from ngsi.constants import (
    DTT_THRESHOLD,
    THROUGHPUT_DEGRADE_MAX,
    UNAVAILABILITY_INCREASE_MAX,
    dtt_to_i_over_n_db,
)


def test_dtt_is_defined_once() -> None:
    assert events.DTT_THRESHOLD == rules.DTT_SHORT_TERM == DTT_THRESHOLD == 0.06


def test_25261d_degraded_throughput_limits() -> None:
    # 47 CFR 25.261(d) (FCC 24-117): 3% time-weighted degraded throughput and
    # 0.4% absolute change in link availability, single-sourced from constants.
    assert rules.THROUGHPUT_DEGRADE_MAX == THROUGHPUT_DEGRADE_MAX == 0.03
    assert rules.UNAVAILABILITY_INCREASE_MAX == UNAVAILABILITY_INCREASE_MAX == 0.004


def test_six_percent_dtt_equals_minus_12_2_db_i_over_n() -> None:
    # Delta T/T = I/N for a single entry; 10 log10(0.06) = -12.218 dB -> -12.2 dB.
    assert dtt_to_i_over_n_db(DTT_THRESHOLD) == pytest.approx(-12.2, abs=0.05)


def test_dtt_to_i_over_n_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        dtt_to_i_over_n_db(0.0)
