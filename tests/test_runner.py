"""Hand-checkable values for the experiment orchestration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ngsi.runner import _run_lengths, event_rate_per_hour, summarise, t95


def test_run_lengths_basic() -> None:
    np.testing.assert_array_equal(_run_lengths(np.array([False, True, True, False, True])), [2, 1])
    np.testing.assert_array_equal(_run_lengths(np.array([True, True, True])), [3])
    assert _run_lengths(np.zeros(4, dtype=bool)).size == 0


def test_event_rate_per_hour() -> None:
    # 10% of 10 s samples flag an event -> 36 events/hour at that cell.
    assert event_rate_per_hour(0.1, 10.0, 86400.0) == pytest.approx(36.0)


def test_t95_lookup_and_fallback() -> None:
    assert t95(1) != t95(1)  # nan for n<2
    assert t95(8) == pytest.approx(2.365)
    assert t95(11) == pytest.approx(2.228)  # nearest-tabled fallback
    assert t95(1000) == pytest.approx(1.96)


def test_summarise_ci_width_scales() -> None:
    rows = [
        {"a": 1, "phasing": p, "lon": lon, "x": 0.1 * p + 0.01 * lon}
        for p in range(4)
        for lon in (0, 1)
    ]
    out = summarise(pd.DataFrame(rows), ("x",), ["a"])
    # Mean over phasings of 0.1*p (+lon mean 0.005) = 0.15 + 0.005.
    assert out.loc[0, "x_mean"] == pytest.approx(0.155)
    assert out.loc[0, "n_phasings"] == 4
    assert out.loc[0, "x_ci95"] > 0.0
