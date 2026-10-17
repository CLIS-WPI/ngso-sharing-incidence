"""Event detection: geometric and Delta T/T criteria (AGENTS.md rule 6)."""

from __future__ import annotations

import numpy as np

from ngsi.events import (
    DTT_THRESHOLD,
    detect_dtt_events,
    detect_events,
    detect_inline_events,
    is_inline,
)


def test_is_inline_boundary() -> None:
    assert is_inline(0.0, 1.0)
    assert is_inline(1.0, 1.0)
    assert not is_inline(1.0001, 1.0)


def test_detect_inline_events_array() -> None:
    sep = np.array([0.0, 0.5, 2.0])
    mask = detect_inline_events(sep, 1.0)
    assert list(mask) == [True, True, False]


def test_dtt_threshold_is_6_percent() -> None:
    assert DTT_THRESHOLD == 0.06
    assert detect_dtt_events(0.07)
    assert not detect_dtt_events(0.05)


def test_detect_events_is_or_of_criteria() -> None:
    sep = np.array([5.0, 0.0])
    dtt = np.array([0.01, 0.20])
    mask = detect_events(sep, dtt, inline_threshold_deg=1.0)
    assert list(mask) == [False, True]
