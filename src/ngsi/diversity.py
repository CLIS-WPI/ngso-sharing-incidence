"""Satellite-diversity resolution of in-line interference events.

Satellite diversity lets a victim earth station switch to a different satellite
of its own constellation when the currently-visible satellite is being
interfered with.  An event is *resolved* when at least one candidate satellite is
clear of co-channel in-line interference; otherwise it is *unresolvable* and the
burden falls on the link.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DiversityOutcome:
    """Result of attempting to resolve one event by satellite diversity."""

    resolved: bool
    chosen_index: int | None
    n_candidates: int


def resolve_diversity(clear_mask: NDArray[np.bool_]) -> DiversityOutcome:
    """Resolve an event given a boolean mask over candidate satellites.

    ``clear_mask[i]`` is True when candidate ``i`` is free of in-line
    interference at this instant.  A clear candidate is chosen if any exists.
    """
    mask = np.asarray(clear_mask, dtype=bool)
    n = int(mask.size)
    if n == 0:
        return DiversityOutcome(resolved=False, chosen_index=None, n_candidates=0)
    clear = np.flatnonzero(mask)
    if clear.size == 0:
        return DiversityOutcome(resolved=False, chosen_index=None, n_candidates=n)
    return DiversityOutcome(resolved=True, chosen_index=int(clear[0]), n_candidates=n)


def unresolvable_rate(outcomes: list[DiversityOutcome]) -> float:
    """Fraction of events that could not be resolved by satellite diversity."""
    if not outcomes:
        return float("nan")
    n_unresolved = sum(1 for o in outcomes if not o.resolved)
    return float(n_unresolved / len(outcomes))
