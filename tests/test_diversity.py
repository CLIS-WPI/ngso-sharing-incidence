"""Satellite-diversity resolution logic (AGENTS.md rule 6)."""

from __future__ import annotations

import numpy as np

from ngsi.diversity import DiversityOutcome, resolve_diversity, unresolvable_rate


def test_resolves_when_a_clear_candidate_exists() -> None:
    out = resolve_diversity(np.array([False, True, False]))
    assert out.resolved
    assert out.chosen_index == 1
    assert out.n_candidates == 3


def test_unresolved_when_all_candidates_interfered() -> None:
    out = resolve_diversity(np.array([False, False]))
    assert not out.resolved
    assert out.chosen_index is None


def test_empty_candidate_set_is_unresolved() -> None:
    out = resolve_diversity(np.array([], dtype=bool))
    assert not out.resolved
    assert out.n_candidates == 0


def test_unresolvable_rate() -> None:
    outcomes = [
        DiversityOutcome(True, 0, 2),
        DiversityOutcome(False, None, 2),
        DiversityOutcome(False, None, 1),
        DiversityOutcome(True, 1, 4),
    ]
    assert unresolvable_rate(outcomes) == 0.5
