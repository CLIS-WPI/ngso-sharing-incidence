"""Hand-checkable values for the capacity-limited diversity model."""

from __future__ import annotations

import numpy as np
import torch

from ngsi.capacity_diversity import (
    _nth_max_per_group,
    regional_grid,
    rescue_with_depletion,
    served_under_capacity,
)


def test_nth_max_per_group() -> None:
    values = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    groups = torch.tensor([[0, 0, 0, 1]])
    out = _nth_max_per_group(values, groups, 2, 2)
    assert out.shape == (1, 2)
    assert out[0, 0].item() == 4.0
    assert out[0, 1].item() == float("-inf")


def test_served_under_capacity_single_satellite() -> None:
    elev = torch.tensor([[30.0, 20.0, 10.0]])
    sat = torch.tensor([[0, 0, 0]])
    vis = torch.ones((1, 3), dtype=torch.bool)
    served, granted, spare = served_under_capacity(elev, sat, vis, 2, 2)
    assert served.tolist() == [[True, True, False]]
    assert granted[0, 0].item() == 2.0
    assert spare[0, 0].item() == 0.0


def test_served_under_capacity_two_satellites() -> None:
    elev = torch.tensor([[30.0, 20.0, 10.0]])
    sat = torch.tensor([[0, 1, 0]])
    vis = torch.ones((1, 3), dtype=torch.bool)
    served, granted, spare = served_under_capacity(elev, sat, vis, 1, 2)
    assert served.tolist() == [[True, True, False]]
    assert granted[0, 0].item() == 1.0
    assert granted[0, 1].item() == 1.0
    assert spare.tolist() == [[0.0, 0.0]]


def test_rescue_depletes_spare_beams() -> None:
    # One time step, 3 cells, candidate rank 1 only (k=2), 2 satellites.
    event = torch.tensor([[True, True, False]])
    dtt_cand = torch.tensor([[[0.0, 0.01], [0.0, 0.50], [0.0, 0.0]]])  # rank1 DTT per cell
    vis_ka = torch.tensor([[[True, True], [True, True], [False, False]]])
    ia = torch.tensor([[[0, 1], [0, 1], [0, 0]]])
    va = torch.tensor([[[50.0, 40.0], [50.0, 30.0], [0.0, 0.0]]])
    spare = torch.tensor([[0.0, 1.0]])  # only satellite 1 has a spare beam
    rescued = rescue_with_depletion(event, dtt_cand, vis_ka, ia, va, spare, 0.06)
    # Cell 1 has DTT 0.50 > 0.06, so it cannot be rescued; cell 0 is saved.
    assert rescued.tolist() == [[True, False, False]]


def test_rescue_one_cell_per_satellite_per_round() -> None:
    # Both cells nominate satellite 1, which owns exactly one spare beam: the
    # higher-elevation cell wins and the satellite's spare is consumed.
    event = torch.tensor([[True, True]])
    dtt_cand = torch.tensor([[[0.0, 0.01], [0.0, 0.01]]])
    vis_ka = torch.tensor([[[True, True], [True, True]]])
    ia = torch.tensor([[[0, 1], [0, 1]]])
    va = torch.tensor([[[50.0, 40.0], [50.0, 30.0]]])
    spare = torch.tensor([[0.0, 1.0]])
    rescued = rescue_with_depletion(event, dtt_cand, vis_ka, ia, va, spare, 0.06)
    assert rescued.tolist() == [[True, False]]


def test_regional_grid_geometry() -> None:
    ecef, up = regional_grid(45.0, 0.0, 1000.0, 20.0, 6371.0)
    assert ecef.shape == (2500, 3)
    assert up.shape == (2500, 3)
    r = np.linalg.norm(ecef, axis=1)
    np.testing.assert_allclose(r, 6371.0, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(up, axis=1), 1.0, rtol=1e-12)
    lat = np.rad2deg(np.arcsin(ecef[:, 2] / r))
    assert lat.max() - lat.min() > 8.0  # ~1000 km north-south at 6371 km radius
