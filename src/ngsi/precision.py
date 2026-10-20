"""Floating-point precision of the heavy batched geometry (torch path).

``float64`` is the default.  ``float32`` is an option for GPUs with weak
double-precision throughput (e.g. RTX A4000, FP64 = 1/32 of FP32): orbital phases
(``u_dot * t``, ``raan_dot * t``) and their sines/cosines are still evaluated in
float64 in :func:`ngsi.propagation.propagate_ecef`, and only the resulting
positions are cast, so every per-station tensor downstream (look vectors,
elevations, off-axis angles, path losses, I/N, C/N, rates) follows the chosen
dtype.  Accumulators (time sums, counts) stay float64.

Select with the environment variable ``NGSI_COMPUTE_DTYPE`` (``float64`` |
``float32``) or :func:`set_compute_dtype`; the value is recorded in every
``run_meta.json`` (``compute_dtype``).
"""

from __future__ import annotations

import os

import torch

_DTYPES = {"float64": torch.float64, "float32": torch.float32}
_current = os.environ.get("NGSI_COMPUTE_DTYPE", "float64")
if _current not in _DTYPES:
    raise ValueError(f"NGSI_COMPUTE_DTYPE must be one of {sorted(_DTYPES)} (got {_current!r})")


def set_compute_dtype(name: str) -> None:
    global _current
    if name not in _DTYPES:
        raise ValueError(f"compute dtype must be one of {sorted(_DTYPES)}")
    _current = name


def compute_dtype_name() -> str:
    return _current


def compute_dtype() -> torch.dtype:
    return _DTYPES[_current]
