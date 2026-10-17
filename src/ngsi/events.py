"""In-line event detection.

Two detection criteria are supported:

* a *geometric proxy*: the angular separation at the victim earth station between
  the wanted and the interfering satellite is below a configured threshold;
* the *Delta T/T* criterion from 47 CFR 25.261(d) / ITU-R, ``Delta T/T > 6%``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# 47 CFR 25.261(d) / ITU-R S.1323 short-term Delta T/T criterion.
DTT_THRESHOLD: float = 0.06


def is_inline(separation_deg: float, inline_threshold_deg: float) -> bool:
    """True when a wanted/interfering pair subtends less than the inline threshold."""
    return abs(float(separation_deg)) <= abs(float(inline_threshold_deg))


def detect_inline_events(
    separation_deg: NDArray[np.float64] | float,
    inline_threshold_deg: float,
) -> NDArray[np.bool_] | bool:
    """Boolean mask of geometric in-line events (elementwise over ``separation_deg``)."""
    sep = np.asarray(separation_deg, dtype=np.float64)
    mask = np.abs(sep) <= abs(float(inline_threshold_deg))
    if np.isscalar(separation_deg):
        return bool(mask)
    return mask


def detect_dtt_events(
    delta_t_over_t: NDArray[np.float64] | float,
    threshold: float = DTT_THRESHOLD,
) -> NDArray[np.bool_] | bool:
    """Boolean mask of events exceeding the Delta T/T threshold."""
    dtt = np.asarray(delta_t_over_t, dtype=np.float64)
    mask = dtt > threshold
    if np.isscalar(delta_t_over_t):
        return bool(mask)
    return mask


def detect_events(
    separation_deg: NDArray[np.float64] | float,
    delta_t_over_t: NDArray[np.float64] | float,
    inline_threshold_deg: float,
    dtt_threshold: float = DTT_THRESHOLD,
) -> NDArray[np.bool_] | bool:
    """An event fires when the pair is geometrically in-line *or* Delta T/T is exceeded."""
    geo = detect_inline_events(separation_deg, inline_threshold_deg)
    dtt = detect_dtt_events(delta_t_over_t, dtt_threshold)
    return np.logical_or(geo, dtt)
