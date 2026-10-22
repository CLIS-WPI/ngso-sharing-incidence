"""Single-source physical constants shared across ``ngsi``.

The 47 CFR 25.261(d) short-term ``Delta T/T`` criterion is defined here **once**
and imported everywhere else (e.g. ``ngsi.rules``) so copies cannot drift.  A test pins the value and its I/N equivalent.
"""

from __future__ import annotations

import numpy as np

# 47 CFR 25.261(d) / ITU-R S.1323 short-term Delta T/T criterion.
DTT_THRESHOLD: float = 0.06

# Equivalent single-entry I/N threshold (dB), derived so the two cannot drift.
IN_THRESHOLD_DB: float = float(10.0 * np.log10(DTT_THRESHOLD))

# Historical labels retained for backward compatibility; these same values are
# the 47 CFR 25.261(d) degraded-throughput protection limits below.
DTT_LONG_TERM: float = 0.03
LONG_TERM_EXCEEDANCE_FRACTION: float = 0.004

# 47 CFR 25.261(d) degraded-throughput protection (added by FCC 24-117): a
# later-round system's compatibility showing must keep the *time-weighted
# average degraded throughput* of an earlier-round link at or below 3 percent
# and the *absolute change in link availability* at or below 0.4 percent, for
# links whose baseline availability is at least 99 percent at a C/N of 0 dB.
THROUGHPUT_DEGRADE_MAX: float = 0.03
UNAVAILABILITY_INCREASE_MAX: float = 0.004


def dtt_to_i_over_n_db(dtt: float) -> float:
    """Equivalent single-entry I/N (dB) for a ``Delta T/T`` increment.

    For one interfering entry, ``Delta T/T = I / (k T_sys B) = I / N``, so the
    equivalent I/N in dB is simply ``10 log10(Delta T/T)``.  The 6% criterion is
    therefore ``-12.2 dB`` (``10 log10 0.06 = -12.218 dB``).
    """
    if dtt <= 0.0:
        raise ValueError("dtt must be positive")
    return float(10.0 * np.log10(dtt))
