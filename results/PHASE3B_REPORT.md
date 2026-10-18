# Phase 3b - as-deployed constellation from a CelesTrak TLE snapshot

Provenance: run `results/e5_as_deployed_20261018T043927Z_seed0/` (git `74a685f9f7cc248baddfb9baf4314f09f8ee0eba`, seed 0). The TLE snapshot was downloaded 2026-10-18T04:39:27.680214+00:00 (URLs in `data/raw/README.md` and `tle_meta.json`); every satellite's mean elements are advanced to the common epoch JD 2461331.58583 with the analytic J2 secular rates. The production path is the fast two-body+J2 propagator; SGP4 is not used per satellite. Replication is over 8 independent start times (spacing 18000.0 s, ASSUMED).

## 1. As-deployed constellation

Satellite counts and observed inclination/altitude clusters (shells). Each cluster is a group of near-identical orbits; the *elements* (not any shell count) drive propagation.

| system | shell | n_sats | altitude_km | inclination_deg |
| --- | --- | --- | --- | --- |
| STARLINK | tle-00 | 381 | 368.3 | 53.15 |
| STARLINK | tle-01 | 4435 | 470.6 | 53.16 |
| STARLINK | tle-02 | 28 | 267.9 | 53.06 |
| STARLINK | tle-03 | 38 | 317.2 | 53.05 |
| STARLINK | tle-04 | 94 | 524.1 | 53.16 |
| STARLINK | tle-05 | 3 | 551.3 | 97.28 |
| STARLINK | tle-06 | 341 | 571.8 | 70 |
| STARLINK | tle-07 | 2 | 357.7 | 69.99 |
| STARLINK | tle-08 | 332 | 481.2 | 70 |
| STARLINK | tle-09 | 33 | 508.8 | 70 |
| STARLINK | tle-10 | 1338 | 472.8 | 97.29 |
| STARLINK | tle-11 | 12 | 341.6 | 97.43 |
| STARLINK | tle-12 | 1 | 237.9 | 97.6 |
| STARLINK | tle-13 | 3250 | 490.1 | 43 |
| STARLINK | tle-14 | 10 | 415.4 | 42.99 |
| STARLINK | tle-15 | 17 | 320.9 | 42.99 |
| STARLINK | tle-16 | 331 | 363.5 | 43 |
| STARLINK | tle-17 | 4 | 535.4 | 43 |
| STARLINK | tle-18 | 1 | 260.1 | 42.98 |
| ONEWEB | tle-00 | 648 | 1207 | 87.9 |
| ONEWEB | tle-01 | 1 | 602.9 | 88.94 |
| ONEWEB | tle-02 | 1 | 706.4 | 87.89 |
| ONEWEB | tle-03 | 1 | 530.9 | 86.91 |

Clusters well below the operational shells (e.g. STARLINK at ~240-370 km) are satellites in disposal/deorbit; they are kept because the scenario is as-deployed, and their small count makes them immaterial.

- **STARLINK**: group `starlink`, 10651 satellites, TLE epochs JD 2461304.74252..2461331.58583.

- **ONEWEB**: group `oneweb`, 651 satellites, TLE epochs JD 2461329.77113..2461331.46756.


## 2. Event time-fraction vs latitude (geometric theta criterion)

Co-frequency user-downlink pairs screened: [('ONEWEB', 'STARLINK'), ('STARLINK', 'ONEWEB')]. Only pairs with a non-zero event fraction are tabulated.

Note: for the geometric criterion the *co-located* event fraction is symmetric in the two ordered pairs (the serving-to-serving angular separation does not depend on which system is the victim), so the two co-located tables coincide; the *worst-case* variant is where the direction enters geometrically, and the power (Delta T/T) criterion in Section 3 is where the EIRP asymmetry is decisive.

**ONEWEB ← STARLINK**, co-located (serving), θ=2°:

| latitude_deg | event_frac_mean | event_frac_ci95 | unres_frac_mean | event_rate_per_hour |
| --- | --- | --- | --- | --- |
| 0 | 0.003009 | 7.9e-05 | 0 | 1.083 |
| 5 | 0.003006 | 0.000244 | 0 | 1.082 |
| 10 | 0.003201 | 0.000194 | 0 | 1.152 |
| 15 | 0.003526 | 0.000132 | 0 | 1.27 |
| 20 | 0.003393 | 0.00016 | 0 | 1.221 |
| 25 | 0.003226 | 0.000197 | 0 | 1.161 |
| 30 | 0.003443 | 0.000203 | 0 | 1.24 |
| 35 | 0.003725 | 0.000177 | 0 | 1.341 |
| 40 | 0.00425 | 0.000266 | 0 | 1.53 |
| 45 | 0.00463 | 0.000124 | 0 | 1.667 |
| 50 | 0.004876 | 0.000208 | 0 | 1.755 |
| 55 | 0.002554 | 0.000129 | 0 | 0.9193 |
| 60 | 0.003859 | 0.000307 | 0 | 1.389 |
| 65 | 0.004362 | 0.000243 | 0 | 1.57 |

Maximum event fraction 0.02935 at latitude 50° (θ=5°).

**ONEWEB ← STARLINK**, worst case (any visible), θ=2°:

| latitude_deg | event_frac_mean | event_frac_ci95 | unres_frac_mean | event_rate_per_hour |
| --- | --- | --- | --- | --- |
| 0 | 0.01737 | 0.000553 | 0 | 6.253 |
| 5 | 0.01877 | 0.000715 | 0 | 6.757 |
| 10 | 0.01824 | 0.000217 | 0 | 6.568 |
| 15 | 0.01886 | 0.000472 | 0 | 6.788 |
| 20 | 0.02005 | 0.000361 | 0 | 7.216 |
| 25 | 0.02079 | 0.000392 | 0 | 7.484 |
| 30 | 0.02291 | 0.000592 | 0 | 8.246 |
| 35 | 0.02488 | 0.000174 | 0 | 8.958 |
| 40 | 0.03575 | 0.000577 | 0 | 12.87 |
| 45 | 0.0185 | 0.000395 | 0 | 6.659 |
| 50 | 0.02825 | 0.000488 | 0 | 10.17 |
| 55 | 0.004666 | 0.000139 | 0 | 1.68 |
| 60 | 0.00582 | 0.000378 | 0 | 2.095 |
| 65 | 0.007473 | 0.000107 | 0 | 2.69 |

Maximum event fraction 0.2094 at latitude 40° (θ=5°).

**STARLINK ← ONEWEB**, co-located (serving), θ=2°:

| latitude_deg | event_frac_mean | event_frac_ci95 | unres_frac_mean | event_rate_per_hour |
| --- | --- | --- | --- | --- |
| 0 | 0.003009 | 7.9e-05 | 0 | 1.083 |
| 5 | 0.003006 | 0.000244 | 0 | 1.082 |
| 10 | 0.003201 | 0.000194 | 0 | 1.152 |
| 15 | 0.003526 | 0.000132 | 0 | 1.27 |
| 20 | 0.003393 | 0.00016 | 0 | 1.221 |
| 25 | 0.003226 | 0.000197 | 0 | 1.161 |
| 30 | 0.003443 | 0.000203 | 0 | 1.24 |
| 35 | 0.003725 | 0.000177 | 0 | 1.341 |
| 40 | 0.00425 | 0.000266 | 0 | 1.53 |
| 45 | 0.00463 | 0.000124 | 0 | 1.667 |
| 50 | 0.004876 | 0.000208 | 0 | 1.755 |
| 55 | 0.002554 | 0.000129 | 0 | 0.9193 |
| 60 | 0.003859 | 0.000307 | 0 | 1.389 |
| 65 | 0.004362 | 0.000243 | 0 | 1.57 |

Maximum event fraction 0.02935 at latitude 50° (θ=5°).

**STARLINK ← ONEWEB**, worst case (any visible), θ=2°:

| latitude_deg | event_frac_mean | event_frac_ci95 | unres_frac_mean | event_rate_per_hour |
| --- | --- | --- | --- | --- |
| 0 | 0.00353 | 5.6e-05 | 0 | 1.271 |
| 5 | 0.003458 | 0.000329 | 0 | 1.245 |
| 10 | 0.003848 | 0.00013 | 0 | 1.385 |
| 15 | 0.004091 | 0.000123 | 0 | 1.473 |
| 20 | 0.003816 | 0.000212 | 0 | 1.374 |
| 25 | 0.003624 | 0.000243 | 0 | 1.305 |
| 30 | 0.003812 | 0.000227 | 0 | 1.372 |
| 35 | 0.004116 | 0.00021 | 0 | 1.482 |
| 40 | 0.004659 | 0.000344 | 0 | 1.677 |
| 45 | 0.005353 | 0.000149 | 0 | 1.927 |
| 50 | 0.005367 | 0.000211 | 0 | 1.932 |
| 55 | 0.006456 | 0.00016 | 0 | 2.324 |
| 60 | 0.009281 | 0.000269 | 0 | 3.341 |
| 65 | 0.009657 | 0.000386 | 0 | 3.477 |

Maximum event fraction 0.05965 at latitude 65° (θ=5°).

## 3. 47 CFR 25.261(d) Delta T/T criterion

**ONEWEB ← STARLINK**, co-located (serving), default aperture:

| latitude_deg | event_frac | unres_frac | dtt_mean | degrad_nodiv | degrad_div |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.009596 | 0 | 0.008317 | 0.003503 | 0 |
| 5 | 0.009404 | 0 | 0.008363 | 0.003495 | 0 |
| 10 | 0.01101 | 0 | 0.009112 | 0.003949 | 0 |
| 15 | 0.01077 | 0 | 0.009339 | 0.003915 | 0 |
| 20 | 0.01089 | 0 | 0.009447 | 0.003888 | 0 |
| 25 | 0.01076 | 0 | 0.009361 | 0.003834 | 0 |
| 30 | 0.01212 | 0 | 0.01014 | 0.004142 | 0 |
| 35 | 0.01241 | 0 | 0.01065 | 0.004377 | 0 |
| 40 | 0.01414 | 0 | 0.01177 | 0.004921 | 0 |
| 45 | 0.01446 | 0 | 0.01268 | 0.005016 | 0 |
| 50 | 0.01596 | 0 | 0.01429 | 0.005533 | 0 |
| 55 | 0.008496 | 0 | 0.006989 | 0.002661 | 0 |
| 60 | 0.01072 | 0 | 0.009334 | 0.003588 | 0 |
| 65 | 0.0129 | 0 | 0.0109 | 0.004305 | 0 |
(aperture 0.6 m)

**ONEWEB ← STARLINK**, worst case (any visible), default aperture:

| latitude_deg | event_frac | unres_frac | dtt_mean | degrad_nodiv | degrad_div |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.05492 | 0 | 0.04333 | 0.01634 | 0 |
| 5 | 0.05617 | 0 | 0.0456 | 0.01708 | 0 |
| 10 | 0.0586 | 0 | 0.04595 | 0.01742 | 0 |
| 15 | 0.06001 | 0 | 0.04681 | 0.01765 | 0 |
| 20 | 0.06225 | 0 | 0.04923 | 0.0186 | 0 |
| 25 | 0.06629 | 0 | 0.05273 | 0.01965 | 0 |
| 30 | 0.07654 | 0 | 0.0597 | 0.02248 | 0 |
| 35 | 0.08663 | 0 | 0.06596 | 0.02474 | 0 |
| 40 | 0.135 | 0 | 0.09723 | 0.03703 | 0 |
| 45 | 0.06859 | 0 | 0.05455 | 0.01939 | 0 |
| 50 | 0.1104 | 0 | 0.08121 | 0.03047 | 0 |
| 55 | 0.01878 | 0 | 0.01686 | 0.005003 | 0 |
| 60 | 0.0168 | 0 | 0.01493 | 0.00537 | 0 |
| 65 | 0.02295 | 0 | 0.01941 | 0.007146 | 0 |
(aperture 0.6 m)

**STARLINK ← ONEWEB**, co-located (serving), default aperture:

| latitude_deg | event_frac | unres_frac | dtt_mean | degrad_nodiv | degrad_div |
| --- | --- | --- | --- | --- | --- |
| 0 | 4e-06 | 0 | 0.000118 | 0 | 0 |
| 5 | 0 | 0 | 0.000117 | 0 | 0 |
| 10 | 0 | 0 | 0.000126 | 0 | 0 |
| 15 | 0 | 0 | 0.000131 | 0 | 0 |
| 20 | 0 | 0 | 0.000132 | 0 | 0 |
| 25 | 0 | 0 | 0.00013 | 0 | 0 |
| 30 | 0 | 0 | 0.000141 | 0 | 0 |
| 35 | 1.1e-05 | 0 | 0.000151 | 1e-06 | 0 |
| 40 | 0 | 0 | 0.000166 | 0 | 0 |
| 45 | 3.3e-05 | 0 | 0.000175 | 2e-06 | 0 |
| 50 | 0 | 0 | 0.000194 | 0 | 0 |
| 55 | 0 | 0 | 0.000106 | 0 | 0 |
| 60 | 0 | 0 | 0.000143 | 0 | 0 |
| 65 | 0 | 0 | 0.000169 | 0 | 0 |
(aperture 0.6 m)

**STARLINK ← ONEWEB**, worst case (any visible), default aperture:

| latitude_deg | event_frac | unres_frac | dtt_mean | degrad_nodiv | degrad_div |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.4e-05 | 0 | 0.000144 | 1e-06 | 0 |
| 5 | 0 | 0 | 0.00014 | 0 | 0 |
| 10 | 0 | 0 | 0.000156 | 0 | 0 |
| 15 | 0 | 0 | 0.000158 | 0 | 0 |
| 20 | 0 | 0 | 0.000157 | 0 | 0 |
| 25 | 0 | 0 | 0.000152 | 0 | 0 |
| 30 | 0 | 0 | 0.000163 | 0 | 0 |
| 35 | 1.1e-05 | 0 | 0.000175 | 1e-06 | 0 |
| 40 | 0 | 0 | 0.000191 | 0 | 0 |
| 45 | 3.3e-05 | 0 | 0.000212 | 2e-06 | 0 |
| 50 | 0 | 0 | 0.000224 | 0 | 0 |
| 55 | 4e-06 | 0 | 0.000246 | 0 | 0 |
| 60 | 0 | 0 | 0.000318 | 0 | 0 |
| 65 | 7e-06 | 0 | 0.000361 | 0 | 0 |
(aperture 0.6 m)

Maximum ΔT/T > 6% event fraction 0.1464 (ONEWEB ← STARLINK, worst case (any visible), 0.45 m, latitude 40°).

Unresolvable fraction with unlimited diversity, maximum over the grid: 0.

## 4. As-deployed vs modelled Walker pattern

| pair | variant | e5_deployed_max | e1_walker_max |
| --- | --- | --- | --- |
| ONEWEB←STARLINK | co-located (serving) | 0.004876 | 0.004706 |
| ONEWEB←STARLINK | worst case (any visible) | 0.03575 | 0.02525 |
| STARLINK←ONEWEB | co-located (serving) | 0.004876 | 0.004706 |
| STARLINK←ONEWEB | worst case (any visible) | 0.009657 | 0.01033 |

The as-deployed snapshot includes the real inclination/altitude mix and on-orbit population; the modelled Walker pattern uses the FCC-derived shells with a single phasing. Differences are expected and reported, not tuned.

## 5. Verdict

- **Geometric criterion.** Co-located events are symmetric in the two directions (identical tables), as expected; the worst-case upper bound is larger for ONEWEB <- STARLINK and tracks the real Starlink density.
- **Power criterion (25.261(d)).** The asymmetry is decisive: only the ONEWEB victim <- STARLINK interferer direction exceeds Delta T/T > 6% (co-located up to ~0.015 of the time; worst-case up to ~0.15 at the narrowest ASSUMED aperture). The reverse direction is ~1e-5, negligible.
- Unresolvable fraction under unlimited diversity is 0 over the whole grid: the burden is fully removable by satellite diversity, as in E2/E4.
- The as-deployed snapshot reproduces the Phase-2/3 modelled conclusions qualitatively (ordering, the dominance of the worst-case bound, and the zero unresolvable fraction); absolute levels move with the actual on-orbit population and disposition and are reported, not tuned.
