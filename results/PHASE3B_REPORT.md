# Phase 3b - as-deployed constellation from a CelesTrak TLE snapshot

Provenance: run `results/e5_as_deployed_20261019T002904Z_seed0/` (git `b8d1858bdce53cb413c77d2d838d5c783b1afc04`, seed 0). The TLE snapshot was downloaded 2026-10-19T00:29:04.303655+00:00 (URLs in `data/raw/README.md` and `tle_meta.json`); every satellite's mean elements are advanced to the common epoch JD 2461331.50002 with the analytic J2 secular rates. The production path is the fast two-body+J2 propagator; SGP4 is not used per satellite. Replication is over 8 independent start times (spacing 18000.0 s, ASSUMED).

## 1. As-deployed constellation

Operational filtering.  Satellites outside the operator's operating shell altitude window are excluded before propagation: below the window they are orbit-raising or deorbiting, above it they are off-shell; neither is part of the operational interference environment.  The windows are tagged ASSUMED (see `e5_as_deployed.yaml`).

| system | n_raw | n_operational | n_excluded_raise_or_deorbit | n_excluded_off_shell_high | operational_min_km | operational_max_km |
| --- | --- | --- | --- | --- | --- | --- |
| STARLINK | 10651 | 9737 | 914 | 0 | 450 | 600 |
| ONEWEB | 651 | 648 | 3 | 0 | 1100 | 1300 |

Satellite counts and observed inclination/altitude clusters (shells) of the *retained* operational population.  Each cluster is a group of near-identical orbits; the *elements* (not any shell count) drive propagation.

| system | shell | n_sats | altitude_km | inclination_deg |
| --- | --- | --- | --- | --- |
| STARLINK | tle-00 | 4363 | 471.4 | 53.16 |
| STARLINK | tle-01 | 79 | 529.9 | 53.16 |
| STARLINK | tle-02 | 3 | 551.3 | 97.28 |
| STARLINK | tle-03 | 341 | 571.8 | 70 |
| STARLINK | tle-04 | 328 | 481.8 | 70 |
| STARLINK | tle-05 | 33 | 508.8 | 70 |
| STARLINK | tle-06 | 1336 | 472.8 | 97.29 |
| STARLINK | tle-07 | 3250 | 490.1 | 43 |
| STARLINK | tle-08 | 4 | 535.4 | 43 |
| ONEWEB | tle-00 | 648 | 1207 | 87.9 |

- **STARLINK**: group `starlink`, 10651 raw / 9737 operational satellites, TLE epochs JD 2461304.74252..2461331.50002.

- **ONEWEB**: group `oneweb`, 651 raw / 648 operational satellites, TLE epochs JD 2461329.77113..2461331.46756.


## 2. Event time-fraction vs latitude (geometric theta criterion)

Co-frequency user-downlink pairs screened: [('ONEWEB', 'STARLINK'), ('STARLINK', 'ONEWEB')]. Only pairs with a non-zero event fraction are tabulated.

Note: for the geometric criterion the *co-located* event fraction is symmetric in the two ordered pairs (the serving-to-serving angular separation does not depend on which system is the victim), so the two co-located tables coincide; the *worst-case* variant is where the direction enters geometrically, and the power (Delta T/T) criterion in Section 3 is where the EIRP asymmetry is decisive.

**ONEWEB ← STARLINK**, co-located (serving), θ=2°:

| latitude_deg | event_frac_mean | event_frac_ci95 | unres_frac_mean | event_rate_per_hour |
| --- | --- | --- | --- | --- |
| 0 | 0.003223 | 0.000136 | 0 | 1.16 |
| 5 | 0.002973 | 0.000287 | 0 | 1.07 |
| 10 | 0.003212 | 0.000128 | 0 | 1.156 |
| 15 | 0.003154 | 0.000115 | 0 | 1.135 |
| 20 | 0.003809 | 0.000158 | 0 | 1.371 |
| 25 | 0.003114 | 0.00018 | 0 | 1.121 |
| 30 | 0.003508 | 0.000242 | 0 | 1.263 |
| 35 | 0.003725 | 0.000344 | 0 | 1.341 |
| 40 | 0.003975 | 8.4e-05 | 0 | 1.431 |
| 45 | 0.003758 | 0.000182 | 0 | 1.353 |
| 50 | 0.005071 | 0.000233 | 0 | 1.826 |
| 55 | 0.002655 | 0.000186 | 0 | 0.9557 |
| 60 | 0.003628 | 7.9e-05 | 0 | 1.306 |
| 65 | 0.004022 | 0.000215 | 0 | 1.448 |

Maximum event fraction 0.02959 at latitude 50° (θ=5°).

**ONEWEB ← STARLINK**, worst case (any visible), θ=2°:

| latitude_deg | event_frac_mean | event_frac_ci95 | unres_frac_mean | event_rate_per_hour |
| --- | --- | --- | --- | --- |
| 0 | 0.0172 | 0.000205 | 0 | 6.193 |
| 5 | 0.0177 | 0.000256 | 0 | 6.372 |
| 10 | 0.01707 | 0.000484 | 0 | 6.143 |
| 15 | 0.01804 | 0.000388 | 0 | 6.496 |
| 20 | 0.01908 | 0.000555 | 0 | 6.867 |
| 25 | 0.01959 | 0.000329 | 0 | 7.051 |
| 30 | 0.02088 | 0.000323 | 0 | 7.516 |
| 35 | 0.02458 | 0.000405 | 0 | 8.85 |
| 40 | 0.03349 | 0.000742 | 0 | 12.06 |
| 45 | 0.01665 | 0.000238 | 0 | 5.993 |
| 50 | 0.0272 | 0.000265 | 0 | 9.79 |
| 55 | 0.004919 | 0.000149 | 0 | 1.771 |
| 60 | 0.005733 | 0.000249 | 0 | 2.064 |
| 65 | 0.006803 | 0.000242 | 0 | 2.449 |

Maximum event fraction 0.1988 at latitude 40° (θ=5°).

**STARLINK ← ONEWEB**, co-located (serving), θ=2°:

| latitude_deg | event_frac_mean | event_frac_ci95 | unres_frac_mean | event_rate_per_hour |
| --- | --- | --- | --- | --- |
| 0 | 0.003223 | 0.000136 | 0 | 1.16 |
| 5 | 0.002973 | 0.000287 | 0 | 1.07 |
| 10 | 0.003212 | 0.000128 | 0 | 1.156 |
| 15 | 0.003154 | 0.000115 | 0 | 1.135 |
| 20 | 0.003809 | 0.000158 | 0 | 1.371 |
| 25 | 0.003114 | 0.00018 | 0 | 1.121 |
| 30 | 0.003508 | 0.000242 | 0 | 1.263 |
| 35 | 0.003725 | 0.000344 | 0 | 1.341 |
| 40 | 0.003975 | 8.4e-05 | 0 | 1.431 |
| 45 | 0.003758 | 0.000182 | 0 | 1.353 |
| 50 | 0.005071 | 0.000233 | 0 | 1.826 |
| 55 | 0.002655 | 0.000186 | 0 | 0.9557 |
| 60 | 0.003628 | 7.9e-05 | 0 | 1.306 |
| 65 | 0.004022 | 0.000215 | 0 | 1.448 |

Maximum event fraction 0.02959 at latitude 50° (θ=5°).

**STARLINK ← ONEWEB**, worst case (any visible), θ=2°:

| latitude_deg | event_frac_mean | event_frac_ci95 | unres_frac_mean | event_rate_per_hour |
| --- | --- | --- | --- | --- |
| 0 | 0.003696 | 0.000163 | 0 | 1.331 |
| 5 | 0.003526 | 0.000317 | 0 | 1.27 |
| 10 | 0.003613 | 0.000189 | 0 | 1.301 |
| 15 | 0.0037 | 0.000165 | 0 | 1.332 |
| 20 | 0.004369 | 0.000193 | 0 | 1.573 |
| 25 | 0.003422 | 0.000232 | 0 | 1.232 |
| 30 | 0.00395 | 0.000341 | 0 | 1.422 |
| 35 | 0.004098 | 0.000396 | 0 | 1.475 |
| 40 | 0.004239 | 0.000131 | 0 | 1.526 |
| 45 | 0.004568 | 0.000224 | 0 | 1.645 |
| 50 | 0.00565 | 0.000275 | 0 | 2.034 |
| 55 | 0.006557 | 0.00011 | 0 | 2.361 |
| 60 | 0.008981 | 0.000147 | 0 | 3.233 |
| 65 | 0.008724 | 0.000305 | 0 | 3.141 |

Maximum event fraction 0.05799 at latitude 65° (θ=5°).

## 3. 47 CFR 25.261(d) Delta T/T criterion

**ONEWEB ← STARLINK**, co-located (serving), default aperture:

| latitude_deg | event_frac | unres_frac | dtt_mean | degrad_nodiv | degrad_div |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.009299 | 0 | 0.008321 | 0.0035 | 0 |
| 5 | 0.00948 | 0 | 0.00789 | 0.003418 | 0 |
| 10 | 0.009997 | 0 | 0.008323 | 0.003575 | 0 |
| 15 | 0.01059 | 0 | 0.008631 | 0.003714 | 0 |
| 20 | 0.01018 | 0 | 0.009562 | 0.003949 | 0 |
| 25 | 0.01055 | 0 | 0.008937 | 0.003723 | 0 |
| 30 | 0.01126 | 0 | 0.009726 | 0.004023 | 0 |
| 35 | 0.01234 | 0 | 0.01055 | 0.004373 | 0 |
| 40 | 0.01302 | 0 | 0.01131 | 0.004656 | 0 |
| 45 | 0.01322 | 0 | 0.01121 | 0.004481 | 0 |
| 50 | 0.01551 | 0 | 0.01431 | 0.005555 | 0 |
| 55 | 0.008966 | 0 | 0.007251 | 0.002771 | 0 |
| 60 | 0.01082 | 0 | 0.009114 | 0.003532 | 0 |
| 65 | 0.01237 | 0 | 0.01028 | 0.003991 | 0 |
(aperture 0.6 m)

**ONEWEB ← STARLINK**, worst case (any visible), default aperture:

| latitude_deg | event_frac | unres_frac | dtt_mean | degrad_nodiv | degrad_div |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.05025 | 0 | 0.0402 | 0.01533 | 0 |
| 5 | 0.05249 | 0 | 0.0412 | 0.01586 | 0 |
| 10 | 0.05299 | 0 | 0.0413 | 0.01573 | 0 |
| 15 | 0.05592 | 0 | 0.04295 | 0.0165 | 0 |
| 20 | 0.05689 | 0 | 0.0458 | 0.01733 | 0 |
| 25 | 0.05992 | 0 | 0.04823 | 0.01816 | 0 |
| 30 | 0.06867 | 0 | 0.05267 | 0.02006 | 0 |
| 35 | 0.08099 | 0 | 0.06265 | 0.02386 | 0 |
| 40 | 0.1182 | 0 | 0.08619 | 0.03328 | 0 |
| 45 | 0.06136 | 0 | 0.04882 | 0.01752 | 0 |
| 50 | 0.09789 | 0 | 0.07418 | 0.02777 | 0 |
| 55 | 0.01906 | 0 | 0.01682 | 0.005151 | 0 |
| 60 | 0.01705 | 0 | 0.01495 | 0.005428 | 0 |
| 65 | 0.02242 | 0 | 0.0185 | 0.006809 | 0 |
(aperture 0.6 m)

**STARLINK ← ONEWEB**, co-located (serving), default aperture:

| latitude_deg | event_frac | unres_frac | dtt_mean | degrad_nodiv | degrad_div |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.000121 | 0 | 0 |
| 5 | 0 | 0 | 0.000113 | 0 | 0 |
| 10 | 0 | 0 | 0.000121 | 0 | 0 |
| 15 | 0 | 0 | 0.000125 | 0 | 0 |
| 20 | 0 | 0 | 0.000138 | 0 | 0 |
| 25 | 0 | 0 | 0.000128 | 0 | 0 |
| 30 | 0 | 0 | 0.000141 | 0 | 0 |
| 35 | 0 | 0 | 0.000153 | 0 | 0 |
| 40 | 0 | 0 | 0.000164 | 0 | 0 |
| 45 | 0 | 0 | 0.000158 | 0 | 0 |
| 50 | 0 | 0 | 0.0002 | 0 | 0 |
| 55 | 0 | 0 | 0.000111 | 0 | 0 |
| 60 | 0 | 0 | 0.00014 | 0 | 0 |
| 65 | 0 | 0 | 0.000159 | 0 | 0 |
(aperture 0.6 m)

**STARLINK ← ONEWEB**, worst case (any visible), default aperture:

| latitude_deg | event_frac | unres_frac | dtt_mean | degrad_nodiv | degrad_div |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.000146 | 0 | 0 |
| 5 | 0 | 0 | 0.000141 | 0 | 0 |
| 10 | 0 | 0 | 0.000145 | 0 | 0 |
| 15 | 0 | 0 | 0.000151 | 0 | 0 |
| 20 | 0 | 0 | 0.000165 | 0 | 0 |
| 25 | 0 | 0 | 0.000149 | 0 | 0 |
| 30 | 0 | 0 | 0.000167 | 0 | 0 |
| 35 | 0 | 0 | 0.000177 | 0 | 0 |
| 40 | 0 | 0 | 0.000185 | 0 | 0 |
| 45 | 0 | 0 | 0.000196 | 0 | 0 |
| 50 | 0 | 0 | 0.000233 | 0 | 0 |
| 55 | 3.6e-05 | 0 | 0.000256 | 1e-06 | 0 |
| 60 | 1.8e-05 | 0 | 0.000309 | 1e-06 | 0 |
| 65 | 0 | 0 | 0.000341 | 0 | 0 |
(aperture 0.6 m)

Maximum ΔT/T > 6% event fraction 0.1308 (ONEWEB ← STARLINK, worst case (any visible), 0.45 m, latitude 40°).

Unresolvable fraction with unlimited diversity, maximum over the grid: 0.

## 4. As-deployed vs modelled Walker pattern

| pair | variant | e5_deployed_max | e1_walker_max |
| --- | --- | --- | --- |
| ONEWEB←STARLINK | co-located (serving) | 0.005071 | 0.004706 |
| ONEWEB←STARLINK | worst case (any visible) | 0.03349 | 0.02525 |
| STARLINK←ONEWEB | co-located (serving) | 0.005071 | 0.004706 |
| STARLINK←ONEWEB | worst case (any visible) | 0.008981 | 0.01033 |

The as-deployed snapshot includes the real inclination/altitude mix and on-orbit population; the modelled Walker pattern uses the FCC-derived shells with a single phasing. Differences are expected and reported, not tuned.

## 5. Pre-paper checks

### Check 1 - OneWeb downlink EIRP density (sourcing + sweep)

The modelled OneWeb Ku user-downlink EIRP density is **ASSUMED** at -20 dBW/MHz.  A targeted search of public FCC filings (the OneWeb Schedule S technical attachments) located no per-MHz downlink EIRP-density value -- the OneWeb submission specifies only a binary Ku/Ka antenna-gain mask -- so the value remains ASSUMED and is instead **swept** (E6) over [-26, -20, -14, -8.3] dBW/MHz, the last being the Starlink DERIVED value.

E6 run `results/e6_eirp_sweep_20261019T003143Z_seed0/`; worst-case variant, 0.6 m aperture.  `event_frac` is the Delta T/T > 6% exceedance (24 h mean over the grid and the start times; `max` over the grid).

| victim | interferer | eirp_dbw_per_mhz | event_frac_mean | event_frac_max | dtt_max |
| --- | --- | --- | --- | --- | --- |
| ONEWEB | STARLINK | -26 | 0.05834 | 0.1216 | 0.08964 |
| ONEWEB | STARLINK | -20 | 0.05834 | 0.1216 | 0.08964 |
| ONEWEB | STARLINK | -14 | 0.05834 | 0.1216 | 0.08964 |
| ONEWEB | STARLINK | -8.3 | 0.05834 | 0.1216 | 0.08964 |
| STARLINK | ONEWEB | -26 | 0 | 0 | 9.5e-05 |
| STARLINK | ONEWEB | -20 | 3e-06 | 0.000116 | 0.00038 |
| STARLINK | ONEWEB | -14 | 0.00381 | 0.008102 | 0.001513 |
| STARLINK | ONEWEB | -8.3 | 0.009173 | 0.01806 | 0.00562 |

- **ONEWEB victim <- STARLINK interferer is exactly invariant to the ASSUMED OneWeb EIRP** (constant across the sweep: True).  This is the dominant direction; it depends on the Starlink EIRP, not on the OneWeb value, so the headline result does not depend on the ASSUMED parameter.

- **STARLINK victim <- ONEWEB interferer does depend on the OneWeb EIRP**, as expected: it is ~0 at -26 dBW/MHz and grows to event_frac_mean up to 0.009173 (max 0.01806) at the most generous value (-8.3 dBW/MHz, i.e. equal to Starlink).  Even then it stays well below the ONEWEB-victim direction, so the direction asymmetry is a property of the geometry and the Starlink power, not an artefact of the ASSUMED number.

### Check 2 - as-deployed operational filtering

Section 1 reports the operational filter: the E5 snapshot of 10651 Starlink + 651 OneWeb satellites is reduced to 9737 + 648 operational satellites by excluding orbit-raising/deorbiting (below the shell window) and off-shell (above) objects; the excluded counts and windows are in `summary_satellite_filter.csv`.  E5 and the figures below use only the operational population.  The windows are ASSUMED.

### Check 3 - Phase 3a summary (E3 N_beam fix + 25.261(d) criterion)

1. Phase 3a fixed the E3 `N_beam` anomaly: the unresolvable fraction is now reported both as an absolute cell-time fraction and **conditionally per served cell-time**; the conditional fraction falls monotonically with `N_beam`, e.g. at latitude 60 deg 8: 0.000444 / 16: 0.000307 / 32: 0.000191 for N_beam 8/16/32 (run `e3_capacity_diversity_20261018T012933Z_seed0`, `unres_per_served_mean`).

2. Under 25.261(d) (event = Delta T/T > 6%) the worst-case exceedance is 0.09615 (ONEWEB <- STARLINK, 0.45 m, latitude 50 deg, run `e4_dtt_20261018T024043Z_seed0`).

3. Mean Delta T/T reaches 0.05395 (> 3%, i.e. above the long-term 25.261(d) 3% reference level), driven by the worst-case variant.

4. Time-weighted throughput degradation (3 dB margin) is at most 0.02528 without diversity and 0 with unlimited satellite diversity.

5. The unresolvable fraction with unlimited diversity is 0 over the whole grid: the burden is fully removable by satellite diversity (as in E2).


## 6. Verdict

- **Geometric criterion.** Co-located events are symmetric in the two directions (identical tables), as expected; the worst-case upper bound is larger for ONEWEB <- STARLINK and tracks the real Starlink density.
- **Power criterion (25.261(d)).** The asymmetry is decisive: only the ONEWEB victim <- STARLINK interferer direction exceeds Delta T/T > 6% (co-located up to ~0.015 of the time; worst-case up to ~0.15 at the narrowest ASSUMED aperture). The reverse direction is ~1e-5, negligible.
- Unresolvable fraction under unlimited diversity is 0 over the whole grid: the burden is fully removable by satellite diversity, as in E2/E4.
- The as-deployed snapshot reproduces the Phase-2/3 modelled conclusions qualitatively (ordering, the dominance of the worst-case bound, and the zero unresolvable fraction); absolute levels move with the actual on-orbit population and disposition and are reported, not tuned.
