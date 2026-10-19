# Phase 3c calibration report: OneWeb downlink EIRP and 95% CIs

This note records two checks requested before approval of the phase-3c letter.

All numbers below come from tracked files under `results/` or from the code that
writes them:

- `results/e4_dtt_20261019T031936Z_seed0/` (as-filed, 8 phasings)
- `results/e5_as_deployed_20261019T032128Z_seed0/` (as-deployed, 8 start times)
- `results/e6_eirp_sweep_20261019T040917Z_seed0/` (EIRP sweep, 4 start times)
- `scripts/make_numbers.py` -> `paper/numbers.tex` (macros quoted below)
- `scripts/calibrate_oneweb_eirp.py` (calibration rule)

## 1. del Portillo et al. (2019) OneWeb user-downlink parameters

del Portillo, Cameron & Crawley, "A technical comparison of three low earth
orbit satellite constellation systems to provide global broadband," *Acta
Astronautica* 159 (2019) 123-135. The relevant table is **Table 3, p. 5 of the
IAC preprint (IAC-18-B2.1.7)**, "Comparison of beam characteristics for the three
different systems", under the *User beam - Downlink* columns.

The paper reports a **per-beam EIRP in dBW and the channel bandwidth in MHz**
(not an EIRP density and not a link-budget C/N). The implied density is the
one-line unit conversion EIRP[dBW] - 10*log10(BW[MHz]):

| User downlink, per beam | EIRP [dBW] | BW [MHz] | Implied EIRP density [dBW/MHz] | Source |
|---|---|---|---|---|
| OneWeb   | 34.6  | 250 | **+10.6** | del Portillo 2019, Table 3, p. 5 |
| Starlink | 36.71 | 250 | **+12.7** | del Portillo 2019, Table 3, p. 5 |
| OneWeb (this model, calibrated) | -   | -   | **+0.2**  | `configs/constellations.yaml`; `scripts/calibrate_oneweb_eirp.py` |
| Starlink (this model)           | -   | -   | **-8.3**  | `configs/constellations.yaml` (FCC-22-91A1 para. 191) |

Notes on reading the table:

- OneWeb *gateway* downlink is a separate row (EIRP 38 dBW over 155 MHz); the
  155 MHz figure quoted in the text is the gateway channel, not the user beam.
- No EIRP density and no link-budget C/N appear anywhere in the paper; only the
  dBW-per-beam figures above are reported.

**Usability verdict.** The paper gives an absolute OneWeb user-beam EIRP, but it
is **not usable as a drop-in primary** in this model, and the calibration is kept
as primary:

1. It is a per-beam total in dBW, not a density; importing it requires dividing
   by a bandwidth.
2. Its implied density is **+10.4 dB above the calibrated +0.2 dBW/MHz**.
3. More decisively, the *same table's* Starlink value (+12.7 dBW/MHz) is
   **+21.0 dB above the FCC-derived Starlink value already used in this model**
   (-8.3 dBW/MHz). The two are therefore not on a common convention, and mixing
   one column of del Portillo with the model's FCC-based Starlink density would
   make OneWeb ~19 dB stronger than Starlink, contradicting the paper's own
   ordering (Starlink 2.1 dB above OneWeb). del Portillo is thus retained as an
   external cross-check only.

This is the basis for the paper's statement (Sec. II-A and Sec. V) that public
absolute values disagree by more than 10 dB and that the OneWeb EIRP density is
calibrated in the model's own scale rather than imported.

## 2. 95% CIs for every number compared to the 3% / 0.4% limits

CI = 95% t-interval over the replication unit (8 phasings as-filed, 8 start
times as-deployed, 4 start times for the sweep), for the specific
station-aperture cell (E4/E5) or sweep point (E6) that attains the quoted
extremum. Verdict rule: "exceeds" is used only when the whole CI lies above the
threshold; otherwise the value is "within" (CI entirely below) or "not
distinguishable" (CI straddles).

| Quantity | Run | Value +/- 95% CI | Limit | Verdict |
|---|---|---|---|---|
| As-filed, co-located throughput degradation | E4 | 0.43 +/- 0.01% | 3% | within |
| As-filed, co-located availability change | E4 | 0.243 +/- 0.017% | 0.4% | within |
| As-filed, worst-case throughput degradation | E4 | 2.05 +/- 0.06% | 3% | within |
| As-filed, worst-case availability change | E4 | 1.772 +/- 0.195% | 0.4% | exceeds |
| As-deployed, co-located throughput degradation | E5 | 0.53 +/- 0.03% | 3% | within |
| **As-deployed, co-located availability change** | E5 | **0.479 +/- 0.028%** | 0.4% | **exceeds** |
| As-deployed, worst-case throughput degradation | E5 | 3.19 +/- 0.03% | 3% | exceeds |
| As-deployed, worst-case availability change | E5 | 3.811 +/- 0.085% | 0.4% | exceeds |
| EIRP sweep, min qualified worst-case degradation | E6 | 1.67 +/- 0.06% | 3% | within |
| EIRP sweep, max qualified worst-case degradation | E6 | 3.30 +/- 0.08% | 3% | exceeds |
| EIRP sweep, value at the calibrated EIRP | E6 | 3.30 +/- 0.08% | 3% | exceeds |

The specifically requested as-deployed co-located availability change is
**0.479 +/- 0.028%** (CI [0.451, 0.507]%), entirely above 0.4%, so "exceeds" is
justified. The CI columns are `degradation_nodiv_ci95` and `unavail_incr_frac_ci95`
in the E4/E5 `summary_dtt.csv` files and the new `degr_max_qual_ci95` column in
the E6 `summary_eirp_sweep.csv` (added in `src/ngsi/runner.py::run_e6`).

## 3. Dependence of the Starlink-victim results on the calibrated EIRP

The Starlink-victim (reverse-direction) results depend on the calibrated OneWeb
EIRP. Across the +/-6 dB sweep (`EirpSweepValues` = -5.8, -2.8, 0.2, 3.2, 6.2
dBW/MHz) the reverse worst-case event fraction is monotone in the OneWeb EIRP:

| OneWeb EIRP [dBW/MHz] | -5.8 | -2.8 | 0.2 | 3.2 | 6.2 |
|---|---|---|---|---|---|
| Starlink-victim worst-case event fraction | 1.63% | 1.93% | 2.22% | 3.83% | 6.87% |

The result therefore holds throughout the sweep; the **minimum sweep EIRP at
which it still holds is -5.8 dBW/MHz** (calibration minus 6 dB), where the
reverse worst-case fraction is 1.63% (`EirpRevHoldMinDbw` / `EirpRevHoldPct` in
`paper/numbers.tex`). The paper states this in Sec. III-A.

The del Portillo value (+10.6 dBW/MHz) lies ~10.4 dB above the calibrated value
and outside this swept range; its effect is not quoted because it was not run.
