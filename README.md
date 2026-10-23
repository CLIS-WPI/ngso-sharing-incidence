# ngso-sharing-incidence: WCNC 2027 submission

Code, configurations, results and LaTeX sources for

> S. B. Hashemi Natanzi, H. Mohammadi, V. Marojevic, B. Tang,
> **Victim Handoff Policies and FCC Interference Protection in Starlink–OneWeb
> Coexistence**, submitted to IEEE WCNC 2027.

When interference from Starlink Gen2 (2020 processing round) hits a OneWeb user
(2016 round), 47 CFR 25.261(d) protects OneWeb up to 3% time-weighted throughput
degradation and 0.4% absolute availability change. The FCC lets a compatibility
analysis *assume* the incumbent's satellite-selection policy. We measure how much
that assumption matters. Both the filed constellation geometries and a 2026-10-20
CelesTrak snapshot of the deployed satellites are evaluated at 56 earth stations
(latitudes 0–65° in 5° steps, four meridians). For
each candidate satellite we recompute the desired signal and the aggregate
interference, then compare three victim policies against one interference-free
baseline. **Main finding:** in the deployed geometry (clear sky, Starlink at its
filed maximum PFD, 25° mask, unrestricted lossless switching), the mean
worst-station degradation is 3.74 ± 0.07% with no handoff, 3.09 ± 0.04% with
event-avoiding handoff and 2.05 ± 0.01% with rate-maximizing handoff. The verdict
against the 3% limit therefore depends on the assumed policy. Event-avoiding
handoff narrowly exceeds the limit because avoiding every in-line event can select
a weaker link. Both handoff policies meet the availability limit, but fail it with
a 60 s minimum dwell or a 55° mask. Compatibility showings should state and test
the incumbent's selection policy together with its switching constraints. (Every
number here is a `paper/numbers.tex` macro read from `results/`; the ± values are
95% t-intervals across days.)

This branch is a clean, single-commit snapshot of what the paper needs. It is tagged
`wcnc2027-submission`. The full development history (Phases 1–4, intermediate
reports, superseded runs, auxiliary checks) is on the
[`development`](https://github.com/CLIS-WPI/ngso-sharing-incidence/tree/development)
branch.

## Build the container

Requires Docker with the NVIDIA container toolkit and one CUDA GPU (the compose file
pins GPU 0). The image is `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` plus
`requirements.txt` and TeX Live (for the paper).

```bash
make precheck   # host: nvidia-smi + docker info | grep nvidia
make build      # build ngsi-dev:latest
make up         # start the container (repo bind-mounted at /workspace)
make test       # python -m pytest -q inside the container
```

Everything runs inside the container. On Windows, run `make` from Git Bash, or call
`docker compose -f docker/docker-compose.yml exec dev ...` directly.

## Regenerate numbers.tex and the figures from results/ (no GPU)

```bash
make regenerate   # scripts/make_numbers.py + figures/make_figures.py
make paper        # make_numbers.py + latexmk -> paper/main.pdf (6 pages)
```

`make_numbers.py` reads only the run directories under `results/` and rewrites
`paper/numbers.tex`, which is byte-identical to the committed file.
`make_figures.py` redraws `paper/figures/B1_events_both.png` and `B2_limits.png`.
`paper/main.tex` prints no number directly: every result is a macro from
`numbers.tex`, and each macro carries a `% Name <- results/<run>/` source comment.

## Re-run the experiments

```bash
make run-all      # scripts/run_all.sh
```

`scripts/run_all.sh` repeats each run the paper cites with the arguments and
precision recorded in that run's `run_meta.json`. It then regenerates the numbers,
the figures and the paper. New run directories get new timestamps, so they become
the newest runs and are the ones that `make_numbers.py` and `make_figures.py` pick
up. The as-deployed runs read the committed TLE snapshot in `data/raw/tle/`;
nothing is downloaded.

GPU time, observed on one NVIDIA RTX A4000 (16 GB). Times are wall-clock from the
run-directory timestamp to its `status.json`:

| run(s) | wall-clock |
|---|---|
| E4 / E5 / E6 headline (Gen2), run concurrently on the one GPU | 44 / 104 / 173 min |
| E4 Gen1, E4 all-Starlink | 21, 33 min |
| E5 Gen1, E5 all-Starlink (4 days each) | 13, 18 min |
| E5 sensitivity: Gen2 43° shell only; 2 s step (2 days) | 24, 25 min |
| E7, E8, E9, E14 | < 1 min each |
| E1, E2, E3 | not recorded (see note) |

The E4–E14 runs total about 7.6 h of wall-clock time. The headline three ran
concurrently, so run alone each should take no longer than listed.

Note on provenance: each `run_meta.json` records the commit that produced the run.
The E4–E9, E14 and sensitivity runs were produced by commits whose `src/`,
`experiments/` and `configs/` are identical to this snapshot. E1, E2 and E3 come
from earlier commits (2026-10-17 and 2026-10-19, on `development`). Their
`status.json` was rewritten after the runs, so their run time is unknown, and
re-running them with this code may not reproduce them exactly. The paper uses E1
only for setup values (shared-band centre frequency, number of phasings, screening
angle), and it
prints no E2/E3 value; E2/E3 macros exist only in `numbers.tex`.

## Paper → scripts → run directories

| paper item | produced by | run directory (`results/…`) |
|---|---|---|
| Table I (parameters) | `make_numbers.py` from the runs' copied configs | `e4_dtt_20261021T174655Z_seed0`, `e6_eirp_sweep_20261021T174655Z_seed0` |
| Fig. 1 (ΔT/T > 6% time fraction vs latitude) | `make_figures.py` `fig_b1_events` | `e4_dtt_20261021T174655Z_seed0` (as filed), `e5_as_deployed_20261021T174655Z_seed0` (as deployed) |
| Table II (25.261(d) metrics by handoff policy and constraint) | `make_numbers.py` | `e5_as_deployed_20261021T174655Z_seed0` (policies, K = 2, dwell), `e6_eirp_sweep_20261021T174655Z_seed0` (55° mask), `e14_interruption_cost_20261021T204021Z_seed0` (τ rows) |
| Fig. 2 (degradation / availability change vs latitude) | `make_figures.py` `fig_b2_limits` | `e5_as_deployed_20261021T174655Z_seed0` |
| Table III (sensitivity cases) | `make_numbers.py` | `e6_eirp_sweep_20261021T174655Z_seed0` |
| Text: Starlink generation split | `make_numbers.py` | `e4_dtt_20261021T185452Z_seed0`, `e5_as_deployed_20261021T192749Z_seed0`, `e5_as_deployed_20261021T194046Z_seed0` (`e4_dtt_20261021T183342Z_seed0` feeds `numbers.tex` only) |
| Text: Gen2 43° only, 2 s time step | `make_numbers.py` | `sensitivity/e5_as_deployed_20261021T193108Z_seed0`, `sensitivity/e5_as_deployed_20261021T200810Z_seed0` |
| Text: analytic cross-check | `experiments/e7_analytic.py`, `scripts/diagnose_analytic.py` | `e7_analytic_20261021T183102Z_seed0`, `e8_analytic_diagnosis_20261021T183333Z_seed0` |
| Text: OneWeb 55° mask diagnosis | `scripts/diagnose_min_elevation.py` | `e9_min_elevation_diagnosis_20261021T200803Z_seed0` |
| Text: interruption cost | `scripts/interruption_cost.py` | `e14_interruption_cost_20261021T204021Z_seed0` |
| Text: band, phasings | `make_numbers.py` | `e1_event_rate_20261018T012116Z_seed0` |
| `numbers.tex` only (not printed) | `make_numbers.py` | `e2_diversity_20261019T145056Z_seed0`, `e3_capacity_diversity_20261018T012933Z_seed0` |

### Evidence for statements without a printed number

These checks back qualitative statements in the text (or a modelling choice). No
`numbers.tex` macro reads them, so `make_numbers.py` does not depend on them.

| check | `main.tex` line it supports | script | run directory (`results/…`) | recorded outcome |
|---|---|---|---|---|
| E10 bandwidth invariance | 244–245: "Both metrics are invariant to the channel bandwidth." | `scripts/check_bandwidth_invariance.py` | `e10_bandwidth_invariance_20261020T165626Z_seed0` | `passed: true`. Every `summary_*.csv` of an E4 run with 250/250 MHz channels matches one with 240/125 MHz within tolerance 1e-9 (largest difference 4.4e-16). |
| E11 float32 vs float64 | none (modelling choice): every E4–E9 and sensitivity run uses `NGSI_COMPUTE_DTYPE=float32` | `scripts/compare_precision.py` | `e11_precision_check_20261020T171436Z_seed0` | `printed_paper_numbers_identical: true` for the 84 E4 paper macros of that time; per-column differences are in `precision_check.json`. |
| E12 J2 vs SGP4 | 618–620: "Without drag, $J_2$ propagation drifts from SGP4 for individual Starlink satellites after the first days, mostly along track, so later days act as further phasings …" | `scripts/check_propagation.py` | `e12_propagation_check_20261020T224422Z_seed0` | `summary_sample_errors.csv`: position (radial / total) and look-angle errors at days 0, 1, 4, 8; `summary_daily_geometry.csv`: per-day visible counts and in-line fraction, J2 vs SGP4. |
| E13 daily dependence | 621–622: "the visible-count and event statistics we use show no trend across days. As-deployed intervals describe day-to-day variation …" | `scripts/analyze_daily_replications.py` | `e13_daily_replications_20261021T204049Z_seed0` | `summary_daily_stats.csv`: lag-1 autocorrelation, OLS trend and Kendall's τ per headline metric over the 8 days; CI basis `t-interval` for 59 metrics and `constant` (zero spread) for 24. |

Re-running them:

- **E12, E13:** reproducible from this branch (`run_all.sh` runs both). E12 reads the
  TLE snapshot and the E5 config (about 17 min on the GPU). E13 post-processes the
  kept E5 runs (under 1 min). E12 was produced at commit `0724e96`, and `src/` and
  `configs/` have changed since then, so a re-run may differ slightly. E13 re-run on
  this branch reproduces both of its CSVs byte-identical.
- **E10, E11:** the recorded outcomes above are final. Their inputs are on
  `development` only. E10 compares `e4_dtt_20261020T151750Z_seed0` against
  `e4_dtt_20261019T183747Z_seed0`. E11 compares the float32/float64 pairs under
  `data/processed/precision_float{32,64}/`. Keeping those superseded E4 runs out of
  `results/` here also keeps them out of the newest-run selection. To re-run either
  check, check out `development`.

`results/PHASE4C_REPORT.md` is the final report behind the paper: verdict tables
and the dwell, interruption and time-step analyses. Its generator
(`scripts/make_phase4c_report.py`) also reads superseded runs, so it is on
`development` only; here the report is a frozen copy.

## Layout

```
configs/       all physics (constellations, stations, bands, link budget, experiments/*.yaml)
src/ngsi/      library; runner.py holds the experiment orchestration
experiments/   e1–e7 entrypoints (CLI flags -> ngsi.runner.run_eN)
scripts/       make_numbers.py, build_paper.sh, run_all.sh, E8/E9/E14 scripts, E10–E13 checks
figures/       make_figures.py (Fig. 1, Fig. 2)
tests/         pytest suite (unit tests with hand-checkable values, paper-macro check)
results/       the 16 run directories numbers.tex cites, the E10–E13 checks, PHASE4C_REPORT.md
data/raw/tle/  CelesTrak TLE snapshot 2026-10-20 (sources and sha256 in data/raw/README.md)
paper/         main.tex, numbers.tex (generated), refs.bib, figures/
docker/        Dockerfile, docker-compose.yml
```

`AGENTS.md` lists the project rules: numbers come only from `results/`, every
parameter in `configs/` carries a `source:`, seeds are fixed, and runs execute only
in the container.

## License

MIT; see `LICENSE`. Copyright (c) 2026 the authors / Worcester Polytechnic Institute.
