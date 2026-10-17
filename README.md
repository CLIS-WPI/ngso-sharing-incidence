# ngso-sharing-incidence

Measure the **geographic incidence** of NGSO-NGSO in-line interference events and
the **burden** imposed by the 47 CFR 25.261 spectrum-sharing rules:

- **25.261(c)** - band-splitting between NGSO systems;
- **25.261(d)** - degraded-throughput protection (Delta T/T criteria).

Results are reported by earth-station **latitude** (0-65 deg, 5 deg steps) and,
in a later phase, weighted by **population / Broadband-Serviceable Locations**.

This is new research. Nothing in this repository aims at a predetermined result;
every reported number is traceable to a file under `results/` (see `AGENTS.md`).

## Setup

Requires an NVIDIA GPU and the NVIDIA Container Toolkit on the host.

```bash
make precheck     # verify host GPU + Docker NVIDIA runtime
make build        # build the dev image (PyTorch CUDA 12.4 runtime)
make up           # start the dev container (GPU 0 only)
make test         # run the test suite inside the container
```

`make shell` drops you into the container at `/workspace`.

## Reproducing every figure

```bash
make run-all      # experiments + figures, inside the container
```

`scripts/run_all.sh` runs each `experiments/*.py` and then
`figures/make_figures.py`. Every run writes a provenance-stamped directory
`results/<experiment>_<utc>_seed<N>/` containing `run_meta.json` (git hash, seed,
timestamp), a copy of the configs, and raw tables. Figures are built **only**
from files under `results/`.

## Layout

```
configs/        all physics parameters (each with source: or ASSUMED)
  constellations.yaml
  stations.yaml
  experiments/  one YAML per experiment
src/ngsi/       library (Walker-Delta, propagation, frames, geometry, antennas,
                interference, events, diversity, rules, capacity, demand,
                metrics, stats, io)
experiments/    thin entrypoints (e1_event_rate, e2_diversity)
figures/        make_figures.py (results/ -> figures/)
scripts/        run_all.sh
tests/          unit tests for the physics functions
data/raw/       gitignored inputs; see data/raw/README.md for sources
results/        run directories (heavy artifacts gitignored)
docker/         Dockerfile + docker-compose.yml (GPU 0 only)
```

## Status

Phase 1 scaffold: modules are typed stubs, tests cover the physics primitives,
and the experiment entrypoints validate configuration and stamp a run directory
but do not run the pipeline yet.
