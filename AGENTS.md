# AGENTS.md

Research project **ngso-sharing-incidence**: measure the geographic incidence of
NGSO-NGSO in-line interference events and the burden imposed by 47 CFR 25.261
spectrum-sharing rules (25.261(c) band-splitting, 25.261(d) degraded-throughput
protection), by earth-station latitude and later by population. This is new
research: **never hard-code or aim for expected results.**

## Non-negotiable rules

1. **Never invent, estimate, or round numbers in reports.** Every number must come
   from code output or a file under `results/`; cite the command or path.
2. **Every physical parameter lives in `configs/`** with a `source:` (FCC filing
   reference) or the literal `ASSUMED`. `ngsi.config` validation rejects anything else.
3. **Angle convention:** antenna gains use off-boresight angles *at the antenna*,
   never ground distances. Count each antenna gain exactly once.
4. **Figures/tables are generated only from `results/` files**, never from
   hard-coded arrays (`figures/make_figures.py`).
5. **Fixed seeds.** The unit of replication is an independent constellation phasing.
6. **Every new physics function gets a unit test with a hand-checkable value.**
7. **Work on git branches, small commits, and stop for approval at the end of each phase.**

## Commands

All work runs inside the dev container (GPU 0 only). Never run the pipeline on the host.

```bash
make precheck     # host: nvidia-smi + docker info | grep nvidia
make build        # build image (base: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime)
make up           # start detached container
make shell        # bash in container
make test         # python -m pytest -q, in container
make run-all      # scripts/run_all.sh: experiments + figures
make down
```

Single test: `docker compose -f docker/docker-compose.yml exec dev python -m pytest -q tests/test_frames.py -k round_trip`

Run an experiment directly:
`docker compose -f docker/docker-compose.yml exec dev python experiments/e1_event_rate.py`

## Environment

- `docker/docker-compose.yml` pins **GPU 0 only**: `device_ids: ["0"]`,
  `NVIDIA_VISIBLE_DEVICES=0`, `shm_size: 8gb`. Inside the container `nvidia-smi`
  and `torch.cuda.device_count()` must both report exactly **1**.
- The base image provides `torch`/CUDA; **do not add torch to `requirements.txt`**
  (a PyPI wheel would replace the CUDA build). Regenerate pins with
  `docker compose -f docker/docker-compose.yml exec dev python -m pip freeze`.
- The package uses a `src/` layout; imports work because the image sets
  `PYTHONPATH=/workspace/src` and the repo is bind-mounted at `/workspace`.

## Layout that matters

- `configs/` - all physics. `constellations.yaml`, `stations.yaml`,
  `experiments/*.yaml` (one per experiment).
- `src/ngsi/` - library. Key modules: `constellation` (Walker-Delta), `propagation`
  (batched torch two-body + NumPy fallback), `frames`, `geometry`, `antennas`
  (ITU-R S.1528 / S.465 / S.1428), `interference` (I/N, Delta T/T), `events`,
  `diversity`, `rules` (25.261 constants), `metrics`, `stats`, `io`.
- `experiments/` - thin entrypoints; heavy logic belongs in `src/ngsi/`.
- `results/<experiment>_<utc>_seed<N>/` - provenance: `run_meta.json` (git hash,
  seed, timestamp), copied `configs/`, raw tables. Created via `ngsi.io.new_run_dir`.
- `data/raw/` is gitignored; record source URL + download date in `data/raw/README.md`.

## Current status

Phase 1 scaffold only. Every module is a docstring + type-hinted stub; the tests
in `tests/` cover `frames`, `geometry`, `antennas`, `events`, `diversity`. The
`experiments/*.py` entrypoints validate config and stamp a run directory but do
not run the pipeline yet. Do not implement experiments without approval.
