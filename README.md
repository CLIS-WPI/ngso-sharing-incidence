# Victim Handoff Policies and FCC Interference Protection in Starlink–OneWeb Coexistence

Starlink and OneWeb send internet signals to users on the same frequencies. When a
Starlink satellite crosses the line of sight between a OneWeb terminal and its
satellite, the OneWeb user receives interference. U.S. rules (47 CFR 25.261) let
OneWeb, the earlier-licensed system, lose at most 3% of its throughput and 0.4% of
its availability to Starlink's second generation. When the FCC checks a system
against these limits, it lets the analysis assume how the protected user picks its
satellite. This code measures how much that assumption changes the verdict. It
simulates both systems as filed and as deployed (a catalog snapshot of
2026-10-20), at earth stations from the equator to 65° latitude, and compares three
behaviours of the OneWeb user: staying on its satellite, switching away from every
interference event, or switching to whichever satellite gives the higher rate. With
the deployed satellites, the average worst-station throughput loss is 3.74%, 3.09%
and 2.05% for these three policies. So the same interference passes or fails the
3% limit depending only on the assumed policy. Avoiding every event can even make
things worse, because it can move the user to a weaker satellite. Compatibility
studies should therefore state and test the protected system's satellite-selection
policy.

## Requirements and how to run

Requirements: Docker with the NVIDIA Container Toolkit and one NVIDIA GPU (only
needed for `make run-all`). Run `make` from a POSIX shell (Git Bash on Windows).

```bash
make build        # build the container image
make up           # start it (the repository is mounted at /workspace)
make test         # run the test suite
make regenerate   # no GPU: output/numbers.tex and output/figures/ from the included results/
make run-all      # full re-run of every experiment, then numbers and figures
```

`make regenerate` reads only the run directories in `results/`. It writes the
paper's numbers (`output/numbers.tex`) and its two figures
(`output/figures/B1_events_both.png`, `output/figures/B2_limits.png`).

`make run-all` re-runs every experiment the paper uses. The runs with recorded
times took about 8 h of wall-clock time in total on one NVIDIA RTX A4000. The three
first-stage experiments (E1–E3) had no recorded time and come on top of that.

## Repository layout

```
configs/      physical parameters, station grid and experiment settings (YAML)
data/raw/     CelesTrak TLE snapshot of 2026-10-20, with sources and sha256 (README.md)
docker/       Dockerfile and docker-compose.yml (GPU 0)
experiments/  experiment entry points E1–E7
figures/      make_figures.py: the paper's two figures from results/
results/      the run directories behind every number and figure in the paper
scripts/      make_numbers.py, run_all.sh, and the E8–E14 analysis and check scripts
src/ngsi/     the simulation library (orbits, geometry, antennas, interference, 25.261 metrics)
tests/        pytest suite
```

## Citation

S. B. Hashemi Natanzi, H. Mohammadi, V. Marojevic, and B. Tang, "Victim Handoff
Policies and FCC Interference Protection in Starlink-OneWeb Coexistence," submitted
to IEEE Wireless Communications and Networking Conference (WCNC), 2027.

```bibtex
@inproceedings{natanzi2027victim,
  author    = {Hashemi Natanzi, Seyed Bagher and Mohammadi, Hossein and Marojevic, Vuk and Tang, Bo},
  title     = {Victim Handoff Policies and {FCC} Interference Protection in {Starlink}-{OneWeb} Coexistence},
  booktitle = {IEEE Wireless Communications and Networking Conference (WCNC)},
  year      = {2027},
  note      = {Submitted}
}
```

## License

MIT (see `LICENSE`).
