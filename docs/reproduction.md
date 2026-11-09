# Reproduction instructions

Run commands from the repository root using Python 3.12. The dependency files pin the package versions present in the original `hal` environment. `requirements-check.txt` is sufficient for data integrity, statistical checks, table reconstruction, and hardware-cost reconstruction. `requirements.txt` adds code construction, circuit compilation, decoding, and HAL routing, including packages imported by HAL.

All supported entry points use `python -m tools.artifact`. They put generated files under `outputs/<command>/`. An alternative parent directory can be supplied with `--output PATH`. Archived source and data directories are protected from use as output locations. The original experiment scripts retain their original output defaults; use the artifact entry points for routine reproduction.

## Archived statistics and tables

```bash
python -m tools.artifact verify
python -m tools.artifact tables
python -m tools.artifact costs
```

The statistical verifier calculates Clopper–Pearson 95% intervals directly from failure counts and recomputes two-sided pooled two-proportion tests and Holm correction. Families are 24 selected-graph comparisons, 48 accepted-graph comparisons, 72 fixed-noise comparisons, four SABRE comparisons, and 24 corrected timing comparisons. Zero-failure baselines leave relative improvement undefined; those conditions still belong to their Holm family.

The table exporter checks the matched edge sets and preserves the archived comparison families. The 48-row hardware table is recomputed from the full local graph and accepted added paths. Its CSV must be byte-identical to the archive.

## Physical routing

```bash
python -m tools.artifact replay --limit 1
python -m tools.artifact replay
python -m tools.artifact scan
```

The replay uses layout seed zero, endpoint length bound six, and budgets of 2% and 5%. HAL's TSV and bump limit parameters are ten; the relative length parameter resolves to the existing default of 1,000. The limit parameters and measured resource counts follow different definitions. Each route passes the independent three-dimensional geometric checker before comparison with the archived accepted edges.

Replay uses one worker, a memory guard, an exclusive lock file, and atomic per-condition writes. On restart it checks source hashes and skips completed conditions. A lock left by an interrupted process must be checked before removal. Do not change routing sources while a replay is running.

`scan` reruns the 24-code study at 1%, 2%, 5%, and 10% budgets. It matches accepted edges and three-round average SWAP counts for all 96 conditions. `--limit 4` provides a short check. This command runs sequentially; it restarts the requested scan when invoked again.

## Circuits and random-link baseline

```bash
python -m tools.artifact random-check
python -m tools.artifact sample --code D4_3 --budget 0.05
```

The random check uses the saved edge sets and starts every trial from the same initial layout for one syndrome-extraction round. It does not redraw random links or use the earlier three-round averages.

The sample command uses the existing accepted-graph implementation: three memory rounds, 3,000 shots, planar-calibrated physical noise, BP–OSD decoding, and the experiment's deterministic seed. It requires exact accepted-edge agreement and reports whether the sampled failures match the archive. The raw seed recipe and parameters are in `foldq/experiments/exp33_workpoint_ler.py`. The command evaluates one condition; it does not rerun or re-estimate a 48-comparison significance family.

The original implementations for selected-graph, fixed-noise, SABRE, and timing experiments are distributed with their inputs. Complete reruns involve new sampling and may take substantially longer than checking archived evidence. Their saved runtime fields are per-condition observations, not portable runtime guarantees. See [data-map.md](data-map.md) for the exact study-to-file mapping and [provenance.md](provenance.md) for unavailable historical snapshots.

## Tests

```bash
python -m pytest foldq/tests -q -p no:cacheprovider
```

Tests cover code commutation and logical operators, SWAP and circuit semantics, three-dimensional geometry including degenerate and reversed segments, and hardware-cost boundary cases. Release checks and their execution scope are recorded in `provenance/validation.json`.
