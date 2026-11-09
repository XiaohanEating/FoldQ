# FoldQ

Code and data for **FoldQ: Logical Folding of qLDPC Codes onto Superconducting Hardware**.

FoldQ selects a budgeted set of additional connections for a fixed qLDPC workload, routes them using HAL, checks their three-dimensional geometry, and compiles the workload on the accepted graph. This repository contains core code, archived measurements, complete replayed routing paths, and commands to reconstruct the numerical results.

## Start with the archived results

Use Python 3.12. From the repository root:

```bash
python -m pip install -r requirements-check.txt
python -m tools.artifact verify
python -m tools.artifact tables
python -m tools.artifact costs
```

`verify` checks the release hashes, recomputes exact binomial confidence intervals and Holm corrections, and independently recalculates all 48 hardware costs from the stored paths. `tables` rebuilds eight CSV files and compares their bytes with the archived tables. `costs` rebuilds the 48-row hardware table. Outputs go under `outputs/`; archived inputs stay in place.

Expected significant LER improvements are 21/24 for selected graphs, 4/24 and 6/24 for accepted graphs at 2% and 5% budgets, 46/72 for the fixed-noise comparisons, and 3/4 for SABRE-compiled memory circuits. The accepted-graph budgets share one 48-test Holm family.

## Rebuild circuits and routes

Install the full experimental dependencies in your own environment:

```bash
python -m pip install -r requirements.txt
python -m pytest foldq/tests -q -p no:cacheprovider
python -m tools.artifact replay
python -m tools.artifact scan
python -m tools.artifact random-check
python -m tools.artifact sample --code D4_3 --budget 0.05
```

- `replay` reruns all 48 physical designs and requires the accepted edge sets and conflict decisions to match the archived LER conditions. `--limit 1` provides a short smoke run; rerunning without a limit resumes it.
- `scan` reruns the 96 budget-scan conditions and checks both accepted edges and the three-round SWAP averages against the archive.
- `random-check` recompiles the 1,050 saved random edge sets using the single-round protocol used in the paper. `--limit 3` checks three codes.
- `sample` rebuilds one accepted-graph memory circuit and runs the original 3,000-shot protocol and seed. It compares the resulting counts and topology with the archive. Change `--code` and `--budget` to choose another archived condition.

The archived experiments used Windows, Python 3.12.11, and the pinned versions in `requirements.txt`. A fixed seed defines the sampling procedure; identical samples across different library versions or platforms must be checked rather than assumed. See [reproduction instructions](docs/reproduction.md) for protocols and execution scope.

## Contents

| Location | Contents |
|---|---|
| `foldq/` | Code construction, link selection, circuit routing, memory simulation, tests, and archived experiment records |
| `HAL/hal/` | The HAL source subset used by FoldQ, including local routing fixes and code definitions |
| `HAL/foldq/experiments/` | FoldQ's HAL adapter and independent geometric acceptance checker |
| `data/derived/` | Eight source-linked numerical tables and their manifest |
| `data/audits/` | Numerical comparisons and geometric audit inputs |
| `tools/artifact/` | Commands that rebuild outputs in separate directories |
| `provenance/` | File hashes, export transformations, environment versions, and release validation |

[Data-to-claim map](docs/data-map.md) identifies the records behind each reported result. [Provenance](docs/provenance.md) distinguishes original measurements from later deterministic routing replay and lists the remaining historical metadata gaps. The numbered experiment filenames are retained so source hashes and archived records remain traceable.

## License and dependencies

FoldQ code, documentation, and generated data are released under the [MIT License](LICENSE). The included HAL code and its data retain the original [HAL MIT License](HAL/LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and modifications.
