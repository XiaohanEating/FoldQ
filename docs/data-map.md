# Results and source data

| Result | Primary records | Derived table / check |
|---|---|---|
| 105-code SWAP and depth comparison | `foldq/experiments/results/exp15/results.json` | `data/derived/architecture_depth_105.csv` (210 rows: two compilers); `selection_105.csv` (105 rows) |
| Random-link baseline, same budget and single-round protocol | `exp15b/tasks.jsonl` (105 codes × ten saved edge sets), `exp15b/single_round_check.json` | `random-check` rebuilds all 1,050 single-round circuits |
| Selected-graph LER, 21/24 improvements | `exp28b/tasks.jsonl`; calibration in `exp25/tasks.jsonl` | `main_ler72.csv`, selected column; Holm across 24 comparisons |
| Accepted-graph LER, 4/24 and 6/24 improvements | `exp33/tasks.jsonl`, matched planar records in `exp28b/tasks.jsonl` | `physical_ler_all48.csv`; Holm across both budgets together |
| Fixed-noise LER, 46/72 improvements | `exp26/fixed_p/tasks.jsonl` | `verify` recomputes all 72 tests as one family |
| SABRE memory circuits, 3/4 improvements | `exp32/tasks.jsonl` | `verify` recomputes all four tests as one family |
| Budget scan, 24 codes × four budgets | `data/audits/scan_replay.json` | `budget_per_code.csv` (96 rows), `budget_aggregate.csv` (four rows) |
| Duration/idle-noise check, eight codes × five conditions | `exp29b/tasks.jsonl`; timed calibration in `exp29/tasks.jsonl` | Three comparisons per code, Holm across 24 tests |
| Hardware complexity at the same 48 accepted designs | `hardware_cost/paths/*.json` and `hardware_cost/measurements.json` | `hardware_cost/workpoints.csv`; `verify` and `costs` recompute from paths |
| Four representative working points | Accepted-graph records and paired hardware measurements | `main_workpoints4.csv` |

Experiment directories in the table are relative to `foldq/experiments/results/` unless a full repository-relative path is given. Derived CSV paths are relative to `data/derived/`. `main_ler72.csv` combines 24 selected-graph and 48 accepted-graph comparisons. It retains negative and nonsignificant outcomes. `workpoint_cases.csv` contains nine cases used during analysis; `main_workpoints4.csv` contains four representative cases.

The `data/audits/` JSON files hold historical numerical audit inputs used by the table exporter. Their original locations and hashes are recorded in `provenance/export_map.json`. Current release validation is in `provenance/validation.json`.

Stored sample records contain failure counts, shots, exact 95% intervals, conditions, and available source metadata. Detector samples for every shot were not retained. A fresh `sample` run regenerates a circuit and samples it using the recorded protocol.

## Metric conventions

`N` counts initially allocated data and syndrome-ancilla qubits; `M` counts all grid sites. The nearest-neighbor grid includes unused sites. Link budgets use `max(1, round(k*N))` with Python's nearest-even rounding. Candidate endpoint Manhattan distance and physical routed trace length are different quantities.

Greedy weighted depth assigns durations of one to CX and three to SWAP. The historical SABRE depth is DAG operation depth with unexpanded SWAPs. Comparisons use each compiler's own planar baseline.

The accepted-graph hardware score uses HAL's four-component normalization. Its path-length and face-transition measurements come from the accepted geometry. The separate `hal_software_compatibility` fields reproduce the HAL software's vertex-count and subedge-count proxies. Resource counts and the normalized score measure interconnect complexity; fabrication costs are outside this dataset.
