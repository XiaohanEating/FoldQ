# Provenance and historical coverage

`provenance/manifest.json` records SHA-256 for every distributed file except itself. `provenance/export_map.json` records the original path and original and public hashes of files copied from the research workspace. Raw trial records are copied byte for byte. The table exporter and hardware-cost summarizer use the public data locations; their calculations are unchanged. Two workstation executable-path fields are replaced by a neutral environment description, and data manifests are updated for the public locations and bytes. The export map identifies these transformations.

The HAL checkout arrived without Git history. Its exact original upstream commit is unavailable. The distributed subset is therefore identified by file hashes and source snapshots, not by an inferred upstream commit. Local fixes in `HAL/hal/routing/straightline.py`, `astar.py`, and `routing_algo.py` address routing conflicts. FoldQ's adapter supplies a separate orientation-invariant three-dimensional segment-intersection check, including endpoint contact, penetration, collinear overlap, and layer transitions. Conflicting edges are removed in pairs. The geometry regression tests include independent rational-arithmetic comparisons.

## Original measurements and later replay

The LER samples in `exp33/tasks.jsonl` were produced using the final geometric acceptance protocol. Those original records contain accepted edges and conflicting segments, but not every successful path. The `hardware_cost/paths/` files were obtained later by deterministic routing replay. Every accepted edge set and conflict decision matches the corresponding original record. Hardware costs are paired with those existing LER samples after that match; replay does not create new LER samples.

The routing and metric source snapshots are retained as `hardware_cost/source_snapshot.zip` and `hardware_cost/metric_source_snapshot.zip`. `replay` writes a fresh snapshot, metadata, logs, and per-condition geometry under `outputs/replay/` and checks against the archived accepted designs.

Historical snapshots retain their original source paths and hashes. The historical metric summarizer refers to the research workspace's table location; the current summarizer reads the same CSV bytes from `data/derived/main_ler72.csv`.

## Historical gaps and superseded results

- The original random trials in the 105-code execution study did not record seeds or every edge set. The paper's random comparison uses the later `exp15b` archive: 1,050 explicit edge sets and their single-round recompilation. Its original three-round averages are not the matched paper baseline.
- Several older experiments have incomplete runtime metadata. Among released inputs, this affects the selected-graph LER study and the original timed calibration records. Current code and available records are included; a source tree saved today does not reconstruct an unavailable historical snapshot.
- Recorded historical hashes for some calibration/fixed-noise runs differ from current sources. The missing original snapshots remain missing. Exact replay has been demonstrated only for the conditions explicitly recorded in validation; it is not asserted for every historical run.
- The original duration sweep used an incorrect idle-channel aggregation formula and an unmatched denominator for a no-idle comparison. Its records are retained because the corrected eight-code study reads their calibration values. Use `exp29b` for the reported duration and idle-noise conclusions; the older full sweep is superseded.
- Earlier geometric acceptance variants missed intersections. Their result directories are excluded from this release. The archived 96-condition scan, 48 accepted-graph LER conditions, and geometry tests use the repaired three-dimensional check.

The release contains archived evidence, deterministic reconstruction commands, and current experiment implementations. These three forms of evidence are identified separately in [the reproduction instructions](reproduction.md).
