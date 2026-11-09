"""Rebuild FoldQ evidence without overwriting archived measurements."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "HAL"), str(ROOT / "HAL/foldq/experiments")]


def output_directory(value: str, command: str) -> Path:
    path = Path(value).resolve() / command
    # Outputs may be placed elsewhere, but must never replace archived inputs.
    for directory in ("foldq", "HAL", "data", "provenance", "docs", "tools"):
        if path == ROOT / directory or ROOT / directory in path.parents:
            raise ValueError("Choose an output directory outside the archived input trees")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "tables", "costs", "replay", "scan", "random-check", "sample"])
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--limit", type=int, help="Limit replay conditions or random-baseline codes for a smoke run")
    parser.add_argument("--code", default="D4_3", help="Code for the 3,000-shot accepted-graph sample")
    parser.add_argument("--budget", type=float, choices=[.02, .05], default=.05)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    out = output_directory(args.output, args.command)
    if args.command == "verify":
        from .verify import verify
        result = verify()
    elif args.command == "tables":
        from . import export_tables as module
        module.OUT = out
        module.main()
        from .verify import sha
        names = sorted(p.name for p in out.glob("*.csv"))
        if len(names) != 8 or any(sha(out / name) != sha(ROOT / "data/derived" / name) for name in names):
            raise ValueError("Rebuilt tables differ from archived tables")
        result = {"status":"passed", "tables_identical":names}
    elif args.command == "costs":
        from foldq.experiments.hardware_cost.summarize import summarize
        for name in ("meta.json", "replay_summary.json"):
            shutil.copy2(ROOT / "foldq/experiments/results/hardware_cost" / name, out / name)
        shutil.copytree(ROOT / "foldq/experiments/results/hardware_cost/paths", out / "paths", dirs_exist_ok=True)
        with (out / "summary.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            summarize(out)
        from .verify import sha
        if sha(out / "workpoints.csv") != sha(ROOT / "foldq/experiments/results/hardware_cost/workpoints.csv"):
            raise ValueError("Rebuilt hardware table differs from archived table")
        result = {"status":"passed", "hardware_table_identical":True}
    elif args.command == "replay":
        from foldq.experiments.hardware_cost.replay import run
        run(out, args.limit)
        from .verify import read
        result = read(out / "replay_summary.json")
    elif args.command == "scan":
        from .verify import read
        from foldq.experiments.exp33_workpoint_ler import _patch_hal, _routed_edges
        from foldq.routing import CouplingGraph, route_round
        _patch_hal()
        expected = read(ROOT / "data/audits/scan_replay.json")
        rows = []
        for row in expected[:args.limit]:
            with (out / "routing.log").open("a", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                code, inter, size, phys, g0, routed, selected, conflicts = _routed_edges(row["code"], row["k_pct"])
            edge_set = lambda edges:{tuple(sorted(e)) for e in edges}
            if edge_set(routed) != edge_set(row["accepted_edges"]) or selected != row["selected"]:
                raise ValueError(f"Budget-scan topology mismatch: {row['code']} at {row['k_pct']}")
            swap_counts = []
            for graph in (g0, CouplingGraph(size, routed)):
                mapping, swaps = dict(phys), 0
                for _ in range(3):
                    result = route_round(inter, mapping, graph, return_ops=True)
                    swaps += result["swaps"]
                    mapping = result["final_phys"]
                swap_counts.append(swaps / 3)
            if swap_counts != [row["baseline_swaps"], row["accepted_swaps"]]:
                raise ValueError(f"Budget-scan SWAP mismatch: {row['code']} at {row['k_pct']}")
            rows.append({"code":row["code"], "k_pct":row["k_pct"], "selected":selected, "accepted":len(routed), "baseline_swaps":swap_counts[0], "accepted_swaps":swap_counts[1]})
            print(f"{row['code']} at {row['k_pct']}: accepted edges and SWAPs match", flush=True)
        result = {"status":"passed", "conditions":len(rows), "rows":rows}
    elif args.command == "random-check":
        from .verify import records, read
        from foldq.experiments.exp15b_random_seeded import _setup
        from foldq.routing import CouplingGraph, route_round
        expected = {r["code"]:r for r in read(ROOT / "foldq/experiments/results/exp15b/single_round_check.json")}
        rows = []
        for row in sorted(records("exp15b"), key=lambda x:(x["n"],x["code"]))[:args.limit]:
            setup = _setup(row["code"])
            swaps = [route_round(setup["inter"], setup["phys"], CouplingGraph(setup["size"], t["cross"]))["swaps"] for t in row["trials"]]
            mean = sum(swaps)/len(swaps)
            if abs(mean - expected[row["code"]]["single_round_mean"]) > 1e-12:
                raise ValueError(f"Single-round random baseline mismatch: {row['code']}")
            rows.append({"code":row["code"], "single_round_swaps":swaps, "mean":mean})
            print(f"{row['code']}: ten saved edge sets match", flush=True)
        result = {"status":"passed", "codes":len(rows), "rows":rows}
    else:
        from foldq.experiments.exp33_workpoint_ler import task_ler
        from .verify import records
        expected = next((r for r in records("exp33") if r["code"] == args.code and r["k_pct"] == args.budget), None)
        if expected is None:
            parser.error("Choose a code and budget present in the 48 accepted-graph records")
        observed = task_ler((args.code, args.budget))
        keys = ("failures", "shots", "ler", "ci95", "routed_edges", "n_cross", "n_routed")
        equal = {key:observed[key] == expected[key] for key in keys}
        result = {"status":"passed" if all(equal.values()) else "mismatch", "record":observed, "matches_archive":equal}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
    if result.get("status") == "mismatch":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
