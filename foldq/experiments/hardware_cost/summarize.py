"""Pair accepted-graph hardware costs with the unchanged 48-comparison LER data."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from foldq.core.hardware_cost import measure_hardware, mean

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "foldq/experiments/results/hardware_cost"
LER_CSV = ROOT / "data/derived/main_ler72.csv"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(out: Path):
    with LER_CSV.open(encoding="utf-8-sig") as f:
        ler = {(r["code"], float(r["k"])): r for r in csv.DictReader(f) if r["graph"] == "accepted"}
    files = sorted((out / "paths").glob("*.json"))
    if len(files) != 48 or len(ler) != 48:
        raise ValueError("Need all 48 replay records and all 48 accepted-graph LER comparisons")
    records, rows = [], []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if not data["accepted_edges_match"] or not data["conflict_decisions_match"]:
            raise ValueError("Unmatched record cannot be paired with historical LER")
        sample = data["exp33_record"]
        reference = ler[(data["code"], data["k_pct"])]
        measured = measure_hardware(data["geometry"], sample["routed_edges"])
        if (abs(float(reference["ler"]) - sample["ler"]) > 1e-15
                or int(reference["failures"]) != sample["failures"]
                or int(reference["shots"]) != sample["shots"]):
            raise ValueError("LER source mismatch")
        item = {"code": data["code"], "k_pct": data["k_pct"], "n_initial": data["n_initial"],
                "selected_edges": sample["n_cross"], "geometry_file": file.relative_to(out).as_posix(),
                "geometry_sha256": sha(file), "hardware": measured, "ler": reference}
        records.append(item)
        inv, raw = measured["inventory"], measured["primary"]["raw"]
        row = {"code": item["code"], "k": item["k_pct"], "N": item["n_initial"],
            "M": inv["sites"], "local_edges": inv["local_edges"], "selected_edges": sample["n_cross"],
            "accepted_edges": inv["accepted_added_edges"], "tiers": raw["tiers"],
            "mean_length_ratio": raw["length_ratio"], "max_mean_bumps": raw["max_mean_bumps"],
            "mean_tsvs": raw["mean_tsvs"], "C_hw": measured["primary"]["complexity"],
            "C_hw_planar": 1.0, "total_bumps": inv["total_bumps"], "total_tsvs": inv["total_tsvs"],
            "max_tsvs_per_added_edge": inv["max_tsvs_per_added_edge"],
            "projected_added_length": inv["added_trace_length"],
            "local_length_min": inv["local_length_min"], "local_length_max": inv["local_length_max"],
            "hal_software_C": measured["hal_software_compatibility"]["complexity"],
            "hal_software_C0": measured["planar_hal_software_compatibility"]["complexity"],
            "ler_planar": reference["baseline_ler"], "ler_foldq": reference["ler"],
            "ler_improvement": reference["ler_improvement"], "p_holm": reference["p_holm"]}
        rows.append(row)
    if len({(r["code"], r["k"]) for r in rows}) != 48:
        raise ValueError("Duplicate working-point key")
    rows.sort(key=lambda r: (r["k"], r["N"], r["code"]))
    with (out / "workpoints.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "measurements.json", records)
    summary = {"conditions": len(rows), "accepted_edge_instances": sum(r["accepted_edges"] for r in rows),
        "empty_accepted_graphs": sum(r["accepted_edges"] == 0 for r in rows),
        "budgets": [], "illustrative_workpoints": []}
    for k in (0.02, 0.05):
        rr = [r for r in rows if r["k"] == k]
        significant = [r for r in rr if float(r["p_holm"]) < 0.05 and float(r["ler_improvement"]) > 0]
        summary["budgets"].append({"k": k, "conditions": len(rr),
            "C_hw_min": min(r["C_hw"] for r in rr), "C_hw_mean": mean(r["C_hw"] for r in rr),
            "C_hw_max": max(r["C_hw"] for r in rr),
            "significant_improvements": len(significant),
            "significant_C_hw_min": min(r["C_hw"] for r in significant),
            "significant_C_hw_max": max(r["C_hw"] for r in significant),
            "significant_points": significant})
    for code, k in [("GT_48_4_8", .05), ("GT_72_8_8", .05), ("GT_96_4_12", .02), ("GT_98_6_12", .05)]:
        summary["illustrative_workpoints"].append(next(r for r in rows if r["code"] == code and r["k"] == k))
    write_json(out / "summary.json", summary)
    sources = [Path(__file__), ROOT / "foldq/core/hardware_cost.py", LER_CSV,
               out / "meta.json", out / "replay_summary.json", *files]
    write_json(out / "metrics_manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "sources_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "outputs_sha256": {name: sha(out / name) for name in ("workpoints.csv", "measurements.json", "summary.json")}})
    with zipfile.ZipFile(out / "metric_source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in [Path(__file__), ROOT / "foldq/core/hardware_cost.py"]:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    summarize(parser.parse_args().output.resolve())
