"""Independently check release integrity, statistical families, and path costs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "foldq/experiments/results"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(name: str):
    return [json.loads(line) for line in (RESULTS / name / "tasks.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def two_proportion(a, b):
    n1, n2, k1, k2 = a["shots"], b["shots"], a["failures"], b["failures"]
    pooled = (k1 + k2) / (n1 + n2)
    variance = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    return float(2 * norm.sf(abs(k1 / n1 - k2 / n2) / math.sqrt(variance))) if variance > 0 else 1.0


def compare(pairs):
    pvalues = [two_proportion(a, b) for _, a, b in pairs]
    adjusted = [0.0] * len(pairs)
    running = 0.0
    for rank, i in enumerate(sorted(range(len(pairs)), key=pvalues.__getitem__)):
        running = max(running, min(1.0, (len(pairs) - rank) * pvalues[i]))
        adjusted[i] = running
    rows = []
    for (key, a, b), p in zip(pairs, adjusted):
        baseline, treatment = a["failures"] / a["shots"], b["failures"] / b["shots"]
        gain = 1 - treatment / baseline if baseline else None
        rows.append({"key": key, "gain": gain, "p_holm": p, "significant": p < .05 and treatment < baseline})
    return rows


def verify() -> dict:
    manifest = read(ROOT / "provenance/manifest.json")
    for name, digest in manifest["files"].items():
        require(sha(ROOT / name) == digest, f"Release hash mismatch: {name}")
    checked_ci = 0
    raw_names = ["exp25", "exp26/fixed_p", "exp28b", "exp29b", "exp32", "exp33"]
    for name in raw_names:
        for row in records(name):
            if not {"failures", "shots", "ler", "ci95"} <= row.keys():
                continue
            k, n = row["failures"], row["shots"]
            require(0 <= k <= n, f"Invalid counts: {name}")
            ci = [beta.ppf(.025, k, n-k+1) if k else 0., beta.ppf(.975, k+1, n-k) if k < n else 1.]
            require(math.isclose(row["ler"], k/n, abs_tol=1e-14), f"LER mismatch: {name}")
            require(np.allclose(row["ci95"], ci, rtol=0, atol=1e-12), f"CI mismatch: {name}")
            checked_ci += 1
    selected_data = records("exp28b")
    planar = {r["code"]: r for r in selected_data if r["cond"] == "base"}
    selected = compare([(r["code"], planar[r["code"]], r) for r in selected_data if r["cond"] == "foldq_l6"])
    physical = records("exp33")
    accepted = compare([((r["code"], r["k_pct"]), planar[r["code"]], r) for r in physical])
    fixed_records = records("exp26/fixed_p")
    fixed_base = {(r["code"], r["p"]): r for r in fixed_records if r["cond"].endswith("_base")}
    fixed = compare([((r["code"], r["p"]), fixed_base[(r["code"], r["p"])], r) for r in fixed_records if r["cond"].endswith("_foldq")])
    sabre_records = records("exp32")
    sabre_base = {r["code"]: r for r in sabre_records if r["cond"] == "sabre_base"}
    sabre = compare([(r["code"], sabre_base[r["code"]], r) for r in sabre_records if r["cond"] == "sabre_foldq"])
    timed_records = records("exp29b")
    timed_lookup = {(r["code"], r["cond"]): r for r in timed_records}
    timed_pairs = []
    for r in timed_records:
        if r["cond"] in ("base", "idle0_base"):
            continue
        base = "idle0_base" if r["cond"].startswith("idle0") else "base"
        timed_pairs.append(((r["code"], r["cond"]), timed_lookup[(r["code"], base)], r))
    timed = compare(timed_pairs)
    require((len(selected), len(accepted), len(fixed), len(sabre), len(timed)) == (24,48,72,4,24), "Incorrect Holm family sizes")
    significant = {"selected_24": sum(r["significant"] for r in selected), "accepted_k02": sum(r["significant"] for r in accepted if r["key"][1] == .02), "accepted_k05": sum(r["significant"] for r in accepted if r["key"][1] == .05), "fixed_noise_72": sum(r["significant"] for r in fixed), "sabre_4": sum(r["significant"] for r in sabre)}
    require(list(significant.values()) == [21,4,6,46,3], f"Significance counts: {significant}")
    timed_counts = {condition:sum(r["significant"] for r in timed if r["key"][1] == condition) for condition in ("a1.5_b1.5", "a3_b3")}
    require(list(timed_counts.values()) == [6,4], f"Timing significance counts: {timed_counts}")
    indexed = {(r["key"], "selected", .1): r for r in selected}
    indexed.update({(r["key"][0], "accepted", r["key"][1]): r for r in accepted})
    with (ROOT / "data/derived/main_ler72.csv").open(encoding="utf-8-sig") as f:
        table = list(csv.DictReader(f))
    require(len(table) == 72, "Incomplete LER table")
    for row in table:
        recomputed = indexed[(row["code"], row["graph"], float(row["k"]))]
        require(math.isclose(float(row["p_holm"]), recomputed["p_holm"], abs_tol=1e-12), "Exported Holm mismatch")
        require(math.isclose(float(row["ler_improvement"]), recomputed["gain"], abs_tol=1e-12), "Exported gain mismatch")
    from foldq.core.hardware_cost import measure_hardware
    paths = sorted((RESULTS / "hardware_cost/paths").glob("*.json"))
    require(len(paths) == 48, "Missing accepted geometry")
    measured = read(RESULTS / "hardware_cost/measurements.json")
    expected = {(r["code"], r["k_pct"]): r["hardware"] for r in measured}
    for path in paths:
        row = read(path)
        hardware = measure_hardware(row["geometry"], row["exp33_record"]["routed_edges"])
        require(hardware == expected[(row["code"], row["k_pct"])], f"Hardware accounting mismatch: {path.name}")
    random = records("exp15b")
    require(len(random) == 105 and sum(len(r["trials"]) for r in random) == 1050, "Missing random edge sets")
    scan = read(ROOT / "data/audits/scan_replay.json")
    require(len(scan) == 96, "Missing budget conditions")
    require(len(read(RESULTS / "exp15/results.json")["results"]) == 105, "Missing execution results")
    return {"status": "passed", "files_verified": len(manifest["files"]), "raw_ci_recomputed": checked_ci, "holm_families": [24,48,72,4,24], "significant_improvements": significant, "timing_significant_improvements":timed_counts, "hardware_costs_recomputed": len(paths), "budget_conditions":len(scan), "saved_random_edge_sets":1050, "new_monte_carlo_sampling":False}
