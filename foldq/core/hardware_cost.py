"""HAL's four-factor normalization on accepted FoldQ interconnect geometry.

The primary score follows the paper's quantities with explicit XY trace length
and actual face transitions. A separate compatibility calculation reproduces
the reference software's point-count and split-edge proxies on the same hardware.
Neither score estimates fabrication prices or electromagnetic behavior.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math


BASELINE = (1.0, 1.0, 0.0, 0.0)
OPTIMISTIC = (5.0, 10.0, 4.0, 3.0)
FACTOR_NAMES = ("tiers", "length_ratio", "max_mean_bumps", "mean_tsvs")


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else 0.0


def normalized_score(raw) -> dict:
    """Equal-weight HAL normalization; no upper clipping at optimistic values."""
    if len(raw) != 4 or any(not math.isfinite(x) for x in raw):
        raise ValueError("Expected four finite hardware quantities")
    if any(x < b for x, b in zip(raw, BASELINE)):
        raise ValueError("Primary hardware quantities cannot lie below their baseline")
    scaled = [(x - b) / (p - b) for x, b, p in zip(raw, BASELINE, OPTIMISTIC)]
    return {"raw": dict(zip(FACTOR_NAMES, raw)),
            "normalized": dict(zip(FACTOR_NAMES, scaled)),
            "complexity": 1.0 + mean(scaled)}


def _compatibility_score(raw) -> dict:
    # The reference implementation clips below zero; empty internal tiers may
    # dilute the per-tier length mean below one, so preserve that behavior here.
    scaled = [max(0.0, (x - b) / (p - b)) for x, b, p in zip(raw, BASELINE, OPTIMISTIC)]
    return {"raw": dict(zip(FACTOR_NAMES, raw)),
            "normalized": dict(zip(FACTOR_NAMES, scaled)),
            "complexity": 1.0 + mean(scaled)}


def _edge(edge):
    if len(edge) != 2 or edge[0] == edge[1]:
        raise ValueError("Expected a non-self edge")
    return tuple(sorted(edge))


def _segments(subpaths):
    return [(tuple(a), tuple(b)) for path in subpaths for a, b in zip(path, path[1:])]


def _segment_multiset(subpaths):
    return Counter(tuple(sorted((a, b))) for a, b in _segments(subpaths) if a != b)


def trace_quantities(subpaths) -> tuple[float, int]:
    """Sum only within recorded subpaths; z is a face label, not a distance."""
    segments = _segments(subpaths)
    if any(len(p) != 3 or p[2] not in (0, 1) for path in subpaths for p in path):
        raise ValueError("Route vertices must carry XY coordinates and a binary face")
    length = math.fsum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segments)
    bumps = sum(a[2] != b[2] for a, b in segments)
    return length, bumps


def measure_hardware(geometry: dict, accepted_edges) -> dict:
    """Measure E0 plus accepted edges, retaining physical tier indices."""
    size = geometry["grid_side"]
    if not isinstance(size, int) or size < 2:
        raise ValueError("The physical grid needs an integer side >= 2")
    sites = {s["id"]: tuple(s["hal_xy"]) for s in geometry["sites"]}
    if len(geometry["sites"]) != size * size or set(sites) != set(range(size * size)):
        raise ValueError("Physical site inventory is incomplete or duplicated")
    if len(set(sites.values())) != len(sites):
        raise ValueError("Distinct physical sites share a HAL coordinate")
    expected_local = {_edge((u, v)) for u in range(size * size)
                      for v in ([u + 1] if u % size + 1 < size else [])
                      + ([u + size] if u + size < size * size else [])}
    local = geometry["local_edges"]
    if len(local) != len(expected_local) or {_edge(e["edge"]) for e in local} != expected_local:
        raise ValueError("Local edge inventory must be the complete square grid")
    local_lengths, local_point_counts = [], []
    for item in local:
        edge, path = _edge(item["edge"]), item["route"]
        if len(path) < 2 or {tuple(path[0][:2]), tuple(path[-1][:2])} != {sites[e] for e in edge}:
            raise ValueError("Local trace endpoints disagree with site mapping")
        length, bumps = trace_quantities([path])
        if bumps or any(p[2] != 0 for p in path) or length <= 0:
            raise ValueError("Local trace must have positive length on face zero")
        local_lengths.append(length)
        local_point_counts.append(len(path))
    length_unit, count_unit = min(local_lengths), min(local_point_counts)
    accepted = {_edge(e) for e in accepted_edges}
    if len(accepted) != len(accepted_edges) or accepted & expected_local:
        raise ValueError("Accepted added links are duplicated or already local")
    paths = [p for p in geometry["added_paths"] if p["accepted"]]
    if len(paths) != len(accepted) or {_edge(p["edge"]) for p in paths} != accepted:
        raise ValueError("Cost paths do not match the accepted graph")
    resources = []
    for item in paths:
        edge, tier = _edge(item["edge"]), item["tier"]
        if not isinstance(tier, int) or tier < 0 or item["missing"]:
            raise ValueError("Invalid accepted tier or missing path")
        subpaths, raw = item["subpaths"], item["raw_route"]
        if not subpaths or any(not p for p in subpaths) or raw is None or len(raw) < 2:
            raise ValueError("Incomplete accepted route geometry")
        if {tuple(raw[0][:2]), tuple(raw[-1][:2])} != {sites[e] for e in edge}:
            raise ValueError("Added trace endpoints disagree with site mapping")
        if raw[0][2] != 0 or raw[-1][2] != 0:
            raise ValueError("Added link must terminate on the qubit-facing surface")
        if _segment_multiset(subpaths) != _segment_multiset([raw]):
            raise ValueError("Stored independent subpaths differ from captured raw geometry")
        length, bumps = trace_quantities(subpaths)
        ratio = length / length_unit
        if ratio < 1:
            raise ValueError("Added trace is shorter than the local reference")
        resources.append({"edge": list(edge), "tier": tier,
            "projected_trace_length": length, "length_ratio": ratio,
            "bumps": bumps, "tsvs": 2 * tier,
            "hal_point_count": sum(len(p) for p in subpaths),
            "hal_split_edge_proxy": item["hal_subedge_count"]})
    resources.sort(key=lambda r: r["edge"])
    required_tiers = 1 + max((p["tier"] for p in resources), default=0)
    by_tier = defaultdict(list)
    for p in resources:
        by_tier[p["tier"]].append(p)
    high = [p for p in resources if p["tier"] > 0]
    tier_records = []
    for t in range(required_tiers):
        rr = by_tier[t]
        count = len(rr) + (len(local) if t == 0 else 0)
        bumps = sum(p["bumps"] for p in rr)
        proxy_bumps = sum(p["hal_split_edge_proxy"] for p in rr)
        point_sum = sum(p["hal_point_count"] for p in rr) + (sum(local_point_counts) if t == 0 else 0)
        tier_records.append({"tier": t, "edges": count,
            "mean_bumps": bumps / count if count else 0.0,
            "hal_mean_split_edge_proxy": proxy_bumps / count if count else 0.0,
            "hal_mean_point_count": point_sum / count if count else 0.0})
    primary = normalized_score((float(required_tiers), mean(p["length_ratio"] for p in high) if high else 1.0,
        max(t["mean_bumps"] for t in tier_records), mean(p["tsvs"] for p in high)))
    compatibility = _compatibility_score((float(required_tiers),
        mean(t["hal_mean_point_count"] / count_unit for t in tier_records),
        max(t["hal_mean_split_edge_proxy"] for t in tier_records),
        mean(p["tsvs"] for p in high)))
    planar_compatibility = _compatibility_score((1.0, mean(local_point_counts) / count_unit, 0.0, 0.0))
    return {"primary": primary, "planar_primary": normalized_score(BASELINE),
        "hal_software_compatibility": compatibility,
        "planar_hal_software_compatibility": planar_compatibility,
        "hal_software_compatibility_delta": compatibility["complexity"] - planar_compatibility["complexity"],
        "inventory": {"sites": size * size, "local_edges": len(local),
            "accepted_added_edges": len(resources), "all_edges": len(local) + len(resources),
            "required_tiers": required_tiers, "router_attempted_tiers": geometry["router_tiers"],
            "local_length_min": length_unit, "local_length_mean": mean(local_lengths),
            "local_length_max": max(local_lengths),
            "local_trace_length": math.fsum(local_lengths),
            "added_trace_length": math.fsum(p["projected_trace_length"] for p in resources),
            "total_bumps": sum(p["bumps"] for p in resources),
            "total_tsvs": sum(p["tsvs"] for p in resources),
            "max_tsvs_per_added_edge": max((p["tsvs"] for p in resources), default=0),
            "max_bumps_per_added_edge": max((p["bumps"] for p in resources), default=0)},
        "tier_resources": tier_records, "edge_resources": resources}
