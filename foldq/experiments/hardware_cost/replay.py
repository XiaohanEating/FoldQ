"""Replay existing layouts and archive geometry without changing routing decisions.

Run from the project root with the hal environment:
    python -m foldq.experiments.hardware_cost.replay

Each completed condition is written atomically. A source-manifest mismatch stops
resumption; existing LER samples and the historical routing sources are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "foldq/experiments/results/hardware_cost"
TASKS = ROOT / "foldq/experiments/results/exp33/tasks.jsonl"
sys.path[:0] = [str(ROOT), str(ROOT / "HAL"), str(ROOT / "HAL/foldq/experiments")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def edge_key(edge) -> tuple[int, int]:
    return tuple(sorted(map(int, edge[:2])))


def integer_route(route) -> list[list[int]]:
    points = [[int(v) for v in p] for p in route]
    if any(tuple(p) != tuple(q) for p, q in zip(points, route)):
        raise ValueError("Nonintegral HAL geometry; do not silently quantize it")
    return points


@contextmanager
def observe_hal():
    """Observe objects and successful raw paths; return every original result."""
    from hal.hal import HardwareAwareLayout
    from hal.routing.routing_algo import TierBasedRouter

    instances, full_routes = [], {}
    old_init = HardwareAwareLayout.__init__
    old_update = TierBasedRouter._update_tier_state_for_successful_route

    def capture_init(self, *args, **kwargs):
        result = old_init(self, *args, **kwargs)
        instances.append(self)
        return result

    def capture_update(self, tier_ind, edge, route, switch_info, tier_state):
        # Capture before the implementation reverses its subedge-order list.
        full_routes[(int(tier_ind), edge_key(edge))] = integer_route(deepcopy(route))
        return old_update(self, tier_ind, edge, route, switch_info, tier_state)

    HardwareAwareLayout.__init__ = capture_init
    TierBasedRouter._update_tier_state_for_successful_route = capture_update
    try:
        yield instances, full_routes
    finally:
        HardwareAwareLayout.__init__ = old_init
        TierBasedRouter._update_tier_state_for_successful_route = old_update


def export_geometry(hal, full_routes, accepted, size: int) -> dict:
    accepted = {edge_key(e) for e in accepted}
    paths = []
    for ti, tier in enumerate(hal.tiers):
        for original, subedges in (tier.switch_edges or {}).items():
            key = edge_key(original)
            parts, missing = [], []
            for edge in subedges or [original]:
                route = tier.G.edges[edge].get("route") if edge in tier.G.edges else None
                if route:
                    parts.append(integer_route(route))
                else:
                    missing.append(repr(edge))
            raw = full_routes.get((ti, key))
            if raw is None and not subedges and len(parts) == 1:
                raw = parts[0]
            if key in accepted and (missing or not parts or raw is None):
                raise ValueError(f"Accepted edge {key} lacks complete captured geometry")
            paths.append({"edge": list(key), "tier": ti, "accepted": key in accepted,
                          "subpaths": parts, "raw_route": raw,
                          "hal_subedge_count": len(subedges), "missing": missing})
    exported = [edge_key(p["edge"]) for p in paths if p["accepted"]]
    if len(exported) != len(set(exported)) or set(exported) != accepted:
        raise ValueError("Accepted paths do not have a unique one-to-one edge correspondence")
    sites = []
    for u in range(size * size):
        xy = hal.tiers[0].G.nodes[u]["pos"]
        sites.append({"id": u, "grid_xy": [u % size, u // size],
                      "hal_xy": [int(xy[0]), int(xy[1])]})
    local = []
    for u in range(size * size):
        for v in ([u + 1] if u % size + 1 < size else []) + ([u + size] if u + size < size * size else []):
            a, b = sites[u]["hal_xy"], sites[v]["hal_xy"]
            dx, dy = b[0] - a[0], b[1] - a[1]
            if bool(dx) == bool(dy):
                raise ValueError("Local grid edge is zero or not axis aligned after rasterization")
            steps = abs(dx) + abs(dy)
            # Nearest-neighbor traces are existing hardware, not HAL routing requests.
            route = [[a[0] + i * dx // steps, a[1] + i * dy // steps, 0]
                     for i in range(steps + 1)]
            local.append({"edge": [u, v], "route": route})
    assert len(local) == 2 * size * (size - 1)
    return {"grid_side": size, "sites": sites, "local_edges": local,
            "router_tiers": len(hal.tiers), "added_paths": paths}


def canonical_conflicts(conflicts):
    return sorted(
        (int(c["tier"]), tuple(sorted([edge_key(c["e1"]), edge_key(c["e2"])])))
        for c in conflicts
    )


def source_manifest() -> dict:
    from foldq.experiments.exp33_workpoint_ler import SOURCES
    files = [ROOT / x for x in SOURCES]
    files += [Path(__file__), ROOT / "HAL/hal/tier.py", ROOT / "HAL/hal/settings.py",
              ROOT / "HAL/hal/utils/grid_utils.py", TASKS]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}


def run(out: Path, limit: int | None = None) -> None:
    from exp27_hal_routing import _patch_hal, MemGuard
    from foldq.experiments.exp33_workpoint_ler import _routed_edges

    out.mkdir(parents=True, exist_ok=True)
    lock = out / "replay.lock"
    # No two writers can own this run. A stale lock requires explicit inspection.
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(__import__("os").getpid()))
    guard = MemGuard()
    try:
        sources = source_manifest()
        meta_path = out / "meta.json"
        if meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if previous["sources_sha256"] != sources:
                raise ValueError("Replay sources changed; use a separate output directory")
        else:
            atomic_json(meta_path, {"created_at": datetime.now(timezone.utc).isoformat(),
                "sources_sha256": sources, "python": sys.executable,
                "python_version": platform.python_version(),
                "packages": {p: version(p) for p in ["numpy", "networkx", "qldpc", "stim", "qiskit", "ldpc"]},
                "layout_seed": 0, "budgets": [0.02, 0.05], "candidate_length": 6,
                "workers": 1, "sampling": "reuse exp33 only after exact accepted-edge match",
                "hal_parameters": {"max_len_argument": None, "max_tsvs": 10, "max_bumps": 10}})
            with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
                for name in sources:
                    z.write(ROOT / name, name)
        records = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(records) == 48 and len({(r["code"], r["k_pct"]) for r in records}) == 48
        _patch_hal()
        guard.start()
        done = 0
        started = time.perf_counter()
        for row in records:
            name, kp = row["code"], row["k_pct"]
            slug = f"{name}_k{round(100 * kp):02d}"
            target = out / "paths" / f"{slug}.json"
            if target.exists():
                saved = json.loads(target.read_text(encoding="utf-8"))
                if saved["exp33_record"] != row or not saved["accepted_edges_match"]:
                    raise ValueError(f"Existing record mismatch: {slug}")
                continue
            t0 = time.perf_counter()
            (out / "logs").mkdir(exist_ok=True)
            with (out / "logs" / f"{slug}.txt").open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log), observe_hal() as (instances, routes):
                    code, _, size, _, _, routed, n_cross, conflicts = _routed_edges(name, kp)
                    if len(instances) != 1:
                        raise ValueError("Expected one HAL object per working point")
                    geometry = export_geometry(instances[0], routes, routed, size)
            if {edge_key(e) for e in routed} != {edge_key(e) for e in row["routed_edges"]}:
                raise ValueError(f"Accepted topology changed for {slug}; LER cannot be reused")
            if canonical_conflicts(conflicts) != canonical_conflicts(row["collision_detail"]):
                raise ValueError(f"Conflict decisions changed for {slug}")
            if n_cross != row["n_cross"]:
                raise ValueError(f"Selected edge count changed for {slug}")
            elapsed = time.perf_counter() - t0
            atomic_json(target, {"code": name, "k_pct": kp, "n_initial": code.num_physical,
                "accepted_edges_match": True, "conflict_decisions_match": True,
                "exp33_record": row, "geometry": geometry,
                "seconds": elapsed, "process_peak_bytes": guard.peak})
            done += 1
            print(f"{slug}: {len(routed)}/{n_cross} accepted; exact match; {elapsed:.2f}s", flush=True)
            instances.clear()
            if limit is not None and done >= limit:
                break
        if source_manifest() != sources:
            raise ValueError("Sources changed during replay")
        completed = sorted((out / "paths").glob("*.json"))
        atomic_json(out / "replay_summary.json", {"conditions_archived": len(completed),
            "conditions_expected": 48, "new_conditions": done,
            "seconds_this_invocation": time.perf_counter() - started,
            "process_peak_bytes": guard.peak, "source_hashes_unchanged": True,
            "paths_sha256": {p.name: sha(p) for p in completed}})
        print(f"Archived {len(completed)}/48 conditions; peak RSS {guard.peak / (1 << 30):.3f} GiB", flush=True)
    finally:
        guard.stop()
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="Maximum newly replayed conditions in this invocation")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run(args.output.resolve(), args.limit)


if __name__ == "__main__":
    main()
