"""Analytic boundary cases for accepted-graph HAL cost measurements."""
from copy import deepcopy
import math

import pytest

from foldq.core.hardware_cost import measure_hardware, normalized_score, trace_quantities


def _geometry(tier=1):
    xy = [(0, 0), (2, 0), (0, 2), (2, 2)]
    sites = [{"id": i, "hal_xy": list(p)} for i, p in enumerate(xy)]
    local = []
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        p, q = xy[a], xy[b]
        local.append({"edge": [a, b], "route": [[p[0], p[1], 0],
            [(p[0]+q[0]) // 2, (p[1]+q[1]) // 2, 0], [q[0], q[1], 0]]})
    raw = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 2, 1], [2, 2, 0]]
    return {"grid_side": 2, "sites": sites, "local_edges": local, "router_tiers": 9,
        "added_paths": [{"edge": [0, 3], "tier": tier, "accepted": True,
            "subpaths": [raw[:2], raw[1:4], raw[3:]], "raw_route": raw,
            "hal_subedge_count": 3, "missing": []}]}


def test_hal_reference_points_and_no_optimistic_cap():
    assert normalized_score((1, 1, 0, 0))["complexity"] == 1
    assert normalized_score((5, 10, 4, 3))["complexity"] == 2
    assert normalized_score((9, 19, 8, 6))["complexity"] == 3


def test_known_path_separates_two_bumps_from_three_subedges():
    result = measure_hardware(_geometry(), [[0, 3]])
    ratio = (1 + math.sqrt(5)) / 2
    expected = 1 + ((2 - 1) / 4 + (ratio - 1) / 9 + 2 / 4 + 2 / 3) / 4
    assert result["primary"]["complexity"] == pytest.approx(expected)
    assert result["primary"]["raw"] == {"tiers": 2, "length_ratio": ratio,
        "max_mean_bumps": 2, "mean_tsvs": 2}
    # Seven stored points = five raw points + two repeated split endpoints.
    legacy_length = (1 + 7 / 3) / 2
    assert result["hal_software_compatibility"]["raw"]["length_ratio"] == pytest.approx(legacy_length)
    assert result["hal_software_compatibility"]["raw"]["max_mean_bumps"] == 3
    assert result["inventory"]["required_tiers"] == 2
    assert result["inventory"]["router_attempted_tiers"] == 9


def test_empty_internal_tiers_preserve_vertical_distance():
    result = measure_hardware(_geometry(tier=3), [[0, 3]])
    assert result["primary"]["raw"]["tiers"] == 4
    assert result["primary"]["raw"]["mean_tsvs"] == 6
    assert [r["edges"] for r in result["tier_resources"]] == [4, 0, 0, 1]


def test_rejected_path_costs_nothing_and_empty_graph_is_planar():
    g = _geometry(tier=8)
    g["added_paths"][0]["accepted"] = False
    result = measure_hardware(g, [])
    assert result["primary"]["complexity"] == 1
    assert result["inventory"]["all_edges"] == 4
    assert result["inventory"]["total_bumps"] == result["inventory"]["total_tsvs"] == 0


def test_subpath_order_and_direction_do_not_affect_cost():
    g = _geometry()
    baseline = measure_hardware(g, [[0, 3]])
    path = g["added_paths"][0]
    path["subpaths"] = [p[::-1] for p in path["subpaths"][::-1]]
    path["raw_route"] = path["raw_route"][::-1]
    path["edge"].reverse()
    assert measure_hardware(g, [[3, 0]]) == baseline


def test_disconnected_subpaths_are_never_joined_into_fake_segments():
    length, bumps = trace_quantities([[[0, 0, 0], [1, 0, 0]], [[20, 0, 1], [21, 0, 1]]])
    assert length == 2 and bumps == 0
    g = _geometry()
    g["added_paths"][0]["subpaths"][1].pop()
    with pytest.raises(ValueError, match="differ from captured raw geometry"):
        measure_hardware(g, [[0, 3]])


def test_score_uses_local_geometry_unit_and_scales_consistently():
    g = _geometry()
    baseline = measure_hardware(g, [[0, 3]])["primary"]
    scaled = deepcopy(g)
    for site in scaled["sites"]:
        site["hal_xy"] = [v * 7 for v in site["hal_xy"]]
    for item in scaled["local_edges"]:
        for p in item["route"]:
            p[0] *= 7
            p[1] *= 7
    for item in scaled["added_paths"]:
        # Deepcopy preserves shared endpoint lists in this fixture. Replace each
        # whole representation to scale each point exactly once.
        item["raw_route"] = [[p[0]*7, p[1]*7, p[2]] for p in item["raw_route"]]
        item["subpaths"] = [[[p[0]*7, p[1]*7, p[2]] for p in part] for part in item["subpaths"]]
    result = measure_hardware(scaled, [[0, 3]])["primary"]
    assert result["complexity"] == pytest.approx(baseline["complexity"])


def test_same_edge_count_does_not_replace_exact_topology_check():
    with pytest.raises(ValueError, match="Cost paths do not match"):
        measure_hardware(_geometry(), [[1, 2]])


def test_missing_local_edge_is_rejected():
    g = _geometry()
    g["local_edges"].pop()
    with pytest.raises(ValueError, match="complete square grid"):
        measure_hardware(g, [[0, 3]])
