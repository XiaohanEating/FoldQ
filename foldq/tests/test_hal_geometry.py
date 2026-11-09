"""Exact geometry regressions for HAL acceptance, independent of routing."""
from fractions import Fraction
import importlib.util
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest


_SOURCE = Path(__file__).resolve().parents[2] / 'HAL/foldq/experiments/exp27_hal_routing.py'
_SPEC = importlib.util.spec_from_file_location('foldq_hal_geometry_test', _SOURCE)
_ADAPTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ADAPTER)


def _rational_reference(a, b, c, d):
    """Solve a+s(b-a)=c+t(d-c) by rational elimination."""
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(d[i] - c[i] for i in range(3))
    w = tuple(c[i] - a[i] for i in range(3))
    for i, j in combinations(range(3), 2):
        det = v[i] * u[j] - u[i] * v[j]
        if det:
            s = Fraction(v[i] * w[j] - w[i] * v[j], det)
            t = Fraction(u[i] * w[j] - w[i] * u[j], det)
            return (0 <= s <= 1 and 0 <= t <= 1
                    and all(a[k] + s * u[k] == c[k] + t * v[k] for k in range(3)))
    if not any(u):
        if not any(v):
            return a == c
        k = next(k for k in range(3) if v[k])
        t = Fraction(a[k] - c[k], v[k])
        return 0 <= t <= 1 and all(a[j] == c[j] + t * v[j] for j in range(3))
    k = next(k for k in range(3) if u[k])
    if any(w[j] * u[k] != w[k] * u[j] for j in range(3)):
        return False
    return max(min(a[k], b[k]), min(c[k], d[k])) <= min(max(a[k], b[k]), max(c[k], d[k]))


@pytest.mark.parametrize('a,b,c,d,expected', [
    ((0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 0, 0), True),
    ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (-2, 0, 0), False),
    ((1, 0, 0), (1, 0, 0), (0, 0, 0), (2, 0, 0), True),
    ((1, 1, 0), (1, 1, 0), (0, 0, 0), (2, 0, 0), False),
    ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), True),
    ((0, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1), False),
    ((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0), True),
    ((0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), False),
    ((0, 0, 0), (0, 0, 1), (-1, 0, 0), (1, 0, 0), True),
    ((0, 0, 0), (3, 3, 3), (2, 2, 2), (4, 4, 4), True),
    ((0, 0, 0), (3, 3, 3), (4, 4, 4), (6, 6, 6), False),
])
def test_closed_segments_are_symmetric_and_direction_invariant(a, b, c, d, expected):
    for first, second in product(((a, b), (b, a)), ((c, d), (d, c))):
        assert _ADAPTER._seg3d_intersect(*first, *second) == expected
        assert _ADAPTER._seg3d_intersect(*second, *first) == expected


def test_all_ten_point_segments_match_independent_rational_reference():
    points = list(product((0, 1), repeat=3)) + [(2, 0, 0), (-1, 0, 0)]
    segments = list(product(points, repeat=2))
    for (a, b), (c, d) in product(segments, repeat=2):
        assert _ADAPTER._seg3d_intersect(a, b, c, d) == _rational_reference(a, b, c, d), (a, b, c, d)


def _tier(edge_routes):
    tier = SimpleNamespace(G=nx.Graph(), switch_edges={})
    for index, (original, routes) in enumerate(edge_routes):
        subedges = []
        for part, route in enumerate(routes):
            edge = ((index, part, 'a'), (index, part, 'b'))
            tier.G.add_edge(*edge, route=route)
            subedges.append(edge)
        tier.switch_edges[original] = subedges
    return tier


def test_path_acceptance_checks_touching_across_separate_edges():
    tier = _tier([((1, 2), [[(0, 0, 0), (1, 0, 0)]]),
                  ((3, 4), [[(2, 0, 0), (1, 0, 0)]])])
    conflicts = _ADAPTER._check_path_conflicts(SimpleNamespace(tiers=[tier]))
    assert len(conflicts) == 1
    assert {conflicts[0]['e1'], conflicts[0]['e2']} == {(1, 2), (3, 4)}


def test_subpaths_are_not_joined_by_invented_segments():
    tier = _tier([((1, 2), [[(0, 0, 0), (1, 0, 0)], [(3, 0, 0), (4, 0, 0)]]),
                  ((3, 4), [[(2, -1, 0), (2, 1, 0)]])])
    assert _ADAPTER._check_path_conflicts(SimpleNamespace(tiers=[tier])) == []


def test_same_coordinates_on_distinct_tiers_are_separate():
    route = [[(0, 0, 0), (1, 0, 0)]]
    tiers = [_tier([((1, 2), route)]), _tier([((3, 4), route)])]
    assert _ADAPTER._check_path_conflicts(SimpleNamespace(tiers=tiers)) == []
