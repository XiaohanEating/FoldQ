"""
路由与跨层选择的正确性测试

运行：python -m pytest foldq/tests -q
"""

import numpy as np

from foldq.core import SurfaceCode, GridTopology
from foldq.analysis import compute_routing_cost
from foldq.codes import load_hal_code
from foldq.optimization.layouts import native_layout, spring_layout, default_grid_size
from foldq.optimization.crosslayer_select import (
    candidate_pairs, select_marginal, select_longest)
from foldq.routing import CouplingGraph, positions_to_phys, static_swap_proxy, route_round


def test_static_proxy_matches_legacy_bfs():
    """新的矩阵实现与旧的 compute_routing_cost 口径一致"""
    code = SurfaceCode(5)
    size = 8
    rng = np.random.default_rng(0)
    phys_list = rng.permutation(size * size)[:code.num_physical]
    phys = {q: int(p) for q, p in enumerate(phys_list)}
    legacy = compute_routing_cost(code, GridTopology(size, size), phys)['estimated_swaps']
    new = static_swap_proxy(code.get_all_interactions(), phys, CouplingGraph(size))
    assert new == legacy


def test_incremental_distance_matches_full_recompute():
    """缓存平面图 + 逐边增量更新，与整图重算最短路逐项相同"""
    from foldq.routing.grid import all_pairs_distance, grid_edges
    size = 9
    rng = np.random.default_rng(1)
    cross = {tuple(sorted(map(int, rng.choice(size * size, 2, replace=False))))
             for _ in range(12)}
    g = CouplingGraph(size, cross)
    full = all_pairs_distance(size * size, grid_edges(size) + sorted(cross))
    assert np.array_equal(g.dist, full)
    assert np.array_equal(CouplingGraph(size).dist,
                          all_pairs_distance(size * size, grid_edges(size)))


def _route_round_reference(interactions, phys, graph):
    """提速前的 route_round 原样实现，用作回归对照"""
    D, adj, cross = graph.dist, graph.adj, graph.cross_set
    phys_of = dict(phys)
    logical_at = {p: q for q, p in phys_of.items()}
    busy = [0] * graph.n
    swaps = cross_swaps = cross_cx = 0

    def schedule(a, b, duration):
        start = max(busy[a], busy[b])
        busy[a] = busy[b] = start + duration

    for mover, target in interactions:
        while True:
            pm, pt = phys_of[mover], phys_of[target]
            if D[pm, pt] == 1:
                break
            nxt = next(nb for nb in adj[pm] if D[nb, pt] == D[pm, pt] - 1)
            other = logical_at.get(nxt)
            logical_at[nxt] = mover
            phys_of[mover] = nxt
            if other is None:
                del logical_at[pm]
            else:
                logical_at[pm] = other
                phys_of[other] = pm
            swaps += 1
            if (min(pm, nxt), max(pm, nxt)) in cross:
                cross_swaps += 1
            schedule(pm, nxt, 3)
        pm, pt = phys_of[mover], phys_of[target]
        if (min(pm, pt), max(pm, pt)) in cross:
            cross_cx += 1
        schedule(pm, pt, 1)
    return {'swaps': swaps, 'depth': max(busy), 'cross_cx': cross_cx,
            'cross_swaps': cross_swaps}


def test_route_round_matches_reference():
    """提速后的路由器与原实现输出完全相同（无跨层 / 有跨层两种情况）"""
    code = load_hal_code('D4_3')
    size = default_grid_size(code)
    phys = positions_to_phys(spring_layout(code, size, seed=0), size)
    inter = code.get_all_interactions()
    g0 = CouplingGraph(size)
    cands = candidate_pairs(inter, phys, g0.dist)
    g1 = CouplingGraph(size, select_marginal(cands, g0.dist, inter, phys, 6))
    for g in (g0, g1):
        assert route_round(inter, phys, g) == _route_round_reference(inter, phys, g)


def _select_marginal_reference(cands, dist, interactions, phys, k, max_degree=1):
    """提速前的 select_marginal 原样实现，用作回归对照"""
    D = dist.astype(np.int64).copy()
    A = np.array([phys[a] for a, _ in interactions])
    B = np.array([phys[b] for _, b in interactions])
    U = np.array([u for u, _ in cands])
    V = np.array([v for _, v in cands])
    degree, chosen = {}, []
    for _ in range(k):
        cur = D[A, B]
        via = np.minimum(D[np.ix_(A, U)] + 1 + D[np.ix_(B, V)],
                         D[np.ix_(A, V)] + 1 + D[np.ix_(B, U)])
        new = np.minimum(cur[:, None], via)
        gain = (np.maximum(cur - 1, 0)[:, None] - np.maximum(new - 1, 0)).sum(axis=0)
        for i, (u, v) in enumerate(cands):
            if degree.get(u, 0) >= max_degree or degree.get(v, 0) >= max_degree:
                gain[i] = -1
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        u, v = cands[best]
        chosen.append((u, v))
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        D = np.minimum(D, np.minimum(D[:, [u]] + 1 + D[[v], :],
                                     D[:, [v]] + 1 + D[[u], :]))
    return chosen


def test_select_marginal_matches_reference():
    """提速后的 marginal 选择与原实现挑出的边集逐条相同"""
    for name, max_degree in (('D4_3', 1), ('D6_3', 1), ('D6_3', 2)):
        code = load_hal_code(name)
        size = default_grid_size(code)
        phys = positions_to_phys(spring_layout(code, size, seed=0), size)
        inter = code.get_all_interactions()
        dist = CouplingGraph(size).dist
        cands = candidate_pairs(inter, phys, dist)
        assert (select_marginal(cands, dist, inter, phys, 10, max_degree)
                == _select_marginal_reference(cands, dist, inter, phys, 10, max_degree))


def test_native_surface_code_needs_no_swaps():
    for d in (3, 5, 7):
        code = SurfaceCode(d)
        pos, size = native_layout(code)
        phys = positions_to_phys(pos, size)
        g = CouplingGraph(size)
        inter = code.get_all_interactions()
        assert static_swap_proxy(inter, phys, g) == 0
        assert route_round(inter, phys, g)['swaps'] == 0


def test_router_keeps_permutation_and_makes_cx_adjacent():
    code = load_hal_code('D4_3')
    size = default_grid_size(code)
    phys = positions_to_phys(spring_layout(code, size, seed=1), size)
    g = CouplingGraph(size)
    res = route_round(code.get_all_interactions(), phys, g)
    assert res['swaps'] > 0
    assert res['depth'] >= len(code.get_all_interactions()) // code.num_physical


def test_marginal_distance_update_matches_recomputed_graph():
    code = load_hal_code('D4_3')
    size = default_grid_size(code)
    phys = positions_to_phys(spring_layout(code, size, seed=2), size)
    inter = code.get_all_interactions()
    g0 = CouplingGraph(size)
    cands = candidate_pairs(inter, phys, g0.dist)

    chosen = select_marginal(cands, g0.dist, inter, phys, k=4)
    g1 = CouplingGraph(size, chosen)
    assert len(chosen) == 4
    assert static_swap_proxy(inter, phys, g1) < static_swap_proxy(inter, phys, g0)

    used = [p for e in chosen for p in e]
    assert len(used) == len(set(used))  # 每个物理比特最多一条跨层耦合器


def test_greedy_selection_is_prefix_consistent():
    """实验10 用最大预算结果的前缀代替逐个预算重算，这里验证两者完全相同"""
    code = load_hal_code('D4_3')
    size = default_grid_size(code)
    phys = positions_to_phys(spring_layout(code, size, seed=4), size)
    inter = code.get_all_interactions()
    dist = CouplingGraph(size).dist
    cands = candidate_pairs(inter, phys, dist)
    m_all = select_marginal(cands, dist, inter, phys, 7)
    l_all = select_longest(cands, dist, 7)
    for k in (1, 3, 7):
        assert select_marginal(cands, dist, inter, phys, k) == m_all[:k]
        assert select_longest(cands, dist, k) == l_all[:k]


def test_length_cap_filters_long_couplers():
    code = load_hal_code('D4_3')
    size = default_grid_size(code)
    phys = positions_to_phys(spring_layout(code, size, seed=5), size)
    inter = code.get_all_interactions()
    dist = CouplingGraph(size).dist
    for L in (3, 6):
        for u, v in candidate_pairs(inter, phys, dist, size=size, max_len=L):
            (r1, c1), (r2, c2) = divmod(u, size), divmod(v, size)
            assert 2 <= abs(r1 - r2) + abs(c1 - c2) <= L


def test_marginal_not_worse_than_longest_on_proxy():
    code = load_hal_code('D6_3')
    size = default_grid_size(code)
    phys = positions_to_phys(spring_layout(code, size, seed=3), size)
    inter = code.get_all_interactions()
    g0 = CouplingGraph(size)
    cands = candidate_pairs(inter, phys, g0.dist)
    k = 5
    m = static_swap_proxy(inter, phys, CouplingGraph(size, select_marginal(cands, g0.dist, inter, phys, k)))
    l = static_swap_proxy(inter, phys, CouplingGraph(size, select_longest(cands, g0.dist, k)))
    assert m <= l


def test_exp15_sabre_uses_layout_mapping():
    """exp15 的 SABRE 必须从给定布局出发：表面码原生布局下应不需要 SWAP"""
    from foldq.experiments.exp15_hal_full import route_with_sabre
    code = SurfaceCode(3)
    pos, size = native_layout(code)
    phys = positions_to_phys(pos, size)
    r = route_with_sabre(code.get_all_interactions(), phys, CouplingGraph(size), n_trials=1)
    assert r['swaps'] == 0
