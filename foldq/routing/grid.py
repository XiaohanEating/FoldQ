"""
硬件耦合图：方格最近邻连接 + 跨层连接，一次算出全源最短距离

物理比特编号 p = 行 * size + 列，与 foldq.core.hardware.GridTopology 一致。

平面方格的邻接表和距离矩阵按 size 缓存；加入跨层边时对距离矩阵做增量更新
（每加一条边 (u, v)，D = min(D, D[:,u] + 1 + D[v,:], D[:,v] + 1 + D[u,:])），
结果与对整张图重算最短路完全相同，见 tests/test_routing.py。
"""

from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path


def grid_edges(size: int):
    edges = []
    for r in range(size):
        for c in range(size):
            p = r * size + c
            if c + 1 < size:
                edges.append((p, p + 1))
            if r + 1 < size:
                edges.append((p, p + size))
    return edges


def all_pairs_distance(n: int, edges) -> np.ndarray:
    src = [a for a, _ in edges] + [b for _, b in edges]
    dst = [b for _, b in edges] + [a for a, _ in edges]
    A = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    dist = shortest_path(A, method='D', unweighted=True)
    if not np.all(np.isfinite(dist)):
        raise ValueError("耦合图不连通")
    return dist.astype(np.int64)


def add_edge_distances(D: np.ndarray, u: int, v: int) -> np.ndarray:
    """加入单位长度边 (u, v) 后的全源最短距离"""
    return np.minimum(D, np.minimum(D[:, [u]] + 1 + D[[v], :],
                                    D[:, [v]] + 1 + D[[u], :]))


@lru_cache(maxsize=None)
def _planar(size: int):
    edges = grid_edges(size)
    adj = [[] for _ in range(size * size)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    D = all_pairs_distance(size * size, edges)
    D.setflags(write=False)
    return tuple(edges), tuple(tuple(sorted(nb)) for nb in adj), D


class CouplingGraph:
    def __init__(self, size: int, cross_edges: Iterable[Tuple[int, int]] = ()):
        self.size = size
        self.n = size * size
        planar_edges, planar_adj, D0 = _planar(size)
        self.planar_edges = list(planar_edges)
        self.cross_edges = sorted({tuple(sorted(e)) for e in cross_edges})
        self.cross_set = set(self.cross_edges)

        adj = [list(nb) for nb in planar_adj]
        D = D0
        for a, b in self.cross_edges:
            adj[a].append(b)
            adj[b].append(a)
            D = add_edge_distances(D, a, b)
        for nb in adj:
            nb.sort()
        self.adj = adj
        self.dist = D if self.cross_edges else D0.copy()


def positions_to_phys(positions, size: int):
    """(行, 列) 布局 -> 物理比特编号"""
    return {q: r * size + c for q, (r, c) in positions.items()}
