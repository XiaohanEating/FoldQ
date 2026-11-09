"""
跨层连接选择：给定布局与预算 k，挑 k 条跨层耦合器

候选集合：所有"辅助-数据"交互对在当前布局下的物理端点（距离 >= 2）。
物理约束：每个物理比特最多 max_degree 条跨层耦合器（默认 1）。

random   ：候选中随机挑，对照组。若其他策略赢不了它，说明"选哪几条"不重要
longest  ：按当前距离从长到短挑，等价于旧 FoldingSelector（频率全为 1 时）
marginal ：每步算出加入每条候选后静态 SWAP 代理的总下降量，挑最大的；
           距离矩阵随已选捷径更新，所以能看到一条捷径同时服务多对交互、
           以及已选捷径让其他候选失效。优化的是静态代理，
           是否真的减少路由 SWAP 由 route_round 独立检验。
"""

from typing import Dict, List, Tuple

import numpy as np


def candidate_pairs(interactions, phys: Dict[int, int], dist: np.ndarray,
                    size: int = None, max_len: int = None):
    """
    max_len：耦合器长度上限，按两端在比特平面上的曼哈顿距离计（单位：比特间距），
    需要同时给出方格边长 size。None 表示不限长度。
    """
    pairs = {tuple(sorted((phys[a], phys[b]))) for a, b in interactions}
    out = []
    for u, v in sorted(pairs):
        if dist[u, v] < 2:
            continue
        if max_len is not None:
            (r1, c1), (r2, c2) = divmod(u, size), divmod(v, size)
            if abs(r1 - r2) + abs(c1 - c2) > max_len:
                continue
        out.append((u, v))
    return out


def _take(ordered, k: int, max_degree: int) -> List[Tuple[int, int]]:
    degree, chosen = {}, []
    for u, v in ordered:
        if len(chosen) == k:
            break
        if degree.get(u, 0) < max_degree and degree.get(v, 0) < max_degree:
            chosen.append((u, v))
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
    return chosen


def select_random(cands, dist, k, rng: np.random.Generator, max_degree=1):
    order = [cands[i] for i in rng.permutation(len(cands))]
    return _take(order, k, max_degree)


def select_longest(cands, dist, k, max_degree=1):
    order = sorted(cands, key=lambda p: (-dist[p], p))
    return _take(order, k, max_degree)


def select_marginal(cands, dist, interactions, phys, k, max_degree=1,
                    wire_penalty: float = 0.0):
    """
    wire_penalty > 0 时打分改为 代理SWAP增益 − wire_penalty×边长（布线成本项，exp30 多目标变体）。默认 0.0 行为不变。
    """
    # 距离不超过比特数，int32 足够；先按行取出交互端点的距离行，再按列切片，
    # 与直接 D[np.ix_(A, U)] 数值相同，只是更快
    D = dist.astype(np.int32)
    A = np.array([phys[a] for a, _ in interactions])
    B = np.array([phys[b] for _, b in interactions])
    U = np.array([u for u, _ in cands])
    V = np.array([v for _, v in cands])
    degree = {}
    chosen = []
    blocked = np.zeros(len(cands), dtype=bool)

    for _ in range(k):
        DA, DB = D[A], D[B]
        cur = DA[np.arange(len(B)), B]
        via = np.minimum(DA[:, U] + DB[:, V], DA[:, V] + DB[:, U])
        via += 1
        np.minimum(via, cur[:, None], out=via)
        np.maximum(via, 1, out=via)
        gain = np.maximum(cur - 1, 0).sum() - (via - 1).sum(axis=0, dtype=np.int64)
        if wire_penalty > 0:
            gain = gain.astype(np.float64) - wire_penalty * dist[U, V]
        gain[blocked] = -1

        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break

        u, v = cands[best]
        chosen.append((u, v))
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        for i, (a, b) in enumerate(cands):
            if (a in (u, v) or b in (u, v)) and (
                    degree.get(a, 0) >= max_degree or degree.get(b, 0) >= max_degree):
                blocked[i] = True
        D = np.minimum(D, np.minimum(D[:, [u]] + 1 + D[[v], :],
                                     D[:, [v]] + 1 + D[[u], :]))
    return chosen
