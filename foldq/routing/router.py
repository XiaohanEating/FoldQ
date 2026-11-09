"""
SWAP 开销的两种算法

static_swap_proxy：每对交互独立计 max(0, 距离-1)，与旧版 compute_routing_cost 口径相同。
    它默认一条跨层捷径可被任意多对交互同时复用，不计拥塞。

route_round：真实执行一轮综合征提取。按稳定子顺序逐个执行 CX，若两比特不相邻，
    就把辅助比特沿最短路一步步 SWAP 过去，途中被换走的比特位置随之改变，
    后续 CX 按更新后的位置计算。因此它能反映比特被挤开的代价。
    CX 逐个串行发出，多对交互共用一条捷径造成的拥塞主要体现在 depth 上，
    而不是 SWAP 数上。
    这是确定性的贪心路由器，不是 SABRE，只用来检验代理指标是否可信。
"""

from typing import Dict, List, Tuple

from foldq.routing.grid import CouplingGraph


def static_swap_proxy(interactions: List[Tuple[int, int]], phys: Dict[int, int],
                      graph: CouplingGraph) -> int:
    D = graph.dist
    return int(sum(max(0, D[phys[a], phys[b]] - 1) for a, b in interactions))


def route_round(interactions: List[Tuple[int, int]], phys: Dict[int, int],
                graph: CouplingGraph, return_ops: bool = False,
                cross_time_factor: float = 1.0,
                return_schedule: bool = False) -> Dict:
    """
    interactions：按执行顺序排列的 (辅助比特, 数据比特)，由辅助比特移动

    返回 swaps、depth（按此执行顺序的 ASAP 深度，CX=1、SWAP=3）、
    cross_cx（落在跨层连接上的 CX 数）、cross_swaps（落在跨层连接上的 SWAP 数）

    return_ops=True 时额外返回 'operations': [(op, q1, q2, is_cross), ...]
    其中 op 是 'CX' 或 'SWAP'，q1、q2 是物理比特编号，is_cross 表示是否在跨层边上

    cross_time_factor：跨层门的时长倍数 β（只影响 ASAP 调度与 depth，不改路由决策和
    SWAP 数）。默认 1.0，行为与旧版完全一致。

    return_schedule=True 时额外返回 'schedule'：[(op, q1, q2, is_cross, start, dur), ...]，
    时间单位为半 tick（平面 CX=2、SWAP=6），跨层门已乘 β。供 build_memory_circuit_timed
    按时间片展开电路（空闲噪声、测量时长）。
    """
    D, adj, cross = graph.dist, graph.adj, graph.cross_set
    phys_of = dict(phys)
    logical_at = {p: q for q, p in phys_of.items()}
    busy = [0] * graph.n

    swaps = cross_swaps = cross_cx = 0
    ops = [] if return_ops else None
    sched = [] if return_schedule else None

    def schedule(a, b, duration, is_cross, op, q1, q2):
        start = max(busy[a], busy[b])
        dur = duration * (cross_time_factor if is_cross else 1.0)
        busy[a] = busy[b] = start + dur
        if sched is not None:
            sched.append((op, q1, q2, is_cross, start, dur))

    for mover, target in interactions:
        # 目标比特在本次移动中不会被换走（路径终点前一步即停），
        # 所以到目标的距离行只取一次，转成列表避免 numpy 标量索引
        pt = phys_of[target]
        drow = D[pt].tolist()
        while True:
            pm = phys_of[mover]
            dm = drow[pm]
            if dm == 1:
                break
            nxt = next(nb for nb in adj[pm] if drow[nb] == dm - 1)
            other = logical_at.get(nxt)
            logical_at[nxt] = mover
            phys_of[mover] = nxt
            if other is None:
                del logical_at[pm]
            else:
                logical_at[pm] = other
                phys_of[other] = pm
            swaps += 1
            edge = (min(pm, nxt), max(pm, nxt))
            is_cross = edge in cross
            if is_cross:
                cross_swaps += 1
            if return_ops:
                ops.append(('SWAP', pm, nxt, is_cross))
            schedule(pm, nxt, 6, is_cross, 'SWAP', pm, nxt)
        pm = phys_of[mover]
        edge = (min(pm, pt), max(pm, pt))
        is_cross = edge in cross
        if is_cross:
            cross_cx += 1
        if return_ops:
            ops.append(('CX', pm, pt, is_cross))
        schedule(pm, pt, 2, is_cross, 'CX', pm, pt)

    result = {'swaps': swaps, 'depth': max(busy) / 2, 'cross_cx': cross_cx,
              'cross_swaps': cross_swaps}
    if return_ops:
        result['operations'] = ops
        result['final_phys'] = phys_of  # 路由结束后各比特所在位置，供下一轮使用
    if sched is not None:
        result['schedule'] = sched
    return result
