"""
Z 基存储实验电路（多轮稳定子测量），按路由结果逐门展开

每一轮：
1. 从当前布局路由一次（route_round），路由结束后的布局作为下一轮起点；
2. 辅助比特在其当前位置复位，X 型再加 H；
3. 回放路由操作：SWAP 展开为 3 个 CNOT；CX 按稳定子类型定方向——
   Z 型：数据比特控制、辅助比特受控；X 型：辅助比特控制、数据比特受控；
4. X 型辅助比特加 H，所有辅助比特按逻辑编号顺序测量（结果以 p_m 翻转）。

探测器：Z 型从第 1 轮起（初态 |0…0⟩ 下 Z 稳定子取值确定），X 型（可选）从第 2 轮起，
都比较相邻两轮结果；最后测量全部数据比特，用其奇偶与最后一轮 Z 型结果再比较一次。
观测量：k 个独立的逻辑 Z 算符（gf2.logical_basis(Hx, Hz)）。

噪声：每个 CNOT 后 DEPOLARIZE2，平面 p_cx、跨层 alpha·p_cx；测量翻转 p_m。
未计空闲噪声与复位噪声，与解析保真度模型的计数口径一致。
"""

from typing import Dict, List, Tuple

import numpy as np
import stim

from foldq.codes.gf2 import logical_basis
from foldq.routing.router import route_round


def route_rounds(interactions, phys0: Dict[int, int], graph, n_rounds: int) -> List[Tuple[List, Dict[int, int]]]:
    """逐轮路由，返回 [(该轮操作序列, 该轮起始布局), ...]"""
    rounds, phys = [], dict(phys0)
    for _ in range(n_rounds):
        r = route_round(interactions, phys, graph, return_ops=True)
        rounds.append((r['operations'], phys))
        phys = r['final_phys']
    return rounds


def _stab_type_of_ancilla(code) -> Dict[int, str]:
    return {s.ancilla_qubit: s.stabilizer_type for s in code.stabilizers}


def build_memory_circuit(code, rounds: List[Tuple[List, Dict[int, int]]],
                         p_cx: float, alpha: float, p_m: float,
                         x_detectors: bool = False) -> stim.Circuit:
    """
    旧版构建器（无时间片/空闲噪声），exp25/26/28b 的电路生成保持不变。

    x_detectors=False：只声明 Z 型探测器。Z 基存储实验的逻辑观测量只受 X 方向错误影响，
    X、Z 分开解码时 X 型探测器用不上；X 型稳定子照常测量，其 CNOT 噪声全部计入。
    这样错误模型只含与 Z 型探测器/观测量相关的部分，矩阵小一个数量级，
    代价是解码器不利用 Y 错误在两侧的关联，基线与 FoldQ 同样处理。
    """
    stype = _stab_type_of_ancilla(code)
    n = code.num_data
    ancillas = list(range(n, code.num_physical))
    A = len(ancillas)
    _, Hz = code.parity_check_matrices()
    z_rows = [s for s in code.stabilizers if s.stabilizer_type == 'Z']
    Hx, _ = code.parity_check_matrices()
    Lz = logical_basis(Hx, Hz)

    c = stim.Circuit()
    phys = None
    for r_idx, (ops, phys_start) in enumerate(rounds):
        at = {p: q for q, p in phys_start.items()}
        pos = dict(phys_start)

        c.append('R', [pos[a] for a in ancillas])
        c.append('H', [pos[a] for a in ancillas if stype[a] == 'X'])

        for op, q1, q2, is_cross in ops:
            p = alpha * p_cx if is_cross else p_cx
            if op == 'SWAP':
                for a, b in ((q1, q2), (q2, q1), (q1, q2)):
                    c.append('CNOT', [a, b])
                    c.append('DEPOLARIZE2', [a, b], p)
                u, v = at.get(q1), at.get(q2)
                at[q1], at[q2] = v, u
                if u is not None:
                    pos[u] = q2
                if v is not None:
                    pos[v] = q1
            else:
                anc = at[q1]  # route_round 的 CX 形如 (辅助比特位置, 数据比特位置)
                ctrl, tgt = (q2, q1) if stype[anc] == 'Z' else (q1, q2)
                c.append('CNOT', [ctrl, tgt])
                c.append('DEPOLARIZE2', [ctrl, tgt], p)

        c.append('H', [pos[a] for a in ancillas if stype[a] == 'X'])
        c.append('M', [pos[a] for a in ancillas], p_m)
        for j, a in enumerate(ancillas):
            cur = stim.target_rec(-A + j)
            if r_idx == 0:
                if stype[a] == 'Z':
                    c.append('DETECTOR', [cur])
            elif stype[a] == 'Z' or x_detectors:
                c.append('DETECTOR', [cur, stim.target_rec(-2 * A + j)])
        phys = pos

    # 最后测量数据比特（Z 基）
    c.append('M', [phys[i] for i in range(n)], p_m)
    anc_index = {a: j for j, a in enumerate(ancillas)}
    for s in z_rows:
        targets = [stim.target_rec(-n + i) for i in sorted(s.data_qubits)]
        targets.append(stim.target_rec(-n - A + anc_index[s.ancilla_qubit]))
        c.append('DETECTOR', targets)
    for m, row in enumerate(Lz):
        c.append('OBSERVABLE_INCLUDE', [stim.target_rec(-n + i) for i in np.nonzero(row)[0]], m)
    return c


# ---------------------------------------------------------------------------
# exp29：带时间片展开的电路（门时长 β、空闲噪声、测量时长）
# ---------------------------------------------------------------------------

def route_rounds_timed(interactions, phys0: Dict[int, int], graph, n_rounds: int,
                       cross_time_factor: float = 1.0):
    """逐轮路由并导出 ASAP 调度（route_round 的 return_schedule），单位半 tick。"""
    rounds, phys = [], dict(phys0)
    for _ in range(n_rounds):
        r = route_round(interactions, phys, graph, return_ops=True,
                        cross_time_factor=cross_time_factor, return_schedule=True)
        rounds.append((r['schedule'], r['final_phys'], dict(phys)))
        phys = r['final_phys']
    return rounds


def build_memory_circuit_timed(code, rounds_timed, p_cx: float, alpha: float,
                               p_m: float, p_idle: float = 0.0,
                               t_meas: int = 20, x_detectors: bool = False) -> stim.Circuit:
    """
    按时间片展开的存储实验电路（在旧版语义上增加三项物理效应）：
      1. 跨层门时长 = 平面门的 β 倍（β 经 route_round 的 cross_time_factor 已编入调度）；
      2. 空闲噪声：每个平面 CX 时长（2 半 tick）内空闲的物理比特发 DEPOLARIZE1，
         连续空闲段合并为一条（概率 1-(1-p_idle)^段tick数）；
      3. 测量时长 t_meas（半 tick）：测量期间数据位置的比特空闲累积错误。

    p_idle=0 且 β=1 时与 build_memory_circuit 语义等价（门集合与噪声相同，
    发射顺序按 ASAP 时间重排，不同比特的操作互相对易，DEM 一致）。
    探测器/观测量声明与旧版完全相同。
    """
    stype = _stab_type_of_ancilla(code)
    n = code.num_data
    ancillas = list(range(n, code.num_physical))
    A = len(ancillas)
    _, Hz = code.parity_check_matrices()
    z_rows = [s for s in code.stabilizers if s.stabilizer_type == 'Z']
    Hx, _ = code.parity_check_matrices()
    Lz = logical_basis(Hx, Hz)
    n_phys = code.num_physical

    c = stim.Circuit()
    for r_idx, (sched, final_phys, phys_start) in enumerate(rounds_timed):
        at = {p: q for q, p in phys_start.items()}
        pos = dict(phys_start)
        c.append('R', [pos[a] for a in ancillas])
        c.append('H', [pos[a] for a in ancillas if stype[a] == 'X'])

        # 事件表：门（SWAP 展开 3 个 CNOT 子事件）与空闲段，均 (start, kind, ...)
        events = []
        busy_until = {}          # 物理位置 -> 忙到的时刻（用于空闲段计算）
        for op, q1, q2, is_cross, start, dur in sched:
            p_gate = alpha * p_cx if is_cross else p_cx
            if op == 'SWAP':
                step = dur / 3   # 3 个 CNOT 均分整个 SWAP 时长
                pairs = ((q1, q2), (q2, q1), (q1, q2))
                for i, (a, b) in enumerate(pairs):
                    events.append((start + i * step, 'gate', a, b, p_gate, is_cross))
                # SWAP 交换两位置上的逻辑比特
                u, v = at.get(q1), at.get(q2)
                at[q1], at[q2] = v, u
                if u is not None:
                    pos[u] = q2
                if v is not None:
                    pos[v] = q1
            else:
                anc = at[q1]
                ctrl, tgt = (q2, q1) if stype[anc] == 'Z' else (q1, q2)
                events.append((start, 'gate', ctrl, tgt, p_gate, is_cross))
            for q in (q1, q2):
                busy_until[q] = max(busy_until.get(q, 0), start + dur)

        round_end = max(busy_until.values())
        # 每个物理位置的空闲段：[0, round_end) 内不在任何门覆盖的部分。
        # 位置编号是方格格点（0..size²-1），不是码的比特编号
        all_slots = sorted({q for op, q1, q2, _, _, _ in sched for q in (q1, q2)})
        intervals = {q: [] for q in all_slots}
        for op, q1, q2, is_cross, start, dur in sched:
            for q in (q1, q2):
                intervals[q].append((start, start + dur))
        for q in all_slots:
            ivs = sorted(intervals[q])
            cur = 0.0
            for a, b in ivs:
                if a > cur and p_idle > 0:
                    events.append((cur, 'idle', q, a - cur))
                cur = max(cur, b)
            if round_end > cur and p_idle > 0:
                events.append((cur, 'idle', q, round_end - cur))

        events.sort(key=lambda e: e[0])
        for ev in events:
            if ev[1] == 'gate':
                _, _, a, b, p_gate, _ = ev
                c.append('CNOT', [a, b])
                c.append('DEPOLARIZE2', [a, b], p_gate)
            else:
                _, _, q, span = ev
                ticks = max(1, round(span / 2))
                # 复核 5.4 修正：t 段 DEPOLARIZE1(p) 的等效组合通道概率是
                # (3/4)[1-(1-4p/3)^t]，不是 1-(1-p)^t（后者把 Bloch 缩放当概率复合）
                c.append('DEPOLARIZE1', [q], 0.75 * (1 - (1 - 4 * p_idle / 3) ** ticks))

        # 测量阶段：数据位置空闲 t_meas（辅助在被测量）
        if p_idle > 0 and t_meas > 0:
            ticks = max(1, round(t_meas / 2))
            c.append('DEPOLARIZE1', [pos[i] for i in range(n)],
                     0.75 * (1 - (1 - 4 * p_idle / 3) ** ticks))
        c.append('H', [pos[a] for a in ancillas if stype[a] == 'X'])
        c.append('M', [pos[a] for a in ancillas], p_m)
        for j, a in enumerate(ancillas):
            cur = stim.target_rec(-A + j)
            if r_idx == 0:
                if stype[a] == 'Z':
                    c.append('DETECTOR', [cur])
            elif stype[a] == 'Z' or x_detectors:
                c.append('DETECTOR', [cur, stim.target_rec(-2 * A + j)])

    c.append('M', [pos[i] for i in range(n)], p_m)
    anc_index = {a: j for j, a in enumerate(ancillas)}
    for s in z_rows:
        targets = [stim.target_rec(-n + i) for i in sorted(s.data_qubits)]
        targets.append(stim.target_rec(-n - A + anc_index[s.ancilla_qubit]))
        c.append('DETECTOR', targets)
    for m, row in enumerate(Lz):
        c.append('OBSERVABLE_INCLUDE', [stim.target_rec(-n + i) for i in np.nonzero(row)[0]], m)
    return c
