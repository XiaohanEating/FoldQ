"""
SABRE 路由器封装：在给定布局上用 qiskit 的 SABRE 路由一轮 QEC 综合征提取

用途：作为 route_round 的对照组，验证我们的贪心路由器是否合理。
SABRE 是启发式的，每次运行结果会略有不同（即使固定种子）。
"""

from typing import Dict, List, Tuple

try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.transpiler.passes import SabreSwap
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def sabre_route(interactions: List[Tuple[int, int]], phys: Dict[int, int],
                coupling_edges: List[Tuple[int, int]], seed: int = 0) -> Dict[str, int]:
    """
    interactions：按执行顺序排列的 (辅助比特, 数据比特)
    phys：逻辑比特 -> 物理比特的初始映射
    coupling_edges：硬件耦合图的边（无向，包括平面+跨层）
    seed：SABRE 的随机种子

    返回 {'swaps': int, 'depth': int, 'cross_swaps': int, 'cross_cx': int}
    depth 是转译后电路的深度，cross_* 需要额外统计（暂时返回 0）
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("qiskit 未安装，无法使用 SABRE 路由器")

    n_logical = len(phys)
    # 找到耦合图中实际用到的最大物理比特编号
    physical_qubits = set()
    for a, b in coupling_edges:
        physical_qubits.add(a)
        physical_qubits.add(b)
    n_physical = max(physical_qubits) + 1 if physical_qubits else n_logical

    # 用物理比特数创建寄存器，逻辑比特映射到其中
    qr = QuantumRegister(n_physical, 'q')
    qc = QuantumCircuit(qr)
    for a, b in interactions:
        qc.cx(phys[a], phys[b])

    coupling_map = CouplingMap(coupling_edges)
    pm = PassManager([SabreSwap(coupling_map, heuristic='decay', seed=seed)])
    qc_routed = pm.run(qc)

    # 统计 SWAP 数：qiskit 插入的 SWAP 门数量
    swaps = qc_routed.count_ops().get('swap', 0)
    depth = qc_routed.depth()

    # TODO: 统计跨层 SWAP 和 CX，需要遍历 qc_routed.data 并查边集
    # 当前返回 0 作为占位
    return {'swaps': swaps, 'depth': depth, 'cross_swaps': 0, 'cross_cx': 0}
