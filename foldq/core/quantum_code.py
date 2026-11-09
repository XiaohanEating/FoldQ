"""
FoldQ核心数据结构：量子纠错码

定义量子纠错码的基本表示，包括：
- 稳定子群
- 逻辑算符
- 综合征提取电路的交互模式
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import numpy as np


@dataclass
class PauliOperator:
    """Pauli算符：I, X, Y, Z的张量积"""

    ops: List[str]  # 每个量子比特上的操作，例如 ['X', 'Z', 'I', 'Y']

    def __post_init__(self):
        valid = {'I', 'X', 'Y', 'Z'}
        for op in self.ops:
            if op not in valid:
                raise ValueError(f"无效的Pauli算符: {op}")

    @property
    def num_qubits(self) -> int:
        return len(self.ops)

    @property
    def support(self) -> Set[int]:
        """返回非I算符作用的量子比特位置"""
        return {i for i, op in enumerate(self.ops) if op != 'I'}

    @property
    def weight(self) -> int:
        """Pauli算符的权重（非I算符的数量）"""
        return len(self.support)

    def commute_with(self, other: 'PauliOperator') -> bool:
        """判断两个Pauli算符是否对易"""
        if self.num_qubits != other.num_qubits:
            raise ValueError("量子比特数不匹配")

        # 两个Pauli算符对易 <=> 它们反对易的位置数量是偶数
        anticommute_count = 0
        for s, o in zip(self.ops, other.ops):
            if s == 'I' or o == 'I':
                continue
            # X和Z反对易，X和Y反对易，Y和Z反对易
            if (s, o) in [('X', 'Z'), ('Z', 'X'),
                          ('X', 'Y'), ('Y', 'X'),
                          ('Y', 'Z'), ('Z', 'Y')]:
                anticommute_count += 1

        return anticommute_count % 2 == 0

    def __str__(self) -> str:
        # 只显示非I的部分，例如 X_0 Z_3 Y_5
        parts = [f"{op}_{i}" for i, op in enumerate(self.ops) if op != 'I']
        return ' '.join(parts) if parts else 'I'


@dataclass
class Stabilizer:
    """稳定子：包含Pauli算符和测量信息"""

    operator: PauliOperator
    ancilla_qubit: int  # 辅助量子比特索引
    stabilizer_type: str  # 'X' 或 'Z'

    def __post_init__(self):
        if self.stabilizer_type not in ['X', 'Z']:
            raise ValueError("稳定子类型必须是'X'或'Z'")

    @property
    def data_qubits(self) -> Set[int]:
        """稳定子作用的数据量子比特"""
        return self.operator.support

    def get_interactions(self) -> List[Tuple[int, int]]:
        """
        返回综合征提取需要的双量子比特门
        返回格式: [(辅助比特, 数据比特1), (辅助比特, 数据比特2), ...]
        """
        interactions = []
        for data_qubit in self.data_qubits:
            interactions.append((self.ancilla_qubit, data_qubit))
        return interactions


class QuantumCode:
    """量子纠错码的基类"""

    def __init__(self, num_physical: int, num_logical: int, distance: int):
        self.num_physical = num_physical      # 物理量子比特数 n
        self.num_logical = num_logical        # 逻辑量子比特数 k
        self.distance = distance              # 码距 d

        self.stabilizers: List[Stabilizer] = []
        self.logical_x: List[PauliOperator] = []
        self.logical_z: List[PauliOperator] = []

        # 数据比特编号为 0..num_data-1，辅助比特在其后；子类负责设置
        self.num_data: int = None

    def parity_check_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """导出 (Hx, Hz)，列为数据比特，用于 GF(2) 有效性检查"""
        if self.num_data is None:
            raise ValueError("num_data 未设置，无法导出校验矩阵")
        xs = [s for s in self.stabilizers if s.stabilizer_type == 'X']
        zs = [s for s in self.stabilizers if s.stabilizer_type == 'Z']
        Hx = np.zeros((len(xs), self.num_data), dtype=np.uint8)
        Hz = np.zeros((len(zs), self.num_data), dtype=np.uint8)
        for i, s in enumerate(xs):
            Hx[i, sorted(s.data_qubits)] = 1
        for i, s in enumerate(zs):
            Hz[i, sorted(s.data_qubits)] = 1
        return Hx, Hz

    def add_stabilizer(self, stabilizer: Stabilizer):
        """添加稳定子"""
        self.stabilizers.append(stabilizer)

    def verify_commutation(self) -> Tuple[bool, str]:
        """
        验证所有稳定子是否相互对易

        返回: (是否通过, 错误信息)
        """
        for i, s1 in enumerate(self.stabilizers):
            for j, s2 in enumerate(self.stabilizers[i+1:], start=i+1):
                if not s1.operator.commute_with(s2.operator):
                    return False, f"稳定子 {i} 和 {j} 不对易"
        return True, "所有稳定子相互对易"

    def get_all_interactions(self) -> List[Tuple[int, int]]:
        """
        获取所有稳定子测量需要的双量子比特门交互

        返回: [(比特1, 比特2), ...] 列表
        """
        interactions = []
        for stab in self.stabilizers:
            interactions.extend(stab.get_interactions())
        return interactions

    def get_interaction_frequency(self) -> dict:
        """
        统计每对量子比特之间的交互频率

        返回: {(比特1, 比特2): 频率}
        """
        from collections import Counter
        interactions = self.get_all_interactions()
        # 归一化：(i, j) 和 (j, i) 视为同一对
        normalized = [tuple(sorted(pair)) for pair in interactions]
        return dict(Counter(normalized))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.num_physical}, k={self.num_logical}, d={self.distance})"


class SurfaceCode(QuantumCode):
    """
    旋转表面码 [[d², 1, d]]

    数据比特 (i, j), 0 <= i, j < d，编号 i*d + j。
    格子 (r, c), -1 <= r, c <= d-1，覆盖数据比特 (r..r+1, c..c+1) ∩ 网格。
    类型按棋盘着色：(r + c) 偶数为 X，奇数为 Z。
    内部格子 (d-1)² 个，权重 4；边界只保留权重 2 的格子：
    上下边界保留 X 型，左右边界保留 Z 型；角上权重 1 的格子不保留。
    合计 d²-1 个稳定子，X、Z 各 (d²-1)/2 个。

    native_positions() 给出标准硬件布局（把码格旋转 45° 放上方格）：
    数据比特 (i,j) -> (i+j, i-j+d-1)，格子 (r,c) -> (r+c+1, r-c+d-1)，
    放在 (2d-1)×(2d-1) 方格上。每个辅助比特与它的数据比特都是最近邻，
    因此无需 SWAP。
    """

    def __init__(self, distance: int):
        if distance % 2 == 0 or distance < 3:
            raise ValueError("表面码距离必须是 >=3 的奇数")

        num_data = distance ** 2
        num_ancilla = distance ** 2 - 1
        super().__init__(num_data + num_ancilla, 1, distance)

        self.distance = distance
        self.num_data = num_data
        self._plaquette_of_ancilla = {}
        self._build_stabilizers()

    def _build_stabilizers(self):
        d = self.distance
        ancilla = self.num_data
        for r in range(-1, d):
            for c in range(-1, d):
                support = [i * d + j
                           for i in (r, r + 1) for j in (c, c + 1)
                           if 0 <= i < d and 0 <= j < d]
                stype = 'X' if (r + c) % 2 == 0 else 'Z'
                if len(support) == 4:
                    keep = True
                elif len(support) == 2:
                    on_top_bottom = r in (-1, d - 1)
                    keep = (stype == 'X') == on_top_bottom
                else:
                    keep = False
                if not keep:
                    continue

                ops = ['I'] * self.num_physical
                for q in support:
                    ops[q] = stype
                self.add_stabilizer(Stabilizer(
                    operator=PauliOperator(ops),
                    ancilla_qubit=ancilla,
                    stabilizer_type=stype,
                ))
                self._plaquette_of_ancilla[ancilla] = (r, c)
                ancilla += 1

    def native_grid_size(self) -> int:
        return 2 * self.distance - 1

    def native_positions(self) -> Dict[int, Tuple[int, int]]:
        """标准最近邻硬件布局：比特 -> (行, 列)"""
        d = self.distance
        pos = {i * d + j: (i + j, i - j + d - 1) for i in range(d) for j in range(d)}
        for a, (r, c) in self._plaquette_of_ancilla.items():
            pos[a] = (r + c + 1, r - c + d - 1)
        return pos


def test_surface_code():
    """测试表面码构建"""
    print("测试 d=3 表面码")
    code = SurfaceCode(distance=3)

    print(f"物理量子比特数: {code.num_physical}")
    print(f"稳定子数量: {len(code.stabilizers)}")

    # 验证对易性
    valid, msg = code.verify_commutation()
    print(f"稳定子对易性检验: {msg}")

    # 统计交互
    interactions = code.get_all_interactions()
    print(f"总交互次数: {len(interactions)}")

    freq = code.get_interaction_frequency()
    print(f"\n交互频率分布:")
    for pair, count in sorted(freq.items(), key=lambda x: -x[1])[:10]:
        print(f"  比特对 {pair}: {count}次")

    # 显示部分稳定子
    print(f"\n前3个稳定子:")
    for i, stab in enumerate(code.stabilizers[:3]):
        print(f"  稳定子{i} ({stab.stabilizer_type}): {stab.operator}")
        print(f"    作用数据比特: {stab.data_qubits}")


if __name__ == "__main__":
    test_surface_code()
