"""
FoldQ核心数据结构：量子比特映射与门调度

定义：
- 量子比特映射（逻辑→物理）
- 门调度（操作→时间）
- SWAP路由记录
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class GateType(Enum):
    """门操作类型"""
    SINGLE = 1    # 单量子比特门
    TWO = 2       # 双量子比特门
    MEASURE = 3   # 测量
    RESET = 4     # 复位
    SWAP = 5      # SWAP门（路由用）


@dataclass
class Gate:
    """门操作"""
    gate_type: GateType
    qubits: Tuple[int, ...]  # 作用的量子比特（逻辑编号）
    time_step: Optional[int] = None  # 调度的时间步（由调度器填充）

    def __post_init__(self):
        # 验证量子比特数量
        if self.gate_type == GateType.SINGLE:
            if len(self.qubits) != 1:
                raise ValueError("单量子比特门需要1个量子比特")
        elif self.gate_type in [GateType.TWO, GateType.SWAP]:
            if len(self.qubits) != 2:
                raise ValueError("双量子比特门/SWAP需要2个量子比特")
        elif self.gate_type in [GateType.MEASURE, GateType.RESET]:
            if len(self.qubits) != 1:
                raise ValueError("测量/复位需要1个量子比特")

    def is_two_qubit(self) -> bool:
        return self.gate_type in [GateType.TWO, GateType.SWAP]

    def __repr__(self) -> str:
        qubits_str = ','.join(map(str, self.qubits))
        time_str = f" @t={self.time_step}" if self.time_step is not None else ""
        return f"{self.gate_type.name}({qubits_str}){time_str}"


class Mapping:
    """量子比特映射：逻辑量子比特 → 物理量子比特"""

    def __init__(self, num_logical: int, num_physical: int):
        self.num_logical = num_logical
        self.num_physical = num_physical

        # 映射：logical_qubit -> physical_qubit
        self._forward: Dict[int, int] = {}

        # 反向映射：physical_qubit -> logical_qubit (or None)
        self._backward: Dict[int, Optional[int]] = {i: None for i in range(num_physical)}

    def map(self, logical: int, physical: int):
        """建立映射关系"""
        if logical >= self.num_logical:
            raise ValueError(f"逻辑量子比特索引超出范围: {logical}")
        if physical >= self.num_physical:
            raise ValueError(f"物理量子比特索引超出范围: {physical}")

        # 检查冲突
        if logical in self._forward:
            old_physical = self._forward[logical]
            if old_physical != physical:
                # 需要先释放旧的映射
                self._backward[old_physical] = None

        if self._backward[physical] is not None:
            old_logical = self._backward[physical]
            if old_logical != logical:
                raise ValueError(f"物理比特{physical}已被逻辑比特{old_logical}占用")

        self._forward[logical] = physical
        self._backward[physical] = logical

    def get_physical(self, logical: int) -> int:
        """获取逻辑量子比特对应的物理量子比特"""
        if logical not in self._forward:
            raise ValueError(f"逻辑量子比特{logical}未映射")
        return self._forward[logical]

    def get_logical(self, physical: int) -> Optional[int]:
        """获取物理量子比特对应的逻辑量子比特（可能为空）"""
        return self._backward.get(physical)

    def is_mapped(self, logical: int) -> bool:
        """判断逻辑量子比特是否已映射"""
        return logical in self._forward

    def swap(self, physical1: int, physical2: int):
        """交换两个物理量子比特上的逻辑量子比特"""
        logical1 = self._backward[physical1]
        logical2 = self._backward[physical2]

        # 交换映射
        self._backward[physical1] = logical2
        self._backward[physical2] = logical1

        if logical1 is not None:
            self._forward[logical1] = physical2
        if logical2 is not None:
            self._forward[logical2] = physical1

    def get_all_mappings(self) -> Dict[int, int]:
        """获取所有映射"""
        return self._forward.copy()

    def __repr__(self) -> str:
        mappings = ', '.join(f"{l}->{p}" for l, p in sorted(self._forward.items()))
        return f"Mapping({mappings})"


@dataclass
class Schedule:
    """门调度结果"""

    gates: List[Gate]  # 所有门操作（已排序）
    total_time_steps: int  # 总时间步数
    swap_gates: List[Gate] = field(default_factory=list)  # 插入的SWAP门

    @property
    def num_swaps(self) -> int:
        """SWAP门数量"""
        return len(self.swap_gates)

    @property
    def num_two_qubit_gates(self) -> int:
        """双量子比特门总数（包括SWAP）"""
        return sum(1 for g in self.gates if g.is_two_qubit())

    @property
    def depth(self) -> int:
        """电路深度（时间步数）"""
        return self.total_time_steps

    def get_critical_path(self) -> List[Gate]:
        """获取关键路径上的门"""
        # 简化实现：返回所有最后时间步的门
        if not self.gates:
            return []
        max_time = max(g.time_step for g in self.gates if g.time_step is not None)
        return [g for g in self.gates if g.time_step == max_time]

    def __repr__(self) -> str:
        return (f"Schedule(gates={len(self.gates)}, depth={self.depth}, "
                f"swaps={self.num_swaps}, 2q_gates={self.num_two_qubit_gates})")


@dataclass
class MappingResult:
    """
    映射与路由的完整结果

    包含：
    - 初始映射
    - 最终调度
    - 性能指标
    """

    initial_mapping: Mapping
    schedule: Schedule
    topology_name: str  # 使用的拓扑名称（如 "2D_Grid", "Folded"）

    # 性能指标
    metrics: Dict[str, float] = field(default_factory=dict)

    def set_metric(self, name: str, value: float):
        """设置性能指标"""
        self.metrics[name] = value

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """获取性能指标"""
        return self.metrics.get(name, default)

    def summary(self) -> str:
        """生成结果摘要"""
        lines = [
            f"拓扑: {self.topology_name}",
            f"映射: {len(self.initial_mapping.get_all_mappings())} 个量子比特",
            f"调度: {self.schedule}",
            f"指标:"
        ]
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name}: {value:.2f}")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"MappingResult({self.topology_name}, swaps={self.schedule.num_swaps})"


def test_mapping():
    """测试映射功能"""
    print("=== 测试量子比特映射 ===")
    mapping = Mapping(num_logical=4, num_physical=9)

    # 建立映射
    mapping.map(0, 0)
    mapping.map(1, 2)
    mapping.map(2, 4)
    mapping.map(3, 8)

    print(f"初始映射: {mapping}")

    print(f"逻辑0 -> 物理{mapping.get_physical(0)}")
    print(f"物理2 -> 逻辑{mapping.get_logical(2)}")

    # 测试SWAP
    print("\n执行 SWAP(0, 2)")
    mapping.swap(0, 2)
    print(f"SWAP后: {mapping}")

    print("\n=== 测试门操作 ===")
    gates = [
        Gate(GateType.RESET, (0,)),
        Gate(GateType.TWO, (0, 1)),
        Gate(GateType.TWO, (1, 2)),
        Gate(GateType.MEASURE, (0,)),
    ]

    # 简单调度：按顺序分配时间
    for i, gate in enumerate(gates):
        gate.time_step = i

    schedule = Schedule(gates=gates, total_time_steps=len(gates))
    print(f"\n调度结果: {schedule}")
    print("门序列:")
    for gate in gates:
        print(f"  {gate}")


if __name__ == "__main__":
    test_mapping()
