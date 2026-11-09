"""FoldQ核心模块"""

from foldq.core.quantum_code import QuantumCode, SurfaceCode, PauliOperator, Stabilizer
from foldq.core.hardware import (
    HardwareTopology,
    GridTopology,
    FoldedTopology,
    add_cross_layer_edges,
    select_diagonal_cross_layer_edges
)
from foldq.core.mapping import (
    Mapping,
    Gate,
    GateType,
    Schedule,
    MappingResult
)

__all__ = [
    # 纠错码
    'QuantumCode',
    'SurfaceCode',
    'PauliOperator',
    'Stabilizer',

    # 硬件拓扑
    'HardwareTopology',
    'GridTopology',
    'FoldedTopology',
    'add_cross_layer_edges',
    'select_diagonal_cross_layer_edges',

    # 映射与调度
    'Mapping',
    'Gate',
    'GateType',
    'Schedule',
    'MappingResult',
]
