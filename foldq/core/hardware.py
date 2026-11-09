"""
FoldQ核心数据结构：硬件拓扑

定义量子芯片的物理连接拓扑，包括：
- 2D网格拓扑
- 跨层连接
- 连接成本模型
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import numpy as np


@dataclass
class Connection:
    """物理连接"""
    qubit1: int
    qubit2: int
    connection_type: str  # '2D' 或 '3D'

    def __post_init__(self):
        # 归一化：总是让小的在前
        if self.qubit1 > self.qubit2:
            self.qubit1, self.qubit2 = self.qubit2, self.qubit1

    def __hash__(self):
        return hash((self.qubit1, self.qubit2))

    def __eq__(self, other):
        return (self.qubit1, self.qubit2) == (other.qubit1, other.qubit2)


class HardwareTopology:
    """硬件拓扑基类"""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.connections: Set[Connection] = set()

    def add_connection(self, q1: int, q2: int, conn_type: str = '2D'):
        """添加连接"""
        if q1 >= self.num_qubits or q2 >= self.num_qubits:
            raise ValueError(f"量子比特索引超出范围: {q1}, {q2}")
        self.connections.add(Connection(q1, q2, conn_type))

    def has_connection(self, q1: int, q2: int) -> bool:
        """判断两个量子比特是否直接相连"""
        conn = Connection(q1, q2, '2D')  # type不影响相等性判断
        return conn in self.connections

    def get_neighbors(self, qubit: int) -> List[int]:
        """获取某个量子比特的邻居"""
        neighbors = []
        for conn in self.connections:
            if conn.qubit1 == qubit:
                neighbors.append(conn.qubit2)
            elif conn.qubit2 == qubit:
                neighbors.append(conn.qubit1)
        return neighbors

    def get_distance(self, q1: int, q2: int) -> Optional[int]:
        """计算两个量子比特之间的最短路径长度（跳数）"""
        if q1 == q2:
            return 0

        # BFS
        from collections import deque
        queue = deque([(q1, 0)])
        visited = {q1}

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.get_neighbors(current):
                if neighbor == q2:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return None  # 不连通

    def count_connections_by_type(self) -> dict:
        """统计不同类型连接的数量"""
        counts = {'2D': 0, '3D': 0}
        for conn in self.connections:
            counts[conn.connection_type] += 1
        return counts

    def __repr__(self) -> str:
        counts = self.count_connections_by_type()
        return f"{self.__class__.__name__}(qubits={self.num_qubits}, 2D={counts['2D']}, 3D={counts['3D']})"


class GridTopology(HardwareTopology):
    """2D网格拓扑"""

    def __init__(self, rows: int, cols: int):
        """
        创建rows×cols的2D网格

        量子比特编号：
        0  1  2  3
        4  5  6  7
        8  9 10 11
        """
        super().__init__(rows * cols)
        self.rows = rows
        self.cols = cols
        self._build_grid()

    def _build_grid(self):
        """构建网格连接"""
        for i in range(self.rows):
            for j in range(self.cols):
                current = i * self.cols + j

                # 右连接
                if j < self.cols - 1:
                    right = i * self.cols + (j + 1)
                    self.add_connection(current, right, '2D')

                # 下连接
                if i < self.rows - 1:
                    down = (i + 1) * self.cols + j
                    self.add_connection(current, down, '2D')

    def get_position(self, qubit: int) -> Tuple[int, int]:
        """获取量子比特的(行, 列)坐标"""
        return (qubit // self.cols, qubit % self.cols)

    def get_qubit_at(self, row: int, col: int) -> int:
        """获取指定位置的量子比特编号"""
        return row * self.cols + col

    def manhattan_distance(self, q1: int, q2: int) -> int:
        """计算曼哈顿距离"""
        r1, c1 = self.get_position(q1)
        r2, c2 = self.get_position(q2)
        return abs(r1 - r2) + abs(c1 - c2)


class FoldedTopology(HardwareTopology):
    """带跨层连接的折叠拓扑"""

    def __init__(self, base_topology: HardwareTopology):
        """
        基于已有拓扑添加跨层连接

        参数:
            base_topology: 基础拓扑（通常是GridTopology）
        """
        super().__init__(base_topology.num_qubits)

        # 复制基础连接
        for conn in base_topology.connections:
            self.connections.add(conn)

        self.base_topology = base_topology

    def add_cross_layer_connection(self, q1: int, q2: int):
        """添加跨层连接"""
        self.add_connection(q1, q2, '3D')

    def get_cross_layer_budget_used(self) -> int:
        """获取已使用的跨层连接数"""
        return self.count_connections_by_type()['3D']


def add_cross_layer_edges(
    topology: HardwareTopology,
    edge_list: List[Tuple[int, int]]
) -> FoldedTopology:
    """
    为拓扑添加跨层连接

    参数:
        topology: 基础拓扑
        edge_list: 跨层连接列表 [(q1, q2), ...]

    返回:
        带跨层连接的新拓扑
    """
    folded = FoldedTopology(topology)
    for q1, q2 in edge_list:
        folded.add_cross_layer_connection(q1, q2)
    return folded


def select_diagonal_cross_layer_edges(
    grid: GridTopology,
    budget: int
) -> List[Tuple[int, int]]:
    """
    为网格拓扑选择对角线跨层连接

    策略：连接对角位置的量子比特（距离最远）

    参数:
        grid: 网格拓扑
        budget: 跨层连接预算

    返回:
        跨层连接列表
    """
    edges = []
    rows, cols = grid.rows, grid.cols

    # 主对角线
    for i in range(min(rows, cols)):
        q1 = grid.get_qubit_at(i, i)
        q2 = grid.get_qubit_at(rows - 1 - i, cols - 1 - i)
        if q1 != q2:
            edges.append((q1, q2))
            if len(edges) >= budget:
                return edges

    # 副对角线
    for i in range(min(rows, cols)):
        q1 = grid.get_qubit_at(i, cols - 1 - i)
        q2 = grid.get_qubit_at(rows - 1 - i, i)
        if q1 != q2 and (q1, q2) not in edges and (q2, q1) not in edges:
            edges.append((q1, q2))
            if len(edges) >= budget:
                return edges

    return edges


def test_topology():
    """测试拓扑构建"""
    print("=== 测试 3×3 网格拓扑 ===")
    grid = GridTopology(3, 3)
    print(grid)
    print(f"0-8的最短路径: {grid.get_distance(0, 8)} 跳")
    print(f"0-8的曼哈顿距离: {grid.manhattan_distance(0, 8)}")
    print(f"0的邻居: {grid.get_neighbors(0)}")
    print(f"4的邻居: {grid.get_neighbors(4)}")

    print("\n=== 添加跨层连接 ===")
    cross_layer = [(0, 8), (2, 6)]
    folded = add_cross_layer_edges(grid, cross_layer)
    print(folded)
    print(f"添加跨层后，0-8的最短路径: {folded.get_distance(0, 8)} 跳")

    print("\n=== 自动选择对角线跨层连接 ===")
    diagonal_edges = select_diagonal_cross_layer_edges(grid, budget=2)
    print(f"选择的跨层连接: {diagonal_edges}")

    folded_diag = add_cross_layer_edges(grid, diagonal_edges)
    print(folded_diag)


if __name__ == "__main__":
    test_topology()
