"""
二部图分析工具

专门针对量子纠错码交互图的二部图特性设计的分析指标
"""

from typing import Tuple, List, Dict, Set
import networkx as nx
import numpy as np


def analyze_bipartite_structure(
    data_qubits: Set[int],
    ancilla_qubits: Set[int],
    interactions: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    分析二部图的结构特性

    参数:
        data_qubits: 数据量子比特集合
        ancilla_qubits: 辅助量子比特集合
        interactions: 交互对列表 [(辅助, 数据), ...]

    返回:
        结构指标字典
    """
    # 构建二部图
    G = nx.Graph()
    G.add_nodes_from(data_qubits, bipartite=0)
    G.add_nodes_from(ancilla_qubits, bipartite=1)
    G.add_edges_from(interactions)

    # 验证是否为二部图
    is_bipartite = nx.is_bipartite(G)

    # 计算每个辅助比特的度（连接的数据比特数）
    ancilla_degrees = [G.degree(a) for a in ancilla_qubits if a in G]
    avg_ancilla_degree = np.mean(ancilla_degrees) if ancilla_degrees else 0
    std_ancilla_degree = np.std(ancilla_degrees) if ancilla_degrees else 0

    # 计算每个数据比特的度（被多少辅助比特访问）
    data_degrees = [G.degree(d) for d in data_qubits if d in G]
    avg_data_degree = np.mean(data_degrees) if data_degrees else 0
    std_data_degree = np.std(data_degrees) if data_degrees else 0

    # 计算交互的空间局部性（假设比特编号反映空间位置）
    if interactions:
        distances = [abs(a - d) for a, d in interactions]
        avg_interaction_span = np.mean(distances)
        max_interaction_span = np.max(distances)
    else:
        avg_interaction_span = 0
        max_interaction_span = 0

    return {
        'is_bipartite': is_bipartite,
        'num_data_qubits': len(data_qubits),
        'num_ancilla_qubits': len(ancilla_qubits),
        'avg_ancilla_degree': avg_ancilla_degree,
        'std_ancilla_degree': std_ancilla_degree,
        'avg_data_degree': avg_data_degree,
        'std_data_degree': std_data_degree,
        'avg_interaction_span': avg_interaction_span,
        'max_interaction_span': max_interaction_span,
    }


def compute_bipartite_layout_cost(
    data_qubits: Set[int],
    ancilla_qubits: Set[int],
    interactions: List[Tuple[int, int]],
    layout_data: Dict[int, Tuple[int, int]],
    layout_ancilla: Dict[int, Tuple[int, int]]
) -> Dict[str, float]:
    """
    计算二部图在给定布局下的成本

    参数:
        data_qubits: 数据量子比特
        ancilla_qubits: 辅助量子比特
        interactions: 交互列表
        layout_data: 数据比特的物理位置 {比特id: (x, y)}
        layout_ancilla: 辅助比特的物理位置 {比特id: (x, y)}

    返回:
        成本指标
    """
    if not interactions:
        return {
            'total_manhattan_distance': 0,
            'avg_manhattan_distance': 0,
            'max_manhattan_distance': 0,
        }

    distances = []
    for ancilla, data in interactions:
        if ancilla not in layout_ancilla or data not in layout_data:
            continue

        ax, ay = layout_ancilla[ancilla]
        dx, dy = layout_data[data]

        manhattan = abs(ax - dx) + abs(ay - dy)
        distances.append(manhattan)

    if not distances:
        return {
            'total_manhattan_distance': 0,
            'avg_manhattan_distance': 0,
            'max_manhattan_distance': 0,
        }

    return {
        'total_manhattan_distance': sum(distances),
        'avg_manhattan_distance': np.mean(distances),
        'max_manhattan_distance': max(distances),
    }


def suggest_folding_targets(
    data_qubits: Set[int],
    ancilla_qubits: Set[int],
    interactions: List[Tuple[int, int]],
    layout_data: Dict[int, Tuple[int, int]],
    layout_ancilla: Dict[int, Tuple[int, int]],
    top_k: int = 5
) -> List[Tuple[Tuple[int, int], float]]:
    """
    建议最值得添加跨层连接的量子比特对

    策略：找距离最远但交互频繁的对

    参数:
        data_qubits: 数据量子比特
        ancilla_qubits: 辅助量子比特
        interactions: 交互列表
        layout_data: 数据比特布局
        layout_ancilla: 辅助比特布局
        top_k: 返回前k个建议

    返回:
        [(辅助比特, 数据比特), 收益分数]
    """
    from collections import Counter

    # 统计交互频率
    freq = Counter(interactions)

    # 计算每对的收益分数：频率 × 距离
    scores = []
    for (ancilla, data), count in freq.items():
        if ancilla not in layout_ancilla or data not in layout_data:
            continue

        ax, ay = layout_ancilla[ancilla]
        dx, dy = layout_data[data]
        manhattan = abs(ax - dx) + abs(ay - dy)

        # 收益分数：频率 × (距离 - 1)，因为距离=1不需要SWAP
        benefit = count * max(0, manhattan - 1)
        scores.append(((ancilla, data), benefit))

    # 按收益排序
    scores.sort(key=lambda x: -x[1])

    return scores[:top_k]


def test_bipartite_analysis():
    """测试二部图分析"""
    print("=== 测试二部图分析 ===\n")

    from foldq.core import SurfaceCode

    code = SurfaceCode(distance=3)

    # 识别数据比特和辅助比特
    data_qubits = set(range(9))  # 0-8
    ancilla_qubits = set(range(9, code.num_physical))  # 9-16

    interactions = code.get_all_interactions()

    # 分析二部图结构
    bipartite_stats = analyze_bipartite_structure(
        data_qubits, ancilla_qubits, interactions
    )

    print("二部图结构分析:")
    print(f"  是否为二部图: {bipartite_stats['is_bipartite']}")
    print(f"  数据比特数: {bipartite_stats['num_data_qubits']}")
    print(f"  辅助比特数: {bipartite_stats['num_ancilla_qubits']}")
    print(f"  辅助比特平均度: {bipartite_stats['avg_ancilla_degree']:.2f}")
    print(f"  数据比特平均度: {bipartite_stats['avg_data_degree']:.2f}")
    print(f"  平均交互跨度: {bipartite_stats['avg_interaction_span']:.2f}")

    # 简单布局（数据比特在左，辅助比特在右）
    layout_data = {i: (i // 3, i % 3) for i in range(9)}
    layout_ancilla = {i: (3 + (i-9)//3, (i-9) % 3) for i in range(9, 17)}

    print(f"\n数据比特布局示例: {list(layout_data.items())[:3]}")
    print(f"辅助比特布局示例: {list(layout_ancilla.items())[:3]}")

    # 计算布局成本
    cost = compute_bipartite_layout_cost(
        data_qubits, ancilla_qubits, interactions,
        layout_data, layout_ancilla
    )

    print(f"\n布局成本:")
    print(f"  总曼哈顿距离: {cost['total_manhattan_distance']}")
    print(f"  平均距离: {cost['avg_manhattan_distance']:.2f}")
    print(f"  最大距离: {cost['max_manhattan_distance']}")

    # 建议跨层连接
    suggestions = suggest_folding_targets(
        data_qubits, ancilla_qubits, interactions,
        layout_data, layout_ancilla, top_k=5
    )

    print(f"\n建议添加的跨层连接（前5个）:")
    for (ancilla, data), benefit in suggestions:
        print(f"  连接 {ancilla}-{data}, 收益分数: {benefit:.0f}")


if __name__ == "__main__":
    test_bipartite_analysis()
