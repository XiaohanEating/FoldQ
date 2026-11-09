"""
FoldQ分析工具：交互图分析

功能：
- 从纠错码构建量子比特交互图
- 统计交互频率
- 识别高频交互对
- 分析图的拓扑特性（聚类、社区结构）
"""

from typing import Dict, List, Tuple, Set
import networkx as nx
import numpy as np
from dataclasses import dataclass

from foldq.core.quantum_code import QuantumCode
from foldq.core.hardware import HardwareTopology


@dataclass
class InteractionGraphAnalysis:
    """交互图分析结果"""

    graph: nx.Graph  # 交互图
    total_interactions: int  # 总交互次数
    high_freq_pairs: List[Tuple[int, int]]  # 高频交互对
    frequency_map: Dict[Tuple[int, int], int]  # 交互频率映射

    # 图拓扑特性
    clustering_coefficient: float  # 聚类系数
    average_degree: float  # 平均度
    communities: List[Set[int]]  # 社区划分

    def get_top_k_interactions(self, k: int = 10) -> List[Tuple[Tuple[int, int], int]]:
        """获取频率最高的k对交互"""
        sorted_pairs = sorted(self.frequency_map.items(), key=lambda x: -x[1])
        return sorted_pairs[:k]


def build_interaction_graph(code: QuantumCode) -> nx.Graph:
    """
    从纠错码构建交互图

    节点：量子比特
    边：有交互的量子比特对
    边权重：交互频率

    参数:
        code: 量子纠错码

    返回:
        交互图（NetworkX Graph）
    """
    G = nx.Graph()

    # 添加所有量子比特作为节点
    G.add_nodes_from(range(code.num_physical))

    # 统计交互频率
    freq = code.get_interaction_frequency()

    # 添加边及权重
    for (q1, q2), count in freq.items():
        G.add_edge(q1, q2, weight=count)

    return G


def analyze_interaction_graph(
    code: QuantumCode,
    high_freq_threshold: float = 0.5
) -> InteractionGraphAnalysis:
    """
    完整分析交互图

    参数:
        code: 量子纠错码
        high_freq_threshold: 高频交互阈值（相对于最大频率）

    返回:
        分析结果
    """
    # 构建图
    G = build_interaction_graph(code)
    freq_map = code.get_interaction_frequency()
    total = sum(freq_map.values())

    # 识别高频交互对
    if freq_map:
        max_freq = max(freq_map.values())
        threshold = max_freq * high_freq_threshold
        high_freq_pairs = [pair for pair, freq in freq_map.items() if freq >= threshold]
    else:
        high_freq_pairs = []

    # 计算图拓扑特性
    clustering = nx.average_clustering(G) if len(G) > 0 else 0.0
    avg_degree = sum(dict(G.degree()).values()) / len(G) if len(G) > 0 else 0.0

    # 社区检测（使用Louvain算法）
    try:
        communities = list(nx.community.louvain_communities(G, seed=42))
    except:
        communities = []

    return InteractionGraphAnalysis(
        graph=G,
        total_interactions=total,
        high_freq_pairs=high_freq_pairs,
        frequency_map=freq_map,
        clustering_coefficient=clustering,
        average_degree=avg_degree,
        communities=communities
    )


def compute_routing_cost(
    code: QuantumCode,
    topology: HardwareTopology,
    mapping: Dict[int, int]  # logical -> physical
) -> Dict[str, float]:
    """
    计算给定映射下的路由成本

    成本指标：
    - 总SWAP数估计（假设最短路径路由）
    - 加权距离（频率 × 距离）
    - 平均交互距离

    参数:
        code: 纠错码
        topology: 硬件拓扑
        mapping: 量子比特映射

    返回:
        成本指标字典
    """
    freq_map = code.get_interaction_frequency()

    total_swaps = 0
    weighted_distance = 0.0
    total_freq = 0

    for (q1_logical, q2_logical), freq in freq_map.items():
        # 映射到物理位置
        q1_physical = mapping.get(q1_logical, q1_logical)
        q2_physical = mapping.get(q2_logical, q2_logical)

        # 计算物理距离
        distance = topology.get_distance(q1_physical, q2_physical)
        if distance is None:
            distance = 999  # 不可达，给一个大值

        # 估算SWAP数：距离-1（相邻不需要SWAP）
        swaps_needed = max(0, distance - 1)

        total_swaps += swaps_needed * freq
        weighted_distance += distance * freq
        total_freq += freq

    avg_distance = weighted_distance / total_freq if total_freq > 0 else 0.0

    return {
        'estimated_swaps': total_swaps,
        'weighted_distance': weighted_distance,
        'average_distance': avg_distance,
        'total_interactions': total_freq
    }


def compare_random_graph(code: QuantumCode) -> Dict[str, float]:
    """
    将交互图与相同规模的随机图对比

    返回:
        对比指标
    """
    G = build_interaction_graph(code)
    n = len(G)
    m = G.number_of_edges()

    if n == 0 or m == 0:
        return {
            'clustering_ratio': 0.0,
            'is_structured': False
        }

    # 生成随机图
    G_random = nx.gnm_random_graph(n, m, seed=42)

    # 计算聚类系数比值
    clustering_real = nx.average_clustering(G)
    clustering_random = nx.average_clustering(G_random)

    clustering_ratio = clustering_real / clustering_random if clustering_random > 0 else 0.0

    # 判断是否有结构（聚类系数明显高于随机图）
    is_structured = clustering_ratio > 1.5

    return {
        'clustering_ratio': clustering_ratio,
        'clustering_real': clustering_real,
        'clustering_random': clustering_random,
        'is_structured': is_structured
    }


def test_interaction_analysis():
    """测试交互图分析"""
    from foldq.core.quantum_code import SurfaceCode

    print("=== 分析 d=5 表面码的交互图 ===\n")
    code = SurfaceCode(distance=5)

    analysis = analyze_interaction_graph(code, high_freq_threshold=0.8)

    print(f"总交互次数: {analysis.total_interactions}")
    print(f"平均度: {analysis.average_degree:.2f}")
    print(f"聚类系数: {analysis.clustering_coefficient:.3f}")
    print(f"社区数量: {len(analysis.communities)}")

    print(f"\n高频交互对数量: {len(analysis.high_freq_pairs)}")
    top_interactions = analysis.get_top_k_interactions(k=10)
    print("\n频率最高的10对交互:")
    for (q1, q2), freq in top_interactions:
        print(f"  ({q1:2d}, {q2:2d}): {freq}次")

    # 与随机图对比
    print("\n=== 与随机图对比 ===")
    comparison = compare_random_graph(code)
    print(f"聚类系数比值: {comparison['clustering_ratio']:.2f}")
    print(f"真实图聚类: {comparison['clustering_real']:.3f}")
    print(f"随机图聚类: {comparison['clustering_random']:.3f}")
    print(f"是否有结构: {'是' if comparison['is_structured'] else '否'}")

    # 测试路由成本
    print("\n=== 路由成本估算 ===")
    from foldq.core.hardware import GridTopology

    # 简单映射：逻辑比特i -> 物理比特i
    simple_mapping = {i: i for i in range(code.num_physical)}

    grid = GridTopology(rows=6, cols=6)  # 6x6网格，36个物理比特
    cost = compute_routing_cost(code, grid, simple_mapping)

    print(f"估计SWAP总数: {cost['estimated_swaps']:.0f}")
    print(f"加权距离: {cost['weighted_distance']:.1f}")
    print(f"平均交互距离: {cost['average_distance']:.2f}跳")


if __name__ == "__main__":
    test_interaction_analysis()
