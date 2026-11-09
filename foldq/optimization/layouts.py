"""
布局基线：比特 -> (行, 列)

native_layout：码自带的最近邻布局（表面码有，qLDPC 没有）
spring_layout：按交互图拓扑做力导向布局再离散化到方格，任何码都适用。
    实现与实验 08 的 spring_layout_baseline 相同，seed 可变，用于真正的重复实验。
"""

import math
from typing import Dict, Tuple

import networkx as nx


def native_layout(code) -> Tuple[Dict[int, Tuple[int, int]], int]:
    return code.native_positions(), code.native_grid_size()


def default_grid_size(code) -> int:
    return math.ceil(math.sqrt(code.num_physical)) + 1


def spring_layout(code, size: int, seed: int) -> Dict[int, Tuple[int, int]]:
    G = nx.Graph()
    G.add_nodes_from(range(code.num_physical))
    G.add_edges_from(code.get_all_interactions())
    xy = nx.spring_layout(G, k=1.5, iterations=100, seed=seed)

    occupied = set()
    positions = {}

    def to_cell(x, y):
        r = int((y + 1) / 2 * (size - 1))
        c = int((x + 1) / 2 * (size - 1))
        return min(max(r, 0), size - 1), min(max(c, 0), size - 1)

    data = range(code.num_data)
    ancillas = range(code.num_data, code.num_physical)
    for group in (data, ancillas):
        for q in group:
            base = to_cell(*xy[q])
            cell = base
            radius = 0
            while cell in occupied:
                radius += 1
                ring = [(base[0] + dr, base[1] + dc)
                        for dr in range(-radius, radius + 1)
                        for dc in range(-radius, radius + 1)
                        if abs(dr) + abs(dc) == radius]
                free = [p for p in ring
                        if 0 <= p[0] < size and 0 <= p[1] < size and p not in occupied]
                if free:
                    cell = free[0]
            positions[q] = cell
            occupied.add(cell)
    return positions
