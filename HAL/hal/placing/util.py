from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from math import ceil
from typing import Dict, Iterable, Tuple

import networkx as nx
import numpy as np


def robust_unique_grid_positions(
    pos: Dict[object, Tuple[float, float]],
) -> Dict[object, Tuple[int, int]]:
    """
    Two-phase approach: place non-conflicting nodes first,
    then systematically place remaining nodes at closest available positions
    """
    # Phase 1: Round positions and identify conflicts
    rounded_pos = {}
    for node, (x, y) in pos.items():
        rounded_pos[node] = (int(round(x)), int(round(y)))

    # Group by position to find conflicts
    position_groups = defaultdict(list)
    for node, position in rounded_pos.items():
        position_groups[position].append(node)

    # Phase 2: Place non-conflicting nodes first
    occupied = set()
    final_positions = {}
    conflicted_nodes = []

    for position, nodes in position_groups.items():
        if len(nodes) == 1:
            # No conflict - place immediately
            final_positions[nodes[0]] = position
            occupied.add(position)
        else:
            # Conflict - add all nodes to conflicted list with their preferred position
            for node in nodes:
                conflicted_nodes.append((node, position))

    # Phase 3: Systematically place conflicted nodes
    # Create priority queue: (distance_to_closest_free, node, preferred_position)
    placement_queue = []

    for node, preferred_pos in conflicted_nodes:
        closest_free, distance = find_closest_free_position_with_distance(preferred_pos, occupied)
        heapq.heappush(placement_queue, (distance, node, preferred_pos, closest_free))

    # Place nodes in order of how close they can get to their preferred position
    while placement_queue:
        distance, node, preferred_pos, closest_free = heapq.heappop(placement_queue)

        if closest_free not in occupied:
            # Position is still available
            final_positions[node] = closest_free
            occupied.add(closest_free)
        else:
            # Position was taken by another node, recalculate
            new_closest, new_distance = find_closest_free_position_with_distance(
                preferred_pos, occupied
            )
            heapq.heappush(placement_queue, (new_distance, node, preferred_pos, new_closest))

    return final_positions


def find_closest_free_position_with_distance(
    target_pos: Tuple[int, int], occupied_positions: Iterable[Tuple[int, int]]
) -> Tuple[Tuple[int, int], float]:
    """
    Find closest free position and return both position and distance
    """
    target_x, target_y = target_pos

    # Search in expanding squares (more systematic than Manhattan rings)
    for radius in range(0, 1000):  # Start from 0 to check target position first
        if radius == 0:
            # Check the target position itself
            if target_pos not in occupied_positions:
                return target_pos, 0
        else:
            # Check all positions in the current radius ring
            candidates = []

            # Generate all positions at this radius (square boundary)
            for dx in range(-radius, radius + 1):
                # Top and bottom edges of square
                for dy in [-radius, radius]:
                    candidate = (target_x + dx, target_y + dy)
                    if candidate not in occupied_positions:
                        candidates.append(candidate)

            # Left and right edges (excluding corners already covered)
            for dy in range(-radius + 1, radius):
                for dx in [-radius, radius]:
                    candidate = (target_x + dx, target_y + dy)
                    if candidate not in occupied_positions:
                        candidates.append(candidate)

            if candidates:
                # Return the candidate with minimum Euclidean distance
                best_candidate = min(
                    candidates,
                    key=lambda pos: (pos[0] - target_x) ** 2 + (pos[1] - target_y) ** 2,
                )
                euclidean_dist = np.sqrt(
                    (best_candidate[0] - target_x) ** 2 + (best_candidate[1] - target_y) ** 2
                )
                return best_candidate, euclidean_dist

    raise RuntimeError("Could not find free position")


def get_systematic_unique_layout(graph, grid_size: int | None = None, verbose: bool = True):
    """
    Complete workflow with systematic conflict resolution
    """
    if grid_size is None:
        grid_size = np.round(np.sqrt(len(graph.nodes)), -1)
        print(f"Auto-calculated grid_size: {grid_size}")

    # Get initial layout
    pos = nx.kamada_kawai_layout(graph, scale=grid_size)

    # Apply robust positioning
    unique_pos = robust_unique_grid_positions(pos)

    # Verification
    positions = list(unique_pos.values())
    unique_positions = set(positions)

    if verbose:
        print(f"✓ Total nodes: {len(positions)}")
        print(f"✓ Unique positions: {len(unique_positions)}")
        print(f"✓ Success: {len(positions) == len(unique_positions)}")

    # Additional verification - check no overlaps
    assert len(positions) == len(unique_positions), "FAILED: Overlapping nodes detected!"

    return unique_pos


def rasterise(
    positions: dict[int | str | tuple, tuple[int, int]], step: int = 1, margin: int = 0
) -> dict[int | str | tuple, tuple[int, int]]:
    """
    Compress the drawing onto the smallest rectangular grid.

    Parameters
    ----------
    positions : dict node -> (x, y)   (output of `solve`)
    step      : distance between two consecutive grid lines in the
                final picture (default = 1)
    margin    : free lattice lines added on every side (default = 0)

    Returns
    -------
    new_positions : dict node -> (x', y')
    """
    xs = sorted({x for x, _ in positions.values()})
    ys = sorted({y for _, y in positions.values()})

    map_x = {x: margin + i * step for i, x in enumerate(xs)}
    map_y = {y: margin + i * step for i, y in enumerate(ys)}

    return {v: (map_x[x], map_y[y]) for v, (x, y) in positions.items()}
