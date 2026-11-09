import heapq
from typing import List, Tuple, Union

import numpy as np


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean distance between 2D points a and b."""
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def astar(
    array: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> Union[bool, list[tuple[int, int]]]:
    """A* on 2D occupancy grid with 4-neighborhood.

    Args:
        array (np.ndarray): 2D grid of cell states.
        start (tuple[int,int]): Start cell (x, y).
        goal (tuple[int,int]): Goal cell (x, y).

    Returns:
        list[tuple[int,int]] | bool: Path as list of cells (excluding start), or False if none.
    """

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:

        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 2:
                        continue
                    if array[neighbor[0]][neighbor[1]] == 10 and neighbor != goal:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False


def heuristic_3d(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Manhattan distance in 3D."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def heuristic_2d(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Manhattan distance in XY plane."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_3d_neighbor(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Neighbor step cost in 3D using Manhattan difference."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def manhattan_heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    """
    Computes the Manhattan distance between two points a and b in 3D.
    This is the sum of the absolute differences of their coordinates.

    Since only axis-aligned moves are allowed, each move changes
    one coordinate by 1 unit. Therefore, the Manhattan distance is
    an admissible and consistent heuristic.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


class AStar3DRouter:
    def __init__(self, grid: np.ndarray):
        """
        Initialize the router with a 3D grid.

        Parameters:
            grid (np.ndarray): A 3D NumPy array where obstacles (and
                previously routed paths) are marked (e.g., with 2 or 4).
        """
        self.grid = grid
        self.max_x, self.max_y, self.max_z = grid.shape

        # Predefine allowed moves.
        # You can adjust these if you want to allow additional or fewer moves.
        self.neighbors = [
            (0, 0, -1),
            (0, 1, 0),
            (0, -1, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
            (0, 0, 1),
        ]

    def route(
        self, start: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) -> Union[bool, List[Tuple[int, int, int]]]:
        """
        Compute and return an optimal path from start to goal using the A* algorithm.

        Parameters:
            start (Tuple[int, int, int]): The starting position.
            goal (Tuple[int, int, int]): The target position.

        Returns:
            A list of (x, y, z) tuples representing the path from start to goal if one is found,
            or False if no path exists.
        """
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic_3d(start, goal)}

        # The open list is maintained as a heap.
        # Each element is a tuple: (fscore, counter, node)
        open_heap = []
        counter = 0  # Tie-breaker to ensure deterministic ordering
        heapq.heappush(open_heap, (fscore[start], counter, start))

        while open_heap:
            current_f, _, current = heapq.heappop(open_heap)

            # If we reached the goal, reconstruct the path.
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # Skip nodes that have already been processed.
            if current in close_set:
                continue
            close_set.add(current)

            # Explore neighbors.
            for dx, dy, dz in self.neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                # Check grid boundaries.
                if not (
                    0 <= neighbor[0] < self.max_x
                    and 0 <= neighbor[1] < self.max_y
                    and 0 <= neighbor[2] < self.max_z
                ):
                    continue

                # Skip if the neighbor is an obstacle.
                cell_val = self.grid[neighbor]
                if cell_val in (2, 4):
                    continue

                dist_goal = heuristic_3d(neighbor, goal)
                if dist_goal > 5 and abs(fscore[start] - dist_goal) > 5 and cell_val == 10:
                    continue

                # Compute tentative cost to move to this neighbor.
                tentative_g = gscore[current] + heuristic_3d_neighbor(current, neighbor)
                if tentative_g >= gscore.get(neighbor, float("inf")):
                    continue  # This is not a better path.

                # Record the best path so far.
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + dist_goal
                counter += 1
                heapq.heappush(open_heap, (fscore[neighbor], counter, neighbor))

        # No path was found.
        return False


class AStar2DRouter:
    def __init__(self, grid: np.ndarray, node_expansion_val: int, edge_expansion_val: int):
        """
        Initialize the router with a 3D grid.

        Parameters:
            grid (np.ndarray): A 3D NumPy array where obstacles (and
                previously routed paths) are marked (e.g., with 2 or 4).
        """
        self.grid = grid
        self.max_x, self.max_y, self.max_z = grid.shape
        self.node_expansion_val = node_expansion_val
        self.edge_expansion_val = edge_expansion_val

        # Predefine allowed moves.
        # You can adjust these if you want to allow additional or fewer moves.
        self.neighbors = [
            (0, 1, 0),
            (0, -1, 0),
            (1, 0, 0),
            (-1, 0, 0),
            # (1,  1, 0), (1, -1, 0), (-1, 1, 0), (-1,-1, 0),
        ]

    def route(
        self, start: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) -> Union[bool, List[Tuple[int, int, int]]]:
        """
        Compute and return an optimal path from start to goal using the A* algorithm.

        Parameters:
            start (Tuple[int, int, int]): The starting position.
            goal (Tuple[int, int, int]): The target position.

        Returns:
            A list of (x, y, z) tuples representing the path from start to goal if one is found,
            or False if no path exists.
        """
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic_2d(start, goal)}

        # The open list is maintained as a heap.
        # Each element is a tuple: (fscore, counter, node)
        open_heap = []
        counter = 0  # Tie-breaker to ensure deterministic ordering
        heapq.heappush(open_heap, (fscore[start], counter, start))

        while open_heap:
            current_f, _, current = heapq.heappop(open_heap)

            # If we reached the goal, reconstruct the path.
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # Skip nodes that have already been processed.
            if current in close_set:
                continue
            close_set.add(current)

            # Explore neighbors.
            for dx, dy, dz in self.neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                # Check grid boundaries.
                if not (0 <= neighbor[0] < self.max_x and 0 <= neighbor[1] < self.max_y):
                    continue

                # Skip if the neighbor is an obstacle.
                # FoldQ 2026-11-06 修复（复核 P0）：原实现仅在"路径中段窗口"
                # （node_expansion_val+1 < dist_start 且 > dist_goal）内把占用格
                # （2=已布线，10=器件）视为障碍，起点/终点邻域全部豁免，导致
                # 不同边的路径在端点邻域相交（GT_98 tier1 实测 14 处）。
                # 改为：占用格始终是障碍，仅豁免 start/goal 自身（器件端点）。
                cell_val = self.grid[neighbor]

                if cell_val in (2, 4, 10) and neighbor not in (start, goal):
                    continue  # FoldQ 2026-11-06: 4 是 face1 占用，原检查漏检

                # Compute tentative cost to move to this neighbor.
                tentative_g = gscore[current] + heuristic_2d(current, neighbor)
                if tentative_g >= gscore.get(neighbor, float("inf")):
                    continue  # This is not a better path.

                # Record the best path so far.
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + dist_goal
                counter += 1
                heapq.heappush(open_heap, (fscore[neighbor], counter, neighbor))

        # No path was found.
        return False
