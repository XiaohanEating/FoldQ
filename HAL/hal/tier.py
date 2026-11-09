from __future__ import annotations

import ast
import copy
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from qldpc.objects import Node


class Tier:
    @property
    def avg_route_length(self) -> float:
        if self.G:
            tot_length = 0
            if self.switch_edges:
                for edge in self.switch_edges:
                    if len(self.switch_edges[edge]) == 0:
                        tot_length += len(self.G.edges[edge]["route"])
                    else:
                        for subedge in self.switch_edges[edge]:
                            tot_length += len(self.G.edges[subedge]["route"])
            else:
                for edge in self.G.edges:
                    tot_length += len(self.G.edges[edge]["route"])

            if self.switch_edges:
                return tot_length / len(self.switch_edges)

            return tot_length / len(self.G.edges)

        return self.cached_avg_route_length

    @property
    def max_face_switches(self) -> int:
        if self.switch_edges:
            return max(len(self.switch_edges[sw_edge]) for sw_edge in self.switch_edges.keys())
        return self.cached_max_face_switches

    @property
    def avg_face_switches(self) -> float:
        if self.switch_edges:
            return sum(
                len(self.switch_edges[sw_edge]) for sw_edge in self.switch_edges.keys()
            ) / len(self.switch_edges)
        return self.cached_avg_face_switches

    @property
    def num_edges(self) -> int:
        if self.switch_edges:
            return len(self.switch_edges)
        return len(self.G.edges)

    @property
    def edge_density(self) -> float:
        occupied_area = 0
        min_x = None
        min_y = None
        max_x = None
        max_y = None

        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                # Using the bottom face (index 0) as before.
                if cell[0] != 0:
                    occupied_area += 1
                    min_x = i if min_x is None else min(i, min_x)
                    min_y = j if min_y is None else min(j, min_y)
                    max_x = i if max_x is None else max(i, max_x)
                    max_y = j if max_y is None else max(j, max_y)

        bounding_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        density = occupied_area / bounding_area if bounding_area > 0 else 0
        return density

    @property
    def routes(self) -> List[np.ndarray]:
        return [self.G.edges[edge]["route"] for edge in self.G.edges]

    def get_combined_routes(self) -> List[np.ndarray]:
        # construct switch edges dictionary and add switch-free routes
        all_routes = []
        switch_routes_dict = {}
        for edge in self.G.edges:
            switch_key = [-1, -1]
            for i, node in enumerate(edge):
                if isinstance(node, tuple) and node[0] == "switch":
                    switch_key[i] = node[1]
            switch_key = tuple(switch_key)

            if all(v == -1 for v in switch_key):
                # No switch nodes, return the route directly
                all_routes.append(self.G.edges[edge]["route"])
            else:
                switch_routes_dict[switch_key] = edge

        # construct full routes from switch_routes_dict
        switch_routes_dict_copy = deepcopy(switch_routes_dict)
        for key, edge in switch_routes_dict.items():
            if key not in switch_routes_dict_copy:
                continue

            full_route = []
            if sum(v == -1 for v in key) == 1:
                route = (
                    self.G.edges[edge]["route"]
                    if key.index(-1) == 0
                    else self.G.edges[edge]["route"][::-1]
                )
                full_route.extend(route)
                switch_routes_dict_copy.pop(key)

                switch_node = key[1 - key.index(-1)]
                route_key = [k for k in switch_routes_dict_copy if switch_node in k][0]
                while all(v != -1 for v in route_key):
                    route = (
                        self.G.edges[edge]["route"]
                        if route_key.index(switch_node) == 0
                        else self.G.edges[edge]["route"][::-1]
                    )
                    full_route.extend(route)
                    switch_routes_dict_copy.pop(route_key)

                    switch_node = route_key[1 - route_key.index(switch_node)]
                    route_key = [k for k in switch_routes_dict_copy if switch_node in k][0]

                route = (
                    self.G.edges[edge]["route"]
                    if route_key.index(switch_node) == 0
                    else self.G.edges[edge]["route"][::-1]
                )
                full_route.extend(route)
                switch_routes_dict_copy.pop(route_key)

                all_routes.append(full_route)

        return all_routes

    def calculate_avg_face_switches(self, routes) -> float:
        """
        Calculate the average number of face switches per edge in the tier.

        Returns:
            float: Average number of face switches per edge.
        """

        num_face_switches_list = []
        routes = routes if routes is not None else self.get_combined_routes()
        for i, r in enumerate(routes):
            num_face_switches = sum(1 for i in range(len(r) - 1) if r[i][2] != r[i + 1][2])
            print(f"Edge {i} has {num_face_switches} face switches.")
            num_face_switches_list.append(num_face_switches)

        return (
            sum(num_face_switches_list) / len(num_face_switches_list)
            if num_face_switches_list
            else 0.0
        )

    def __init__(
        self, G: nx.Graph, grid_size_x: int, grid_size_y: int, node_expansion_val: int
    ) -> None:
        self.G = G

        grid = np.zeros((grid_size_x, grid_size_y, 2))
        expanded_grid = np.zeros((grid_size_x, grid_size_y, 2))
        for node in self.G.nodes:
            # Map node position to grid indices
            grid_ind = self.G.nodes[node]["pos"]
            x_idx, y_idx = int(grid_ind[0]), int(grid_ind[1])
            grid[x_idx, y_idx, 0] = 10
            # Expand node footprint on the expanded grid
            x_low = max(x_idx - node_expansion_val, 0)
            x_high = min(x_idx + node_expansion_val, grid.shape[0] - 1)
            y_low = max(y_idx - node_expansion_val, 0)
            y_high = min(y_idx + node_expansion_val, grid.shape[1] - 1)
            expanded_grid[x_low : x_high + 1, y_low : y_high + 1, 0] = 10

        self.grid = grid
        self.expanded_grid = expanded_grid

        self.switch_nodes = None
        self.switch_nodes_pos = None
        self.switch_edges = None
        self.switch_edges_route = None
        self.switch_num = None
        self.edges_attempted = None
        self.routed_edge_order = None
        self.tot_route_length = None
        self.max_routed_face_switches = None

        self.cached_avg_route_length = None
        self.cached_max_face_switches = None
        self.cached_avg_face_switches = None

    @classmethod
    def from_database(cls, G, grid, expanded_grid, metrics_path: Optional[str] = None) -> "Tier":
        tier = cls(G, grid.shape[0], grid.shape[1], 1)
        tier.grid = grid
        tier.expanded_grid = expanded_grid
        if metrics_path is not None:

            class GF:
                def __init__(self, value, order):
                    self.value = value
                    self.order = order

            metrics = pd.read_csv(metrics_path)
            switch_val = metrics["switch_edges"].iloc[0] if "switch_edges" in metrics else None
            tier.switch_edges = (
                ast.literal_eval(switch_val) if isinstance(switch_val, str) and switch_val else None
            )
            reo_val = (
                metrics["routed_edge_order"].iloc[0] if "routed_edge_order" in metrics else None
            )
            tier.routed_edge_order = (
                ast.literal_eval(reo_val) if isinstance(reo_val, str) and reo_val else None
            )

            tier.cached_avg_route_length = metrics["avg_route_length"].iloc[0]
            tier.cached_max_face_switches = metrics["max_face_switches"].iloc[0]
            tier.cached_avg_face_switches = metrics["avg_face_switches"].iloc[0]
        else:
            tier.switch_edges = None
            tier.routed_edge_order = None

            tier.cached_avg_route_length = None
            tier.cached_max_face_switches = None
            tier.cached_avg_face_switches = None
        return tier

    def compute_avg_route_length(self) -> float:
        total_route_length = 0
        for edge in self.G.edges:
            route = self.G.edges[edge]["route"]
            total_route_length += len(route)
        return total_route_length / self.num_edges

    def save_metrics(self, filename: str, path: Path) -> None:
        """
        Save the metrics of the tier to a CSV file.

        Args:
            Tier: The tier object containing metrics.
            filename (str): The name of the file to save metrics.
            path (Path): The directory path where the file will be saved.
        """
        metrics = {
            "avg_route_length": self.avg_route_length,
            "max_face_switches": self.max_face_switches,
            "avg_face_switches": self.avg_face_switches,
            "num_edges": self.num_edges,
            "edge_density": self.edge_density,
            "routed_edge_order": self.routed_edge_order,
            "switch_edges": self.switch_edges,
        }
        df = pd.DataFrame([metrics])
        df.to_csv(path / f"{filename}.csv", index=False)
