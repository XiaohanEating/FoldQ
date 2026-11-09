from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional

import networkx as nx
from tqdm import tqdm

from hal.routing.astar import AStar3DRouter
from hal.routing.straightline import straightline, straightline_no_face_switches
from hal.tier import Tier
from hal.utils.graph_utils import get_ordered_edges


class TierBasedRouter:
    """Encapsulate the multi-tier routing logic for `HardwareAwareLayout`.

    Attributes
    ----------
    tiers : list[hal.tier.Tier]
        Working tiers holding graphs and occupancy grids.
    max_bump_transitions_per_coupler : int | None
        Maximum allowed count of face switches on a single coupler.
    max_tsvs_per_coupler : int | None
        Maximum allowed TSVs per coupler; exceeding triggers stop/promotion.
    routing_edge_order : str
        Edge ordering metric determining routing order.
    edge_expansion_val : int
        Expansion radius used when marking routed traces on the grid.
    routing_algorithm : str
        Selected algorithm variant; currently "straightline_astar".
    max_coupler_length : int | None
        Relative length limit with respect to the shortest coupler.
    verbose : bool
        Emit additional logs when True.
    min_length_coupler : int
        Cached shortest coupler length on the base tier.
    """

    def __init__(
        self,
        tiers: List[Tier],
        max_bump_transitions_per_coupler: int,
        max_tsvs_per_coupler: Optional[int],
        routing_edge_order: str,
        edge_expansion_val: int,
        routing_algorithm: str,
        max_coupler_length: Optional[int],
        node_expansion_val: int = 1,
        verbose: bool = False,
    ):
        self.tiers = tiers
        self.max_bump_transitions_per_coupler = max_bump_transitions_per_coupler
        self.max_tsvs_per_coupler = max_tsvs_per_coupler
        self.routing_edge_order = routing_edge_order
        self.edge_expansion_val = edge_expansion_val
        self.routing_algorithm = routing_algorithm
        self.max_coupler_length = max_coupler_length
        self.verbose = verbose
        self.node_expansion_val = node_expansion_val
        self.min_length_coupler = 0

    def route_base_tier(self) -> int:
        """
        Route the base tier using straightline connections.

        Returns:
            int: Total route length for the base tier.
        """
        tot_route_length = 0
        edges_to_remove = []

        edges_to_route = copy.deepcopy(get_ordered_edges(self.tiers[0].G, self.routing_edge_order))
        self.tiers[0].switch_edges = {}

        for edge in tqdm(
            edges_to_route,
            desc="- Route base tier edges using straightline connections",
        ):
            source_pos = self.tiers[0].G.nodes[edge[0]]["pos"]
            target_pos = self.tiers[0].G.nodes[edge[1]]["pos"]
            start = (int(source_pos[0]), int(source_pos[1]), 0)
            goal = (int(target_pos[0]), int(target_pos[1]), 0)

            route = straightline_no_face_switches(self.tiers[0].expanded_grid, start, goal)

            if route is False:
                edges_to_remove.append(edge)
                continue

            self.tiers[0].G.edges[edge[0], edge[1]]["route"] = copy.deepcopy(route)
            tot_route_length += len(route)

            if "full_edge" not in self.tiers[0].G.edges[edge[0], edge[1]]:
                self.tiers[0].G.edges[edge[0], edge[1]]["full_edge"] = (
                    edge[0],
                    edge[1],
                )
                self.tiers[0].switch_edges[(edge[0], edge[1])] = []

            # Mark the grid cells along the route
            self._mark_grid_route(self.tiers[0], route)

        # Move failed edges to higher tier
        for edge in edges_to_remove:
            self.tiers[0].G.remove_edge(edge[0], edge[1])
            self.tiers[1].G.add_edge(edge[0], edge[1])

        # 2026-11-06（FoldQ 复核配套）：tier 0 无待布边（仅 marginal 边入图的
        # 紧凑适配用法）时 routes 为空，min() 崩溃；分母语义取最短已布路径，空则记 1
        routes0 = list(self.tiers[0].routes)
        self.min_length_coupler = min((len(r) for r in routes0), default=1)

        return tot_route_length

    def attempt_higher_tier_on_base(self) -> int:
        """
        Attempt to route higher tier edges on the base tier.

        Returns:
            int: Additional route length from successful routes.
        """
        additional_length = 0
        edges_to_route = copy.deepcopy(get_ordered_edges(self.tiers[1].G, self.routing_edge_order))
        edges_to_remove = []

        for edge in tqdm(edges_to_route, desc="- Attempt to route higher tier edges on base tier"):
            source_pos = self.tiers[0].G.nodes[edge[0]]["pos"]
            target_pos = self.tiers[0].G.nodes[edge[1]]["pos"]
            start = (int(source_pos[0]), int(source_pos[1]), 0)
            goal = (int(target_pos[0]), int(target_pos[1]), 0)

            route = straightline_no_face_switches(self.tiers[0].expanded_grid, start, goal)

            if route is not False:
                if self.verbose:
                    print("added to base tier")
                edges_to_remove.append(edge)
                self.tiers[0].G.add_edge(edge[0], edge[1])
                self.tiers[0].G.edges[edge[0], edge[1]]["route"] = copy.deepcopy(route)
                additional_length += len(route)

                if "full_edge" not in self.tiers[0].G.edges[edge[0], edge[1]]:
                    self.tiers[0].G.edges[edge[0], edge[1]]["full_edge"] = (
                        edge[0],
                        edge[1],
                    )

                self.tiers[0].switch_edges[(edge[0], edge[1])] = []
                self._mark_grid_route(self.tiers[0], route)

        print(
            f"- Prune {len(edges_to_remove)} edges that were successfully routed on base tier from higher tiers"
        )
        for edge in edges_to_remove:
            self.tiers[1].G.remove_edge(edge[0], edge[1])

        self.min_length_coupler = min((len(r) for r in self.tiers[0].routes), default=1)  # FoldQ 2026-11-06: 空 tier 保护

        return additional_length

    def route_higher_tiers(self) -> None:
        """Route remaining edges on higher tiers using the core routing routine."""
        edges_to_route = copy.deepcopy(get_ordered_edges(self.tiers[1].G, self.routing_edge_order))

        if self.verbose:
            print("Edges to route: ", len(self.tiers[1].G.edges))

        progress_bar = tqdm(
            total=len(edges_to_route), desc="- Route remaining edges on higher tiers"
        )
        self._core_routing_routine(edges_to_route, progress_bar)

    def _core_routing_routine(
        self, edges_to_route: List, progress_bar=None, reattempt: bool = False
    ) -> None:
        """
        Core routing routine that handles multi-tier routing with face switches.

        Args:
            edges_to_route: List of edges to route
            progress_bar: Optional progress bar for tracking
            reattempt: Whether this is a reattempt (preserves existing state)
        """
        tier_ind = 1

        # Initialize or restore tier state
        tier_state = self._initialize_tier_state(tier_ind, reattempt)
        router = AStar3DRouter(self.tiers[tier_ind].expanded_grid)

        try:
            while edges_to_route:
                edge = edges_to_route.pop(0)

                # Check if edge was repeatedly attempted - promote to next tier
                if edge in tier_state["edges_attempted"]:
                    if self.max_tsvs_per_coupler is not None:
                        if tier_ind + 1 > self.max_tsvs_per_coupler and len(edges_to_route) > 0:
                            return

                    self._promote_edges_to_next_tier(tier_ind, edges_to_route, edge, tier_state)
                    tier_ind += 1
                    tier_state = self._initialize_tier_state(tier_ind, reattempt)
                    edges_to_route = copy.deepcopy(
                        get_ordered_edges(self.tiers[tier_ind].G, self.routing_edge_order)
                    )
                    router = AStar3DRouter(self.tiers[tier_ind].expanded_grid)
                    continue

                tier_state["edges_attempted"].append(edge)

                # Try to route the edge
                route_result = self._route_single_edge(
                    tier_ind, edge, router, tier_state["switch_num"]
                )

                if route_result is None:
                    # Routing failed, try again later
                    edges_to_route.append(edge)
                    continue

                route, switch_info = route_result

                # Check constraints
                if not self._check_routing_constraints(edge, route, switch_info):
                    edges_to_route.append(edge)
                    continue

                # Route successful - update state
                self._update_tier_state_for_successful_route(
                    tier_ind, edge, route, switch_info, tier_state
                )

                if progress_bar:
                    progress_bar.update(1)

            # Add switch nodes to the tier graph
            for idx, switch_node in enumerate(tier_state["switch_nodes"]):
                self.tiers[tier_ind].G.add_node(switch_node)
                self.tiers[tier_ind].G.nodes[switch_node]["pos"] = tier_state["switch_nodes_pos"][
                    idx
                ]

            # Add switch edges to the tier graph
            for e, switch_list in tier_state["switch_edges"].items():
                if switch_list:
                    # Only remove edge if it exists
                    if self.tiers[tier_ind].G.has_edge(e[0], e[1]):
                        self.tiers[tier_ind].G.remove_edge(e[0], e[1])
                for switch_edge in switch_list:
                    self.tiers[tier_ind].G.add_edge(switch_edge[0], switch_edge[1])
                    self.tiers[tier_ind].G.edges[switch_edge[0], switch_edge[1]]["route"] = (
                        tier_state["switch_edges_route"][(switch_edge[0], switch_edge[1])]
                    )
                    self.tiers[tier_ind].G.edges[switch_edge[0], switch_edge[1]]["full_edge"] = (
                        e[0],
                        e[1],
                    )
            # Finalize tier
            self._finalize_tier(tier_ind, tier_state)

        finally:
            if progress_bar:
                progress_bar.close()

    def _initialize_tier_state(self, tier_ind: int, reattempt: bool) -> dict:
        """Initialize or restore the state for a given tier."""
        if not reattempt:
            return {
                "routes": [],
                "switch_nodes": [],
                "switch_nodes_pos": [],
                "switch_edges": {},
                "switch_edges_route": {},
                "switch_num": 0,
                "edges_attempted": [],
                "routed_edge_order": [],
                "tot_route_length": 0,
                "max_routed_face_switches": 0,
            }

        # Restore from existing tier state
        tier = self.tiers[tier_ind]
        return {
            "routes": getattr(tier, "routes", []),
            "switch_nodes": getattr(tier, "switch_nodes", []),
            "switch_nodes_pos": getattr(tier, "switch_nodes_pos", []),
            "switch_edges": getattr(tier, "switch_edges", {}),
            "switch_edges_route": getattr(tier, "switch_edges_route", {}),
            "switch_num": getattr(tier, "switch_num", 0),
            "edges_attempted": [
                edge
                for edge in getattr(tier, "edges_attempted", [])
                if not self._edge_exists_in_list(
                    edge, []
                )  # Filter out edges no longer in route list
            ],
            "routed_edge_order": getattr(tier, "routed_edge_order", []),
            "tot_route_length": getattr(tier, "tot_route_length", 0),
            "max_routed_face_switches": getattr(tier, "max_routed_face_switches", 0),
        }

    def _route_single_edge(self, tier_ind: int, edge, router, switch_num) -> Optional[tuple]:
        """
        Attempt to route a single edge using straightline first, then A*.

        Returns:
            tuple: (route, switch_info) if successful, None if failed
        """
        source_pos = self.tiers[tier_ind].G.nodes[edge[0]]["pos"]
        target_pos = self.tiers[tier_ind].G.nodes[edge[1]]["pos"]
        start = (int(source_pos[0]), int(source_pos[1]), 0)
        goal = (int(target_pos[0]), int(target_pos[1]), 0)

        # Try straightline first
        route = straightline(self.tiers[tier_ind].expanded_grid, start, goal)

        if isinstance(route, bool):
            if self.verbose:
                print(f"Straight line routing failed for edge {edge} on tier {tier_ind}")
            # Try A* routing
            route = router.route(start, goal)
            if isinstance(route, bool):
                if self.verbose:
                    print(f"A* routing failed for edge {edge} on tier {tier_ind}")
                return None
            route = route[::-1]
        else:
            route = route[::-1]

        # Process face switches
        switch_info = self._process_face_switches(edge, route, switch_num)
        return route, switch_info

    def _process_face_switches(self, edge, route, switch_num) -> dict:
        """Process face switches in a route and return switch information."""
        switch_info = {
            "edges": [],
            "routes": {},
            "nodes": [],
            "positions": [],
            "num_switches": 0,
        }

        curr_switch_index = 0

        for i in range(len(route) - 1):
            if route[i][2] != route[i + 1][2]:
                switch_node = ("switch", switch_num)
                switch_pos = (route[i][0], route[i][1])

                switch_info["nodes"].append(switch_node)
                switch_info["positions"].append(switch_pos)

                if not switch_info["edges"]:
                    switch_info["edges"].append((edge[0], switch_node))
                    switch_info["routes"][(edge[0], switch_node)] = route[: i + 1]
                    curr_switch_index = i
                else:
                    prev_switch = switch_info["edges"][-1][1]
                    switch_info["routes"][(prev_switch, switch_node)] = route[
                        curr_switch_index : i + 1
                    ]
                    switch_info["edges"].append((prev_switch, switch_node))
                    curr_switch_index = i

                switch_num += 1
                switch_info["num_switches"] += 1

        if switch_info["edges"]:
            last_switch = switch_info["edges"][-1][1]
            switch_info["routes"][(last_switch, edge[1])] = route[curr_switch_index:]
            switch_info["edges"].append((last_switch, edge[1]))

        return switch_info

    def _check_routing_constraints(self, edge, route, switch_info) -> bool:
        """Check if the routing satisfies all constraints."""
        if self.max_bump_transitions_per_coupler is not None:
            if len(switch_info["edges"]) > self.max_bump_transitions_per_coupler:
                if self.verbose:
                    print(f"Max face switches exceeded for edge {edge}")
                return False

        if self.max_coupler_length is not None:
            coupler_length = len(route)
            if coupler_length / self.min_length_coupler > self.max_coupler_length:
                return False

        return True

    def _update_tier_state_for_successful_route(
        self, tier_ind: int, edge, route, switch_info, tier_state
    ) -> None:
        """Update tier state after a successful route."""
        tier_state["switch_edges"][(edge[0], edge[1])] = switch_info["edges"]
        tier_state["switch_edges_route"].update(switch_info["routes"])
        tier_state["switch_nodes"].extend(switch_info["nodes"])
        tier_state["switch_nodes_pos"].extend(switch_info["positions"])
        tier_state["switch_num"] += switch_info["num_switches"]

        tier_state["tot_route_length"] += len(route)
        if len(switch_info["edges"]) > tier_state["max_routed_face_switches"]:
            tier_state["max_routed_face_switches"] = len(switch_info["edges"])

        self.tiers[tier_ind].G.edges[edge[0], edge[1]]["route"] = copy.deepcopy(route)

        if "full_edge" not in self.tiers[tier_ind].G.edges[edge[0], edge[1]]:
            self.tiers[tier_ind].G.edges[edge[0], edge[1]]["full_edge"] = (
                edge[0],
                edge[1],
            )

        # Check for collisions (for debugging)
        collisions = [
            list(set(r).intersection(route))
            for r in tier_state["routes"]
            if set(r).intersection(route)
        ]
        if self.verbose:
            print("Collisions for edge", edge, ":", collisions)
        tier_state["routes"].append(route)

        # Mark the grid
        self._mark_grid_route(self.tiers[tier_ind], route)

        # Update routed edge order
        if switch_info["edges"]:
            switch_info["edges"].reverse()
            for sw_edge in switch_info["edges"]:
                tier_state["routed_edge_order"].append(sw_edge)
        else:
            tier_state["routed_edge_order"].append((edge[0], edge[1]))

    def _promote_edges_to_next_tier(
        self, tier_ind: int, edges_to_route: List, edge, tier_state
    ) -> None:
        """Promote edges to the next tier when constraints are violated."""
        # Add switch nodes to current tier
        for idx, switch_node in enumerate(tier_state["switch_nodes"]):
            self.tiers[tier_ind].G.add_node(switch_node)
            self.tiers[tier_ind].G.nodes[switch_node]["pos"] = tier_state["switch_nodes_pos"][idx]

        # Add switch edges to current tier
        for e, switch_list in tier_state["switch_edges"].items():
            if switch_list:
                # Only remove edge if it exists
                if self.tiers[tier_ind].G.has_edge(e[0], e[1]):
                    self.tiers[tier_ind].G.remove_edge(e[0], e[1])
            for switch_edge in switch_list:
                self.tiers[tier_ind].G.add_edge(switch_edge[0], switch_edge[1])
                self.tiers[tier_ind].G.edges[switch_edge[0], switch_edge[1]]["route"] = tier_state[
                    "switch_edges_route"
                ][(switch_edge[0], switch_edge[1])]
                self.tiers[tier_ind].G.edges[switch_edge[0], switch_edge[1]]["full_edge"] = (
                    e[0],
                    e[1],
                )

        # Remove remaining edges from current tier
        remaining_edges = edges_to_route.copy()
        remaining_edges.append(edge)
        self.tiers[tier_ind].G.remove_edges_from(remaining_edges)

        # Finalize current tier
        self._finalize_tier(tier_ind, tier_state)

        # Create or use next tier
        if tier_ind + 1 >= len(self.tiers):
            new_tier_graph = nx.Graph()
            print(f"- Created new tier {len(self.tiers)}")

            self.tiers.append(
                Tier(
                    new_tier_graph,
                    self.tiers[0].grid.shape[0],
                    self.tiers[0].grid.shape[1],
                    self.node_expansion_val,
                )
            )
        else:
            print("Using existing higher tier")
            new_tier_graph = self.tiers[tier_ind + 1].G

        # Copy nodes to the new tier
        for e in remaining_edges:
            new_tier_graph.add_node(
                e[0],
                pos=self.tiers[tier_ind].G.nodes[e[0]]["pos"],
            )
            new_tier_graph.add_node(
                e[1],
                pos=self.tiers[tier_ind].G.nodes[e[1]]["pos"],
            )
        new_tier_graph.add_edges_from(remaining_edges)

    def _finalize_tier(self, tier_ind: int, tier_state) -> None:
        """Finalize a tier with computed metrics and state."""
        # Cache metrics
        if tier_state["switch_edges"]:
            self.tiers[tier_ind].cached_avg_route_length = tier_state["tot_route_length"] / len(
                tier_state["switch_edges"]
            )
            self.tiers[tier_ind].cached_avg_face_switches = sum(
                len(tier_state["switch_edges"][edge]) for edge in tier_state["switch_edges"].keys()
            ) / len(tier_state["switch_edges"])
        else:
            self.tiers[tier_ind].cached_avg_route_length = 0
            self.tiers[tier_ind].cached_avg_face_switches = 0

        self.tiers[tier_ind].cached_max_face_switches = tier_state["max_routed_face_switches"]

        # Store state in tier
        self.tiers[tier_ind].switch_nodes = tier_state["switch_nodes"]
        self.tiers[tier_ind].switch_nodes_pos = tier_state["switch_nodes_pos"]
        self.tiers[tier_ind].switch_edges = tier_state["switch_edges"]
        self.tiers[tier_ind].switch_edges_route = tier_state["switch_edges_route"]
        self.tiers[tier_ind].switch_num = tier_state["switch_num"]
        self.tiers[tier_ind].edges_attempted = tier_state["edges_attempted"]
        self.tiers[tier_ind].routed_edge_order = tier_state["routed_edge_order"]
        self.tiers[tier_ind].tot_route_length = tier_state["tot_route_length"]
        self.tiers[tier_ind].max_routed_face_switches = tier_state["max_routed_face_switches"]

    def _mark_grid_route(self, tier, route) -> None:
        """Mark grid cells along a route."""
        for x0, y0, z0 in route:
            if tier.grid[x0, y0, z0] == 10:
                continue

            x_low = max(x0 - self.edge_expansion_val, 0)
            x_high = min(x0 + self.edge_expansion_val, tier.grid.shape[0] - 1)
            y_low = max(y0 - self.edge_expansion_val, 0)
            y_high = min(y0 + self.edge_expansion_val, tier.grid.shape[1] - 1)

            if z0 == 0:
                tier.grid[x0, y0, z0] = 2
                for i in range(x_low, x_high + 1):
                    for j in range(y_low, y_high + 1):
                        if tier.expanded_grid[i, j, z0] != 10:
                            tier.expanded_grid[i, j, z0] = 2
            elif z0 == 1:
                tier.grid[x0, y0, z0] = 4
                tier.expanded_grid[x_low : x_high + 1, y_low : y_high + 1, z0] = 4

    def _edge_exists_in_list(self, edge, edge_list) -> bool:
        """Check if an edge exists in a list (helper function from original code)."""
        for i, edge_i in enumerate(edge_list):
            exists = True
            if isinstance(edge_i[0], tuple):
                continue
            if isinstance(edge_i[1], tuple):
                continue
            nodes = [
                (edge_i[0].index, edge_i[0].is_data),
                (edge_i[1].index, edge_i[1].is_data),
            ]
            for node in edge[:2]:
                if isinstance(node, tuple):
                    continue
                tuple_node = (node.index, node.is_data)
                exists = exists and tuple_node in nodes
            if exists:
                return True
        return False
