from __future__ import annotations

import json
import multiprocessing
import os
import time
from copy import deepcopy
from functools import wraps
from itertools import product
from pathlib import Path
from typing import Mapping, Optional

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import hal.placing.util as placing
import hal.routing.routing_algo as routing
import hal.utils.disk_utils as disku
import hal.utils.graph_utils as graphu
import hal.utils.grid_utils as gridu
import hal.utils.visualization_utils as vizu
from hal.settings import Settings
from hal.tier import Tier


def time_it(func):
    """Decorator to measure and store the execution time of a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time

        func_name = func.__name__
        if not hasattr(self, "perf_metrics"):
            self.perf_metrics = {}

        self.perf_metrics[f"{func_name}_time"] = duration
        print(f"⏱️  '{func_name}' executed in {duration:.4f} seconds.")
        return result

    return wrapper


def _run_single_config(args: tuple) -> Optional[pd.DataFrame]:
    """
    Worker function to process a single combination of settings.
    """
    base_init_args, base_settings, value_overrides, name_overrides = args

    # Use the clean 'name_overrides' for the directory/file name
    override_str = "_".join(f"{k}={v}" for k, v in name_overrides.items())
    unique_name = f"{base_init_args['name']}_{override_str}"

    try:
        task_settings = deepcopy(base_settings)
        # Use the real 'value_overrides' to configure the settings
        for key, value in value_overrides.items():
            setattr(task_settings, key, value)

        task_settings.verbose = False
        task_init_args = base_init_args.copy()
        task_init_args["name"] = unique_name
        task_init_args["settings"] = task_settings
        task_init_args["directory_path"] = str(Path(base_init_args["directory_path"]).parent)

        print(f"🚀 Starting process for: {unique_name}")
        ham = HardwareAwareLayout(**task_init_args)
        ham.place()
        ham.route()
        bench_results = ham.benchmark()

        # Add the clean 'name_overrides' as columns to the results DataFrame
        for key, value in name_overrides.items():
            bench_results[key] = value

        print(f"✅ Finished process for: {unique_name}")
        return bench_results

    except Exception as e:
        print(f"❌ Process for {unique_name} failed: {e}")
        return None


class HardwareAwareLayout:
    """Place and route a Tanner graph onto tiered hardware grids.

    Uses a tiered planarity-based placement followed by grid routing.

    Attributes
    ----------
    name : str
        Identifier for the mapping instance.
    path_to_directory : pathlib.Path
        Directory where artifacts for this run are saved.
    settings : hal.settings.Settings
        Configuration controlling placement, routing and benchmarking.
    verbose : bool
        Whether to print detailed progress information.
    routed : bool
        True after higher-tier routing has completed; False otherwise.
    perf_metrics : dict[str, float]
        Timing metrics populated by the `time_it` decorator (e.g. place_time).
    parity_check_matrix : np.ndarray | None
        Parity-check matrix used to generate the Tanner graph if provided.
    G_tanner : nx.Graph | None
        Tanner graph representation of the code.
    tiers : list[hal.tier.Tier] | None
        Per-tier graphs and grids after placement; updated after routing.
    grid_size : int | None
        Square grid dimension used when `toric_code` is not provided.
    grid_size_x : int | None
        Grid width (rectangular case or derived from settings when square).
    grid_size_y : int | None
        Grid height (rectangular case or derived from settings when square).
    margin_size : float | None
        Margin in grid cells (square grid case) used for rasterization.
    grid_spacing : np.ndarray | None
        Spacing used to map continuous positions onto the grid (square case).
    node_expansion_val : int | None
        Reserved expansion radius around each node on the grid.
    """

    def __init__(
        self,
        name: str,
        directory_path: str,
        parity_check_matrix: np.ndarray = None,
        tanner_graph: nx.Graph = None,
        settings: Settings = None,
        path_to_tiers: Optional[list[str]] = None,
        path_to_layers: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize a HardwareAwareLayout instance.

        Either a parity check matrix or a Tanner graph must be provided.

        Parameters
        ----------
        name : str
            Human-friendly mapping identifier used in filenames.
        directory_path : str
            Output directory where a timestamped run folder will be created.
        parity_check_matrix : np.ndarray | None, optional
            Full parity-check matrix; if provided, a Tanner graph is generated.
        tanner_graph : nx.Graph | None, optional
            Pre-constructed Tanner graph to use directly instead of `parity_check_matrix`.
        settings : Settings | None, optional
            Settings instance; if None, defaults are used and printed.
        path_to_tiers : list[str] | None, optional
            Paths to previously saved tier artifacts (graphs/grids/metrics) to load.
        path_to_layers : list[str] | None, optional
            Backwards compatible alias for `path_to_tiers`.

        Raises
        ------
        ValueError
            If neither `parity_check_matrix` nor `tanner_graph` is provided.
        """
        self.name = name
        self.path_to_directory = disku.create_path(
            directory_path=directory_path,
            measurement_name=self.name,
            create_directory=True,
        )

        # Initialize settings with defaults and merge with provided settings
        if settings is None:
            print("=" * 60)
            print("🔧 HardwareAwareLayout: Using DEFAULT settings")
            print("=" * 60)
            self.settings = Settings()
            print(f"📋 {self.settings}")
            print("💡 Tip: Pass a Settings object to customize behavior")
        else:
            print("=" * 60)
            print("🔧 HardwareAwareLayout: Using USER-PROVIDED settings")
            print("=" * 60)
            self.settings = settings
            print(f"📋 {self.settings}")

        print("=" * 60)
        print()

        self.verbose = self.settings.verbose
        self.routed = False
        self.perf_metrics = {}

        if parity_check_matrix is not None:
            self.parity_check_matrix = parity_check_matrix
            self.G_tanner = graphu.generate_tanner_graph(parity_check_matrix)
        elif tanner_graph is not None:
            self.G_tanner = tanner_graph
            self.parity_check_matrix = None
        else:
            raise ValueError("Must provide either parity check matrix or Tanner graph")

        # Load tier graphs if provided (backward compat: path_to_layers)
        if path_to_tiers is not None or path_to_layers is not None:
            tier_graphs = {}
            grids = {}
            expanded_grids = {}
            metrics_paths = {}
            inds = []
            for path in path_to_tiers or path_to_layers:
                ind = int(os.path.split(path)[-1].split("_")[-1].split(".")[0])
                inds.append(ind)
                if "graph" in path:
                    tier_graphs[ind] = disku.load_graph(path)
                if "grid" in path and not "expanded" in path:
                    grids[ind] = disku.load_grid(path)
                if "grid" in path and "expanded" in path:
                    expanded_grids[ind] = disku.load_grid(path)
                if "metric" in path:
                    metrics_paths[ind] = path

            self.tiers = []
            for ind in range(max(inds) + 1):
                graph = tier_graphs[ind]
                grid = grids[ind]
                expanded_grid = expanded_grids[ind]
                metrics_path = metrics_paths[ind]
                self.tiers.append(
                    Tier.from_database(graph, grid, expanded_grid, metrics_path=metrics_path)
                )
            self.routed = True
        else:
            self.tiers = None

    @classmethod
    def from_database(cls, directory_path: str) -> "HardwareAwareLayout":
        """
        Create an instance from a mapping database directory.

        Parameters
        ----------
        directory_path : str
            Directory containing mapping data previously saved by `save`.

        Returns
        -------
        HardwareAwareLayout
            A new instance loaded from disk.
        """
        path_list = os.path.split(directory_path)
        name = path_list[-1].split("_")[-1]

        try:
            parity_check_matrix_path = os.path.join(directory_path, "parity_check_matrix.csv")
            parity_check_matrix = np.loadtxt(parity_check_matrix_path, delimiter=",")
        except FileNotFoundError:
            parity_check_matrix = None

        try:
            tanner_graph_path = os.path.join(directory_path, "tanner")
            tanner_graph = disku.load_graph(tanner_graph_path)
        except FileNotFoundError:
            tanner_graph = None

        if tanner_graph is None and parity_check_matrix is None:
            raise ValueError("Cannot load from folder without parity check matrix or tanner graph")

        root_tiers_path = os.path.join(directory_path, "tiers")
        if any(os.scandir(root_tiers_path)):
            tiers_paths = [
                os.path.join(root_tiers_path, f)
                for f in os.listdir(root_tiers_path)
                if os.path.isfile(os.path.join(root_tiers_path, f))
            ]
        else:
            tiers_paths = None

        settings_path = os.path.join(directory_path, "settings.json")
        try:
            settings = Settings.load_from_json(settings_path)
        except FileNotFoundError:
            print(f"No settings.json file found in {directory_path}")
            settings = None

        return cls(
            name,
            directory_path,
            parity_check_matrix=parity_check_matrix,
            tanner_graph=tanner_graph,
            path_to_tiers=tiers_paths,
            settings=settings,
        )

    @property
    def complexity(self) -> float:
        """
        Calculate the hardware complexity of the mapping.

        The complexity is defined as the product of the number of tiers,
        the number of nodes, and the number of edges in the Tanner graph.

        Returns
        -------
        float
            The calculated hardware complexity.
        """
        if self.tiers is None or len(self.tiers) == 0:
            raise ValueError("Placing not performed. Please run 'place' method first.")
        if not self.routed:
            raise ValueError("Routing not performed. Please run 'route' method first.")

        return self.benchmark()["hardware_complexity"].iloc[0]

    @property
    def num_tiers(self) -> int:
        """
        Return the number of tiers.
        """
        if self.tiers is None:
            return 0
        return len(self.tiers)

    @time_it
    def place(
        self,
        layout: Optional[str] = None,
        mps_edge_order: Optional[str] = None,
        grid_size: Optional[int] = None,
        margin: Optional[float] = None,
        node_expansion_val: Optional[int] = None,
        orthogonalize_layout: bool = True,
        custom_pos: Optional[dict] = None,
        toric_code: Optional[dict] = None,
    ) -> None:
        """
        Compute node placements and generate tiered graphs with corresponding grids.

        The method:
          1. Determines node positions using the specified layout.
          2. Orders edges and partitions them into tiers.
          3. Normalizes node positions.
          4. Creates grid and expanded grid representations for each tier.

        Parameters
        ----------
        layout : str | None
            Layout algorithm ('community', 'spring', 'kamada_kawai', 'grid', etc.).
        mps_edge_order : str | None
            Ordering used while extracting a maximum planar subgraph (e.g. 'length_asc').
        grid_size : int | None
            Square grid dimension used when not placing a toric code.
        margin : float | None
            Margin ratio used during rasterization (square grid case).
        node_expansion_val : int | None
            Expansion radius (in cells) reserved around each node.
        orthogonalize_layout : bool
            If True, derive an orthogonalized base layer before tiering.
        custom_pos : dict | None
            Optional user-provided positions for nodes; bypasses layout.
        toric_code : dict | None
            Enable rectangular placement tailored for toric-code style layouts.

        Returns
        -------
        None
        """
        # Merge settings with provided parameters
        effective_settings = self.settings

        # Resolve parameters: method args override settings, settings override defaults
        layout = layout if layout is not None else effective_settings.layout
        mps_edge_order = (
            mps_edge_order if mps_edge_order is not None else effective_settings.mps_edge_order
        )
        grid_size = grid_size if grid_size is not None else effective_settings.grid_size
        margin = margin if margin is not None else effective_settings.place_margin
        node_expansion_val = (
            node_expansion_val
            if node_expansion_val is not None
            else effective_settings.node_expansion_val
        )

        # Use custom positions from settings if not provided directly
        if custom_pos is None and effective_settings.custom_positions is not None:
            custom_pos = effective_settings.custom_positions

        # Print settings summary for this operation
        print(f"🔧 Placement settings: {effective_settings.operation_summary('place')}")

        # Check for parameter overrides
        overrides = effective_settings.show_overrides(
            grid_size=grid_size,
            layout=layout,
            node_expansion_val=node_expansion_val,
            margin=margin,
        )
        if "No settings overridden" not in overrides:
            print(f"⚠️  Parameter overrides: {overrides}")

        # Choose layout for node positions
        progress_print_length = 150
        substep_print_length = 60
        graphu.dash_print(f" Running HAL on code: {self.name} ", progress_print_length, "=")

        graphu.dash_print(" Place nodes ", progress_print_length, "-")

        layout_name = "custom_pos" if custom_pos is not None else layout
        print(f"- Order nodes using {layout_name} layout")

        if custom_pos is not None:
            pos = custom_pos
            orthogonalize_layout = False
        elif layout == "kamada_kawai":
            self.G_tanner = nx.Graph(self.G_tanner)
            pos = nx.kamada_kawai_layout(self.G_tanner)
        elif layout == "shell":
            self.G_tanner = nx.Graph(self.G_tanner)
            pos = nx.shell_layout(self.G_tanner)
        elif layout == "circular":
            self.G_tanner = nx.Graph(self.G_tanner)
            pos = nx.circular_layout(self.G_tanner)
        elif layout == "spiral":
            self.G_tanner = nx.Graph(self.G_tanner)
            pos = nx.spiral_layout(self.G_tanner)
        elif layout == "spring":
            self.G_tanner = nx.Graph(self.G_tanner)
            pos = nx.spring_layout(self.G_tanner)
        elif layout == "spectral":
            self.G_tanner = nx.Graph(self.G_tanner)
            pos = nx.spectral_layout(self.G_tanner)
        elif layout == "random":
            self.G_tanner = nx.Graph(self.G_tanner)
            pos = nx.random_layout(self.G_tanner)
        elif layout == "grid":
            self.G_tanner = nx.Graph(self.G_tanner)
            min_grid_size = np.ceil(np.sqrt(len(self.G_tanner.nodes)))
            pos = placing.get_systematic_unique_layout(self.G_tanner, grid_size=min_grid_size)
        elif layout == "community":
            self.G_tanner = nx.Graph(self.G_tanner)
            pos = graphu.community_layout(self.G_tanner)
        else:
            raise ValueError(f"Unsupported layout type: {layout}")

        # Assign positions to the Tanner graph nodes
        for node in self.G_tanner.nodes:
            self.G_tanner.nodes[node]["pos"] = pos[node]

        # Orthogonalize the base tier if desired
        if orthogonalize_layout:
            print("- Orthogonalize base tier")

            ordered_edges = graphu.get_ordered_edges(self.G_tanner, mps_edge_order)

            # Partition the edges into tiers based on thickness
            tier_list = graphu.thickness(self.G_tanner, ordered_edges)
            higher_tiers_combined = (
                nx.compose_all(tier_list[1:]) if len(tier_list) > 1 else nx.Graph()
            )
            qubit_tier_graph = tier_list[0]
            min_grid_size = np.ceil(np.sqrt(len(qubit_tier_graph.nodes)))

            print("- Place node positions using Kamada Kawai spring layout")
            positions = placing.get_systematic_unique_layout(
                qubit_tier_graph, grid_size=min_grid_size, verbose=self.verbose
            )

            print("- Rasterize node positions")
            positions = placing.rasterise(positions)
        else:
            positions = nx.get_node_attributes(self.G_tanner, "pos")

            long_range_edges = [
                (u, v)
                for u, v in self.G_tanner.edges()
                if abs(pos[u][0] - pos[v][0]) + abs(pos[u][1] - pos[v][1]) != 1
            ]

            qubit_tier_graph = deepcopy(self.G_tanner)
            higher_tiers_combined = deepcopy(self.G_tanner)
            higher_tiers_combined.remove_edges_from(list(higher_tiers_combined.edges()))

            for edge in long_range_edges:
                qubit_tier_graph.remove_edge(*edge)
                higher_tiers_combined.add_edge(*edge)

        # Normalize node positions so that all coordinates are positive
        xs = {node: pos[0] for node, pos in positions.items()}
        ys = {node: pos[1] for node, pos in positions.items()}
        x_offset = abs(min(xs.values())) if min(xs.values()) < 0 else -1 * min(xs.values())
        y_offset = abs(min(ys.values())) if min(ys.values()) < 0 else -1 * min(ys.values())
        for node in positions:
            new_pos = (xs[node] + x_offset, ys[node] + y_offset)
            positions[node] = new_pos

            if node in qubit_tier_graph.nodes:
                qubit_tier_graph.nodes[node]["pos"] = (
                    int(new_pos[0]),
                    int(new_pos[1]),
                )

        # Generate grid representations for each tier
        if toric_code:
            xs = {node: pos[0] for node, pos in positions.items()}
            ys = {node: pos[1] for node, pos in positions.items()}

            x_min, x_max = min(xs.values()), max(xs.values())
            y_min, y_max = min(ys.values()), max(ys.values())

            delta_x = x_max - x_min
            delta_y = y_max - y_min

            if delta_x > delta_y:
                more_qubits = delta_x + 1
                less_qubits = delta_y + 1
                scaling_factor = np.sqrt(less_qubits / more_qubits)
                grid_size_y = int(
                    np.ceil(grid_size * scaling_factor)
                )  # give benefit to shorter side
                grid_size_x = int(np.floor(grid_size / scaling_factor))

                margin_size_x = grid_size_x * margin
                margin_size_y = (
                    grid_size_y - 1 - delta_y / (delta_x / (grid_size_x - margin_size_x - 1))
                )
            else:
                less_qubits = delta_x + 1
                more_qubits = delta_y + 1
                scaling_factor = np.sqrt(less_qubits / more_qubits)
                grid_size_x = int(
                    np.ceil(grid_size * scaling_factor)
                )  # give benefit to shorter side
                grid_size_y = int(np.floor(grid_size / scaling_factor))

                margin_size_y = grid_size_y * margin
                margin_size_x = (
                    grid_size_x - 1 - delta_x / (delta_y / (grid_size_y - margin_size_y - 1))
                )

            grid_spacing = np.array(
                [
                    delta_x / (grid_size_x - margin_size_x - 1),
                    delta_y / (grid_size_y - margin_size_y - 1),
                ]
            )

            if self.verbose:
                print(
                    f"Grid size: {grid_size_x} x {grid_size_y}, "
                    f"Total number of cells: {grid_size_x * grid_size_y}, "
                    f"Node expansion value: {node_expansion_val}, "
                    f"Grid spacing: {[int(np.round(1/grid_spacing[0])), int(np.round(1/grid_spacing[1]))]}, "
                    f"Margin sizes: ({int(np.round(margin_size_x)), int(np.round(margin_size_y))})"
                )

            for node in qubit_tier_graph:
                # Map node position to grid indices
                grid_ind = gridu.position_to_rectangular_grid(
                    qubit_tier_graph.nodes[node]["pos"],
                    margin_size_x=margin_size_x,
                    margin_size_y=margin_size_y,
                    grid_spacing=grid_spacing,
                )
                x_idx, y_idx = int(grid_ind[0]), int(grid_ind[1])

                qubit_tier_graph.nodes[node]["pos"] = (x_idx, y_idx)

                if node in higher_tiers_combined.nodes:
                    higher_tiers_combined.nodes[node]["pos"] = (x_idx, y_idx)

        else:
            margin_size = grid_size * margin

            x_min, x_max = min(xs.values()), max(xs.values())
            y_min, y_max = min(ys.values()), max(ys.values())
            grid_spacing = np.array(
                [
                    (x_max - x_min) / (grid_size - margin_size - 1),
                    (y_max - y_min) / (grid_size - margin_size - 1),
                ]
            )

            for node in qubit_tier_graph:
                # Map node position to grid indices
                grid_ind = gridu.position_to_grid(
                    qubit_tier_graph.nodes[node]["pos"],
                    margin_size=margin_size,
                    grid_spacing=grid_spacing,
                )
                x_idx, y_idx = int(grid_ind[0]), int(grid_ind[1])
                qubit_tier_graph.nodes[node]["pos"] = (x_idx, y_idx)

                if node in higher_tiers_combined.nodes:
                    higher_tiers_combined.nodes[node]["pos"] = (x_idx, y_idx)

            grid_size_y = grid_size
            grid_size_x = grid_size
            if self.verbose:
                print(
                    f"Grid size: {grid_size_x} x {grid_size_y}, "
                    f"Total number of cells: {grid_size_x * grid_size_y}, "
                    f"Node expansion value: {node_expansion_val}, "
                    f"Grid spacing: {[int(np.round(1/grid_spacing[0])), int(np.round(1/grid_spacing[1]))]}, "
                    f"Margin sizes: ({int(np.round(margin_size)), int(np.round(margin_size))})"
                )

        # For edges with bends, assign a "full_edge" attribute to reduce ambiguity
        for edge in qubit_tier_graph.edges:
            if not isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
                full_edge_source = edge[0]
                curr_node = edge[1]
                edges_in_route = [(full_edge_source, curr_node)]
                while isinstance(curr_node, tuple) and "bend" in curr_node:
                    for search_edge in qubit_tier_graph.edges:
                        if curr_node in search_edge and search_edge not in edges_in_route:
                            edges_in_route.append(search_edge)
                            curr_node = (
                                search_edge[0] if search_edge[1] == curr_node else search_edge[1]
                            )
                            break
                full_edge_goal = curr_node
                full_edge = (
                    min(full_edge_source, full_edge_goal),
                    max(full_edge_source, full_edge_goal),
                )
                for e in edges_in_route:
                    qubit_tier_graph.edges[e[0], e[1]]["full_edge"] = full_edge

        self.routed = False
        self.tiers = []
        self.tiers.append(
            Tier(
                qubit_tier_graph,
                grid_size_x,
                grid_size_y,
                node_expansion_val,
            )
        )
        self.tiers.append(
            Tier(
                higher_tiers_combined,
                grid_size_x,
                grid_size_y,
                node_expansion_val,
            )
        )

        self.save()

    @time_it
    def route(
        self,
        max_bump_transitions_per_coupler: Optional[int] = None,
        max_coupler_length: Optional[int] = None,
        max_tsvs_per_coupler: Optional[int] = None,
        routing_edge_order: Optional[str] = None,
        edge_expansion_val: Optional[int] = None,
    ) -> None:
        """
        Route edges across the tiers.

        The method first routes the base tier using a straightline connection,
        then attempts to route remaining edges. If conflicts arise (or an edge is
        repeatedly attempted), the edge is moved to a new tier.

        Parameters
        ----------
        max_bump_transitions_per_coupler : int | None
            Maximum allowed face switches per routed edge before promotion.
        max_coupler_length : int | None
            Maximum allowed coupler length per routed edge before promotion.
        max_tsvs_per_coupler : int | None
            Maximum allowed TSVs per coupler; if exceeded, routing stops.
        routing_edge_order : str | None
            Ordering method for selecting edges to route.
        edge_expansion_val : int | None
            Grid-cell expansion used to mark routed traces.

        Returns
        -------
        None
        """
        # Merge settings with provided parameters
        effective_settings = self.settings

        # Resolve parameters: method args override settings, settings override defaults
        max_bump_transitions_per_coupler = (
            max_bump_transitions_per_coupler
            if max_bump_transitions_per_coupler is not None
            else effective_settings.max_bump_transitions_per_coupler
        )
        max_tsvs_per_coupler = (
            max_tsvs_per_coupler
            if max_tsvs_per_coupler is not None
            else effective_settings.max_tsvs_per_coupler
        )
        routing_edge_order = (
            routing_edge_order
            if routing_edge_order is not None
            else effective_settings.route_edge_order
        )
        edge_expansion_val = (
            edge_expansion_val
            if edge_expansion_val is not None
            else effective_settings.edge_expansion_val
        )
        max_coupler_length = (
            max_coupler_length
            if max_coupler_length is not None
            else effective_settings.max_coupler_length
        )
        node_expansion_val = effective_settings.node_expansion_val

        # Print settings summary for this operation
        print(f"🔧 Routing settings: {effective_settings.operation_summary('route')}")

        # Check for parameter overrides
        overrides = effective_settings.show_overrides(
            max_bump_transitions_per_coupler=max_bump_transitions_per_coupler,
            max_coupler_length=max_coupler_length,
            max_tsvs_per_coupler=max_tsvs_per_coupler,
            edge_expansion_val=edge_expansion_val,
        )
        if "No settings overridden" not in overrides:
            print(f"⚠️  Parameter overrides: {overrides}")

        progress_print_length = 150
        graphu.dash_print(" Route edges ", progress_print_length, "-")

        # Create the tier-based router
        router = routing.TierBasedRouter(
            tiers=self.tiers,
            max_bump_transitions_per_coupler=max_bump_transitions_per_coupler,
            max_tsvs_per_coupler=max_tsvs_per_coupler,
            routing_edge_order=routing_edge_order,
            edge_expansion_val=edge_expansion_val,
            routing_algorithm="straightline_astar",  # Fixed for now since we only have one approach
            max_coupler_length=max_coupler_length,
            node_expansion_val=node_expansion_val,
            verbose=self.verbose,
        )

        # Route the base tier
        tot_route_length = router.route_base_tier()

        if len(self.tiers) == 0:
            raise ValueError("No tiers routed")

        # Early return if no higher tier edges need routing
        if self.tiers[-1].num_edges == 0:
            # FoldQ 2026-11-06: tier 0 无边（仅 marginal 边入图）时除零保护
            n0 = len(self.tiers[0].G.edges)
            self.tiers[0].cached_avg_route_length = (
                tot_route_length / n0 if n0 else 0.0)
            self.tiers[0].cached_max_face_switches = 0
            self.tiers[0].cached_avg_face_switches = 0
            self.tiers[0].routed_edge_order = list(self.tiers[0].G.edges(data=True))
            self.tiers.pop()
            self.save()
            return

        # Attempt to route higher tier edges on the base tier
        additional_length = router.attempt_higher_tier_on_base()
        tot_route_length += additional_length

        # # Update base tier metrics
        # FoldQ 2026-11-06: tier 0 无边（仅 marginal 边入图）时除零保护
        _n0 = len(self.tiers[0].G.edges)
        self.tiers[0].cached_avg_route_length = tot_route_length / _n0 if _n0 else 0.0
        self.tiers[0].cached_max_face_switches = 0
        self.tiers[0].cached_avg_face_switches = 0
        self.tiers[0].routed_edge_order = list(self.tiers[0].G.edges(data=True))

        # Route remaining edges on higher tiers
        router.route_higher_tiers()

        self.routed = True
        self.save()

    def benchmark_sweep(
        self,
        sweep_params: dict[str, list],
        num_processes: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run a parallelized sweep over selected settings and aggregate results.

        Each entry in `sweep_params` maps a Settings attribute name to a list of
        values to try. List entries can be:
        - a raw value (used both as the label and the value),
        - None (labelled as "Automatic"), or
        - a tuple (label, value) to decouple filenames/labels from actual values.

        Parameters
        ----------
        sweep_params : dict[str, list]
            Mapping from Settings attribute name to a list of candidate values.
        num_processes : int | None, optional
            Cap on worker processes; defaults to `os.cpu_count()`.

        Returns
        -------
        pandas.DataFrame
            Concatenated benchmark DataFrame across all successful configurations,
            with extra columns for the sweep labels.
        """
        if not sweep_params:
            raise ValueError("The 'sweep_params' dictionary cannot be empty.")

        # 1. Pre-process sweep parameters to separate labels from actual values
        param_keys = list(sweep_params.keys())
        value_lists = []
        name_lists = []

        for key in param_keys:
            current_values = []
            current_names = []
            for item in sweep_params[key]:
                if isinstance(item, tuple) and len(item) == 2:
                    # User provided a (name, value) tuple for a complex object
                    name, value = item
                    current_names.append(name)
                    current_values.append(value)
                elif item is None:
                    # Special handling for None, give it a clear label
                    current_names.append("Automatic")
                    current_values.append(None)
                elif isinstance(item, dict):
                    # Fallback for an unnamed dictionary: use a short, unique hash
                    # This is not descriptive but prevents crashing on long filenames.
                    label = f"dict_{abs(hash(str(item))) % 10000}"
                    print(
                        f"Warning: Unnamed dictionary found for '{key}'. Using fallback label: '{label}'"
                    )
                    current_names.append(label)
                    current_values.append(item)
                else:
                    # For simple parameters, the name is the value itself
                    current_names.append(item)
                    current_values.append(item)

            name_lists.append(current_names)
            value_lists.append(current_values)

        # 2. Generate combinations for both values and names
        value_combos = list(product(*value_lists))
        name_combos = list(product(*name_lists))

        # 3. Prepare arguments for each worker process
        base_init_args = {
            "name": self.name,
            "directory_path": self.path_to_directory,
            "parity_check_matrix": self.parity_check_matrix,
            "tanner_graph": self.G_tanner,
        }

        tasks = []
        for val_combo, name_combo in zip(value_combos, name_combos):
            value_overrides = dict(zip(param_keys, val_combo))
            name_overrides = dict(zip(param_keys, name_combo))
            tasks.append((base_init_args, self.settings, value_overrides, name_overrides))

        # ... (the rest of the function remains the same, executing the pool) ...
        if num_processes is None:
            num_processes = os.cpu_count() or 1
        process_count = min(num_processes, len(tasks))

        print(
            f"\n🔬 Starting benchmark sweep with {len(tasks)} configurations using {process_count} processes...\n"
        )

        with multiprocessing.Pool(processes=process_count) as pool:
            all_results = list(
                tqdm(
                    pool.imap(_run_single_config, tasks),
                    total=len(tasks),
                    desc="Running Sweep",
                )
            )

        print("\n📊 Parallel processing finished. Aggregating results...")
        successful_results = [res for res in all_results if res is not None]

        if not successful_results:
            print("⚠️ Warning: All sweep configurations failed.")
            return pd.DataFrame()

        final_results_df = pd.concat(successful_results, ignore_index=True)
        print("\n--- Sweep Complete ---")
        print(f"Successfully completed {len(successful_results)} of {len(tasks)} configurations.")

        return final_results_df

    @time_it
    def benchmark(
        self,
        bad: Optional[Mapping[str, float]] = None,
        baseline: Optional[Mapping[str, float]] = None,
        weights: Optional[Mapping[str, float]] = None,
        clip_high: bool = False,  # <-- changed default
    ) -> pd.DataFrame:
        """
        Baseline  -> score == 1
        Bad point -> score == 2
        Worse     -> score > 2     (because we do NOT cap above 'bad')

        Parameters
        ----------
        bad : dict
            Threshold where each metric is "equally bad".
        baseline : dict
            Reference point that should evaluate to 1.
            Defaults: 1, 1, 0, 0
        weights : dict or None
            Metric weights.  None ⇒ all 1.
        clip_high : bool, default False
            If True, r_m is clamped to 1 (old behaviour).
        """
        # Use baseline and bad defaults from settings, allow method parameter override
        baseline = {**self.settings.baseline_defaults, **(baseline or {})}
        bad = {**self.settings.bad_defaults, **(bad or {})}

        # Print settings summary for this operation
        print(f"🔧 Benchmark settings: {self.settings.operation_summary('benchmark')}")

        # Check for parameter overrides
        overrides = self.settings.show_overrides(baseline=baseline, bad=bad)
        if "No settings overridden" not in overrides:
            print(f"⚠️  Parameter overrides: {overrides}")

        # ---------------- raw numbers ---------------------------------
        if self.num_tiers == 0:
            raise ValueError("Mapping has no tiers")

        shortest = min(len(r) for r in self.tiers[0].routes)

        avg_coupler_length = (
            sum(tier.avg_route_length / shortest for tier in self.tiers) / self.num_tiers
        )

        max_avg_face_switches = max(tier.avg_face_switches for tier in self.tiers)

        edges_per_tier = [tier.num_edges for tier in self.tiers[1:]]
        if sum(edges_per_tier):
            avg_tsvs_per_edge = sum(
                2 * (i + 1) * n
                for i, n in enumerate(
                    edges_per_tier
                )  # changed to (i + 0) since we are now starting from tier 0
            ) / sum(edges_per_tier)
        else:
            avg_tsvs_per_edge = 0.0

        raw = {
            "num_tiers": float(self.num_tiers),
            "avg_coupler_length": float(avg_coupler_length),
            "max_avg_face_switches": float(max_avg_face_switches),
            "avg_tsvs_per_edge": float(avg_tsvs_per_edge),
        }

        # --------------- user knobs -----------------------------------
        # baseline and bad defaults now come directly from settings
        # Method parameters can still override individual keys
        weights = weights or {k: 1.0 for k in raw}

        # --------------- normalisation -------------------------------
        def ratio(val: float, g: float, b: float) -> float:
            if b <= g:
                raise ValueError("bad must be larger than baseline")
            r = max(0.0, (val - g) / (b - g))  # never go below 0
            if clip_high:
                r = min(1.0, r)
            return r

        r_values = {k: ratio(v, baseline[k], bad[k]) for k, v in raw.items()}

        # --------------- aggregate -----------------------------------
        total_w = sum(weights.values())
        hw_complexity = 1.0 + sum(r_values[k] * weights[k] for k in raw) / total_w

        # --------------- dataframe -----------------------------------
        df = pd.DataFrame(
            {
                **raw,
                **{f"{k}_norm": v for k, v in r_values.items()},
                "hardware_complexity": hw_complexity,
            },
            index=[self.name],
        )
        return df

    def display_grids(
        self,
        save: bool = False,
        grid_inds: Optional[list] = None,
        path: Optional[str] = None,
        dpi: Optional[int] = None,
    ) -> None:
        """Display and optionally save grid images for each tier.

        Delegates to `hal.utils.visualization_utils.display_grids`.
        """
        vizu.display_grids(self, save=save, grid_inds=grid_inds, path=path, dpi=dpi)

    def display_line_plot(self) -> None:
        """Plot nodes and routed edges as static line plots for each tier."""
        vizu.display_line_plot(self)

    def display(
        self,
        save: bool = False,
        path: Optional[str] = None,
        tiers: Optional[list] = None,
    ) -> tuple[list, list]:
        """Display an interactive graph of the mapping for selected tiers."""
        return vizu.display(self, save=save, path=path, tiers=tiers)

    def display_gif(
        self,
        tier_ind: int,
        save: bool = False,
        path: Optional[str] = None,
        fps: int = 1,
        step: int = 1,
    ):
        """Create an animated GIF of the routing process for a specific tier."""
        return vizu.display_gif(self, tier_ind=tier_ind, save=save, path=path, fps=fps, step=step)

    def save(self, path: Optional[str] = None) -> None:
        """Save all artifacts (graphs, grids, metrics, settings) to disk.

        Parameters
        ----------
        path : str | None, optional
            Directory root for saving. Defaults to the instance directory.
        """
        path_to_directory = Path(self.path_to_directory if path is None else path)
        if self.parity_check_matrix is not None:
            np.savetxt(
                f"{path_to_directory}/parity_check_matrix.csv",
                self.parity_check_matrix,
                delimiter=",",
            )
        if self.G_tanner is not None:
            disku.save_graph(self.G_tanner, "tanner", self.path_to_directory)

        if hasattr(self, "tiers") and self.tiers is not None:
            tiers_dir = Path(self.path_to_directory / "tiers")
            os.makedirs(tiers_dir, exist_ok=True)
            for ind, tier in enumerate(self.tiers):
                disku.save_graph(tier.G, f"graph_{ind}", tiers_dir)
                disku.save_grid(tier.grid, f"grid_{ind}", tiers_dir)
                disku.save_grid(tier.expanded_grid, f"expanded_grid_{ind}", tiers_dir)
                if self.routed:
                    tier.save_metrics(f"metrics_{ind}", tiers_dir)

            grid_view_dir = Path(self.path_to_directory / "grid_view")
            os.makedirs(grid_view_dir, exist_ok=True)
            self.display_grids(save=True, path=grid_view_dir)
            self.display(save=True, path=str(path_to_directory))

        if self.routed:
            self.benchmark().to_csv(path_to_directory / "benchmark.csv", index=False)

        if self.perf_metrics:
            metrics_to_save = self.perf_metrics.copy()

            # Calculate combined metrics if components exist
            place_time = metrics_to_save.get("place_time", 0)
            route_time = metrics_to_save.get("route_time", 0)

            if place_time > 0 and route_time > 0:
                metrics_to_save["place_and_route_time"] = place_time + route_time

            # Sum of all individual measurements (place, route, benchmark)
            metrics_to_save["total_time"] = sum(self.perf_metrics.values())

            perf_file_path = path_to_directory / "perf.json"
            with open(perf_file_path, "w") as f:
                json.dump(metrics_to_save, f, indent=4)

        self.settings.save_to_json(path_to_directory / "settings.json")
