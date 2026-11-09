import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class Settings:
    """Configuration for placement, routing, and benchmarking.

    Attributes
    ----------
    grid_size : int
        Base grid dimension used when square grids are employed.
    grid_size_x : int | None
        Grid width override; defaults to `grid_size` when unset.
    grid_size_y : int | None
        Grid height override; defaults to `grid_size` when unset.
    custom_positions : dict[int, tuple[int, int]] | None
        Explicit node positions; if set, bypasses layout for placement.
    layout : str
        Placement layout algorithm identifier (e.g. 'community', 'spring').
    mps_edge_order : str
        Edge selection order for maximal planar subgraph extraction.
    place_margin : float
        Margin ratio used during rasterization in placement.
    node_expansion_val : int
        Expansion radius around each node for local wiring clearance.
    toric_code : dict | None
        If set, enables rectangular placement tailored to toric codes.
    max_bump_transitions_per_coupler : int | None
        Limit on face-switch events along any single routed edge.
    max_tsvs_per_coupler : int | None
        Maximum TSVs permitted per coupler before routing aborts.
    max_coupler_length : int | None
        Upper bound on relative coupler length (normalized to the shortest).
    route_edge_order : str
        Edge ordering used when routing (e.g. 'length_asc').
    edge_expansion_val : int
        Expansion radius (in cells) to reserve around routed traces.
    save : bool
        Whether to save intermediate and final artifacts.
    verbose : bool
        Emit detailed logs during place/route/benchmark.
    baseline_defaults : dict
        Baseline metric thresholds where score equals 1.
    bad_defaults : dict
        "Bad" metric thresholds where score equals 2.
    """

    # === Placement Settings ===
    grid_size: int = 500
    """Grid size determining overall device area, aspect ratio, and layout granularity."""

    grid_size_x: Optional[int] = None
    """Override grid width. If None, uses grid_size."""

    grid_size_y: Optional[int] = None
    """Override grid height. If None, uses grid_size."""

    custom_positions: Optional[Dict[int, Tuple[int, int]]] = None
    """Explicit map p0: V → ℤ² that overrides automatic placement for vertices."""

    layout: str = "community"
    """Layout algorithm for placement: 'kamada_kawai', 'community', 'planar', etc."""

    mps_edge_order: str = "crossings_asc"
    """Edge ordering for MPS placement: 'length_desc', 'crossings_asc', etc."""

    place_margin: float = 0.02
    """Margin ratio for grid normalization during placement."""

    node_expansion_val: int = 1
    """Radius added around each node before routing; reserves area for local wiring."""

    toric_code: Optional[dict] = None
    """Whether code being layed out is a toric code or not."""

    # === Routing Settings ===
    max_bump_transitions_per_coupler: Optional[int] = 10
    """Maximum bump transitions per connection. When violated, edge pops to next tier."""

    max_tsvs_per_coupler: Optional[int] = None
    """Maximum through-silicon vias per connection. When violated, routing fails."""

    max_coupler_length: Optional[int] = 1000
    """Maximum connection length between nodes in units of smallest connection length between
        nodes. When violated, pops to next tier."""

    route_edge_order: str = "length_asc"
    """Edge ordering for routing: 'crossings_asc', 'length_desc', etc."""

    edge_expansion_val: int = 1
    """Safety margin (in grid cells) around every routed trace."""

    # === Output and Debugging ===
    save: bool = True
    """Whether to save intermediate results."""

    verbose: bool = False
    """Enable verbose output during placement and routing."""

    # === Benchmarking ===
    baseline_defaults: dict = field(
        default_factory=lambda: {
            "num_tiers": 1.0,
            "avg_coupler_length": 1.0,
            "max_avg_face_switches": 0.0,
            "avg_tsvs_per_edge": 0.0,
        }
    )
    """Baseline values for benchmark scoring (score = 1)."""

    bad_defaults: dict = field(
        default_factory=lambda: {
            "num_tiers": 5.0,
            "avg_coupler_length": 10.0,
            "max_avg_face_switches": 4.0,
            "avg_tsvs_per_edge": 3.0,
        }
    )
    """Bad threshold values for benchmark scoring (score = 2)."""

    def __post_init__(self):
        """Set derived values after initialization and validate benchmark dictionaries."""
        # Set grid dimensions if not explicitly provided
        if self.grid_size_x is None:
            self.grid_size_x = self.grid_size
        if self.grid_size_y is None:
            self.grid_size_y = self.grid_size

        # Validate benchmark dictionaries have required keys
        required_benchmark_keys = {
            "num_tiers",
            "avg_coupler_length",
            "max_avg_face_switches",
            "avg_tsvs_per_edge",
        }

        baseline_keys = set(self.baseline_defaults.keys())
        bad_keys = set(self.bad_defaults.keys())

        if baseline_keys != required_benchmark_keys:
            missing_baseline = required_benchmark_keys - baseline_keys
            extra_baseline = baseline_keys - required_benchmark_keys
            raise ValueError(
                f"baseline_defaults must contain exactly these keys: {required_benchmark_keys}. "
                f"Missing: {missing_baseline}, Extra: {extra_baseline}"
            )

        if bad_keys != required_benchmark_keys:
            missing_bad = required_benchmark_keys - bad_keys
            extra_bad = bad_keys - required_benchmark_keys
            raise ValueError(
                f"bad_defaults must contain exactly these keys: {required_benchmark_keys}. "
                f"Missing: {missing_bad}, Extra: {extra_bad}"
            )

    def save_to_json(self, filepath: str | Path) -> None:
        """Saves the current settings to a JSON file.

        This method converts the dataclass instance into a dictionary and
        writes it to the specified file path in a human-readable format.

        Args:
            filepath: The path (as a string or Path object) to the output JSON file.
        """

        # Convert the dataclass instance to a dictionary for JSON serialization
        settings_dict = asdict(self)

        with open(filepath, "w") as f:
            json.dump(settings_dict, f, indent=4)

    @classmethod
    def load_from_json(cls, filepath: str | Path) -> "Settings":
        """Loads settings from a JSON file.

        This class method reads a JSON file, and creates a new instance
        of the Settings class from its contents.

        Args:
            filepath: The path (as a string or Path object) to the input JSON file.

        Returns:
            A new instance of the Settings class.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        if "custom_positions" in data and data["custom_positions"] is not None:
            data["custom_positions"] = {int(k): v for k, v in data["custom_positions"].items()}

        return cls(**data)

    def summary(self) -> str:
        """Return a concise summary of key settings."""
        summary_parts = [
            f"Grid: {self.grid_size_x}×{self.grid_size_y}",
            f"Layout: {self.layout}",
            f"Node expansion: {self.node_expansion_val}",
            f"Edge expansion: {self.edge_expansion_val}",
        ]

        if self.max_bump_transitions_per_coupler is not None:
            summary_parts.append(f"Max bumps: {self.max_bump_transitions_per_coupler}")
        if self.max_tsvs_per_coupler is not None:
            summary_parts.append(f"Max TSVs: {self.max_tsvs_per_coupler}")
        if self.max_coupler_length is not None:
            summary_parts.append(f"Max length: {self.max_coupler_length}")
        if self.custom_positions is not None:
            summary_parts.append(f"Custom positions: {len(self.custom_positions)} nodes")

        return " | ".join(summary_parts)

    def operation_summary(self, operation: str) -> str:
        """Return a summary of settings relevant to a specific operation.

        Args:
            operation: One of 'place', 'route', or 'benchmark'

        Returns:
            Formatted string showing relevant settings for the operation
        """
        if operation == "place":
            relevant = [
                f"Grid: {self.grid_size_x}×{self.grid_size_y}",
                f"Layout: {self.layout}",
                f"Node expansion: {self.node_expansion_val}",
                f"Place margin: {self.place_margin}",
            ]
            if self.custom_positions is not None:
                relevant.append(f"Custom positions: {len(self.custom_positions)} nodes")
            if self.toric_code is not None:
                relevant.append("Toric code layout enabled")

        elif operation == "route":
            relevant = [
                f"Edge expansion: {self.edge_expansion_val}",
                f"Route order: {self.route_edge_order}",
            ]
            if self.max_bump_transitions_per_coupler is not None:
                relevant.append(f"Max bumps: {self.max_bump_transitions_per_coupler}")
            if self.max_tsvs_per_coupler is not None:
                relevant.append(f"Max TSVs: {self.max_tsvs_per_coupler}")
            if self.max_coupler_length is not None:
                relevant.append(f"Max length: {self.max_coupler_length}")

        elif operation == "benchmark":
            relevant = [
                f"Baseline thresholds: {list(self.baseline_defaults.keys())}",
                f"Bad thresholds: {list(self.bad_defaults.keys())}",
            ]
        else:
            relevant = [f"Unknown operation: {operation}"]

        return " | ".join(relevant)

    def show_overrides(self, **kwargs) -> str:
        """Show which settings are being overridden by method parameters.

        Args:
            **kwargs: Method parameters that might override settings

        Returns:
            Formatted string showing what's being overridden
        """
        overrides = []

        # Check placement parameters
        if "grid_size" in kwargs and kwargs["grid_size"] != self.grid_size:
            overrides.append(f"grid_size: {self.grid_size} → {kwargs['grid_size']}")
        if "layout" in kwargs and kwargs["layout"] != self.layout:
            overrides.append(f"layout: {self.layout} → {kwargs['layout']}")
        if (
            "node_expansion_val" in kwargs
            and kwargs["node_expansion_val"] != self.node_expansion_val
        ):
            overrides.append(
                f"node_expansion_val: {self.node_expansion_val} → {kwargs['node_expansion_val']}"
            )
        if "margin" in kwargs and kwargs["margin"] != self.place_margin:
            overrides.append(f"place_margin: {self.place_margin} → {kwargs['margin']}")

        # Check routing parameters
        if (
            "max_bump_transitions_per_coupler" in kwargs
            and kwargs["max_bump_transitions_per_coupler"] != self.max_bump_transitions_per_coupler
        ):
            overrides.append(
                f"max_bump_transitions_per_coupler: {self.max_bump_transitions_per_coupler} → {kwargs['max_bump_transitions_per_coupler']}"
            )
        if (
            "max_tsvs_per_coupler" in kwargs
            and kwargs["max_tsvs_per_coupler"] != self.max_tsvs_per_coupler
        ):
            overrides.append(
                f"max_tsvs_per_coupler: {self.max_tsvs_per_coupler} → {kwargs['max_tsvs_per_coupler']}"
            )
        if (
            "edge_expansion_val" in kwargs
            and kwargs["edge_expansion_val"] != self.edge_expansion_val
        ):
            overrides.append(
                f"edge_expansion_val: {self.edge_expansion_val} → {kwargs['edge_expansion_val']}"
            )

        if not overrides:
            return "No settings overridden by method parameters"

        return " | ".join(overrides)

    def __str__(self) -> str:
        """Return a user-friendly string representation."""
        return f"Settings({self.summary()})"

    def __repr__(self) -> str:
        """Return a detailed representation for debugging."""
        return (
            f"Settings(\n"
            f"  # Placement\n"
            f"  grid_size={self.grid_size}, grid_size_x={self.grid_size_x}, "
            f"grid_size_y={self.grid_size_y}\n"
            f"  custom_positions={self.custom_positions}, layout='{self.layout}', "
            f"mps_edge_order='{self.mps_edge_order}'\n"
            f"  place_margin={self.place_margin}, node_expansion_val={self.node_expansion_val}\n"
            f"  toric_code={self.toric_code}\n"
            f"  # Routing\n"
            f"  max_bump_transitions_per_coupler={self.max_bump_transitions_per_coupler}, "
            f"max_tsvs_per_coupler={self.max_tsvs_per_coupler}\n"
            f"  max_coupler_length={self.max_coupler_length}, "
            f"route_edge_order='{self.route_edge_order}'\n"
            f"  edge_expansion_val={self.edge_expansion_val}\n"
            f"  # Output\n"
            f"  save={self.save}, verbose={self.verbose}\n"
            f"  # Benchmark\n"
            f"  baseline_defaults={self.baseline_defaults}\n"
            f"  bad_defaults={self.bad_defaults}\n"
            f")"
        )
