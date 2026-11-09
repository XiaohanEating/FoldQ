from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from netgraph import InteractiveGraph

# Assuming this utility function exists in your project structure
from hal.utils.disk_utils import save_pdf_multipage

# Use TYPE_CHECKING to import the class for type hints, avoiding circular dependencies
if TYPE_CHECKING:
    from hal.hal import HardwareAwareLayout


def display_grids(
    hal: "HardwareAwareLayout",
    save: bool = False,
    grid_inds: Optional[list] = None,
    path: Optional[str] = None,
    dpi: Optional[int] = None,
) -> None:
    """Displays and optionally saves grid images for each tier."""
    cmap = colors.ListedColormap(["white", "red", "blue", "green", "black"])
    bounds = [-1, 1, 3, 5, 7, 11]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    path_to_directory = Path(hal.path_to_directory if path is None else path)

    if grid_inds is None:
        grid_inds = range(len(hal.tiers))

    dpi = dpi if dpi is not None else 200

    for grid_ind in grid_inds:
        fig = plt.figure(f"grid_{grid_ind}", dpi=dpi)
        sum_grid = np.sum(hal.tiers[grid_ind].grid, axis=2)
        img = plt.imshow(sum_grid, cmap=cmap, norm=norm)
        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 2, 4, 6, 10])
        if save:
            fig.savefig(str(path_to_directory / f"tier_grid_{grid_ind}.png"), dpi=800)

    if save:
        save_pdf_multipage(str(path_to_directory / "tier_grids.pdf"), dpi=800)
        plt.close("all")


def display_line_plot(hal: "HardwareAwareLayout") -> None:
    """Plots nodes and routed edges as a static line plot for each tier."""
    for i, tier in enumerate(hal.tiers):
        plt.figure(f"line_plot_{i}")
        positions = nx.get_node_attributes(tier.G, "pos")
        if not positions:
            continue

        xs, ys = zip(*positions.values())
        plt.scatter(xs, ys)

        for edge in tier.G.edges:
            route = tier.G.edges[edge].get("route")
            if not route:
                print(f"Missing route for edge: {edge} on tier {i}")
                continue

            route_xs, route_ys, _ = zip(*route)
            plt.plot(route_xs, route_ys)

    plt.show()


def display(
    hal: "HardwareAwareLayout",
    save: bool = False,
    path: Optional[str] = None,
    tiers: Optional[list] = None,
) -> tuple[list, list]:
    """Displays an interactive graph of the mapping for each tier."""
    plots, figs = [], []
    path_to_directory = Path(hal.path_to_directory if path is None else path)
    tier_indices = tiers if tiers is not None else range(len(hal.tiers))

    for tier_ind in tier_indices:
        tier = hal.tiers[tier_ind]
        fig = plt.figure(f"interactive_tier_{tier_ind}")
        figs.append(fig)

        combined_nodes = {}
        normal_nodes = {}
        bend_nodes = {}
        switch_nodes = {}
        node_shape = {}
        node_size = {}
        node_edge_width = {}
        node_color = {}
        node_alpha = {}
        node_zorder = {}

        for node in tier.G.nodes:
            pos = tier.G.nodes[node]["pos"]
            if isinstance(node, tuple):
                if node[0] == "bend":
                    bend_nodes[f"bend{node[1]}"] = pos
                    node_shape[f"bend{node[1]}"] = "o"
                    node_size[f"bend{node[1]}"] = 10
                    node_edge_width[f"bend{node[1]}"] = 10
                    node_zorder[f"bend{node[1]}"] = 100
                    node_color[f"bend{node[1]}"] = "g"
                    node_alpha[f"bend{node[1]}"] = 1
                elif node[0] == "switch":
                    # if node[1] not in switch_nodes:
                    if not any(edge for edge in tier.routed_edge_order if node in edge):
                        continue
                    switch_nodes[f"switch{node[1]}"] = pos
                    node_shape[f"switch{node[1]}"] = "s"
                    node_size[f"switch{node[1]}"] = 200
                    node_edge_width[f"switch{node[1]}"] = 0
                    node_zorder[f"switch{node[1]}"] = 200
                    node_color[f"switch{node[1]}"] = "#92D050"
                    node_alpha[f"switch{node[1]}"] = 1
            else:
                normal_nodes[node] = pos
                node_edge_width[node] = 0
                node_zorder[node] = 100
                node_shape[node] = "o"
                if tier_ind > 0:
                    node_size[node] = 250

                    node_on_tier = False
                    for edge in tier.G.edges:
                        if node in edge:
                            node_alpha[node] = 1
                            node_on_tier = True
                            node_color[node] = "#C00000"
                            break

                    if not node_on_tier:
                        node_size[node] = 450
                        node_color[node] = "#0070C0"
                        node_alpha[node] = 0

                else:
                    node_size[node] = 450
                    node_alpha[node] = 1
                    node_color[node] = "#0070C0"

        combined_nodes = {**normal_nodes, **bend_nodes, **switch_nodes}
        edge_list = []
        edge_paths = {}
        edge_colors = {}
        edge_width = {}
        edge_zorder = {}
        mapping = {}

        # tier.routed_edge_order.reverse()
        if tier.routed_edge_order is None:
            continue

        for edge in tier.routed_edge_order:
            edge0 = edge[0] if not isinstance(edge[0], tuple) else f"{edge[0][0]}{edge[0][1]}"
            edge1 = edge[1] if not isinstance(edge[1], tuple) else f"{edge[1][0]}{edge[1][1]}"
            edge_list.append((edge0, edge1))
            full_edge = tier.G.edges[edge[0], edge[1]]["full_edge"]
            mapping.setdefault(full_edge, []).append((edge0, edge1))
            route = tier.G.edges[edge[0], edge[1]].get("route", [])
            xs = [pt[0] for pt in route]
            ys = [pt[1] for pt in route]
            xs.reverse()
            ys.reverse()
            edge_paths[(edge0, edge1)] = np.array(list(zip(xs, ys)))
            edge_colors[(edge0, edge1)] = (
                "k" if route and route[len(route) // 2][2] == 0 else "#E57C09"
            )
            edge_width[(edge0, edge1)] = 150
            edge_zorder[(edge0, edge1)] = 50

        plot = InteractiveGraph(
            edge_list,
            nodes=list(combined_nodes.keys()),
            node_layout=combined_nodes,
            node_size=node_size,
            node_shape=node_shape,
            node_edge_width=node_edge_width,
            node_color=node_color,
            node_alpha=node_alpha,
            node_zorder=node_zorder,
            edge_layout=edge_paths,
            edge_color=edge_colors,
            edge_width=edge_width,
            edge_zorder=edge_zorder,
            edge_alpha=1,
            # check_mouseover_highlight_mapping=mapping,
            node_labels=False,
        )
        plots.append(plot)

        if save:
            # Ensure the directory for saving plots exists
            os.makedirs(path_to_directory, exist_ok=True)
            fig.savefig(str(path_to_directory / f"tier_interactive_{tier_ind}.png"), dpi=800)
            fig.savefig(
                str(path_to_directory / f"tier_interactive_{tier_ind}.svg"),
                format="svg",
            )

    if not save:
        plt.show()
    else:
        plt.close("all")

    return figs, plots


def display_gif(
    hal: "HardwareAwareLayout",
    tier_ind: int,
    save: bool = False,
    path: Optional[str] = None,
    fps: int = 1,
    step: int = 1,
) -> Optional[FuncAnimation]:
    """Creates an animated GIF of the routing process for a specific tier."""
    # This function now calls the standalone `display` utility
    figs, plots = display(hal, tiers=[tier_ind], save=False)

    if not plots:
        print(f"Could not generate plot for tier {tier_ind}. Aborting GIF creation.")
        return None

    plot = plots[0]
    fig = figs[0]
    edges = hal.tiers[tier_ind].routed_edge_order

    if not edges:
        print(f"No routed edges found for tier {tier_ind}. Nothing to animate.")
        return None

    total_frames = len(edges) // step
    path_to_directory = Path(hal.path_to_directory if path is None else path)

    def update(ii):
        ii = ii * step
        for i in range(ii + 1):
            edge = edges[i]
            edge0 = edge[0] if not isinstance(edge[0], tuple) else f"{edge[0][0]}{edge[0][1]}"
            edge1 = edge[1] if not isinstance(edge[1], tuple) else f"{edge[1][0]}{edge[1][1]}"
            try:
                artist = plot.edge_artists[(edge0, edge1)]
            except KeyError:
                artist = plot.edge_artists[(edge1, edge0)]
            artist.set_visible(True)

        for j in range(ii + 1, total_frames):
            edge = edges[j]
            edge0 = edge[0] if not isinstance(edge[0], tuple) else f"{edge[0][0]}{edge[0][1]}"
            edge1 = edge[1] if not isinstance(edge[1], tuple) else f"{edge[1][0]}{edge[1][1]}"
            try:
                artist = plot.edge_artists[(edge0, edge1)]
            except KeyError:
                artist = plot.edge_artists[(edge1, edge0)]
            artist.set_visible(False)

        return plot.edge_artists.values()

    animation = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=True)

    if save:
        os.makedirs(path_to_directory, exist_ok=True)
        animation.save(
            str(path_to_directory / f"tier_{tier_ind}_routing.gif"),
            dpi=200,
            writer=PillowWriter(fps=fps),
        )
        plt.close(fig)
        return None

    return animation


def plot_sweep_results(results_df: pd.DataFrame, sweep_params: dict):
    """
    Visualizes the impact of each swept parameter on hardware complexity.

    This function generates a grid of box plots, with one plot for each
    parameter in the sweep, showing its effect on the hardware_complexity score.

    Args:
        results_df: The DataFrame returned by the benchmark_sweep method.
        sweep_params: The dictionary of parameters that was used to run the sweep.
    """
    if results_df.empty:
        print("DataFrame is empty. Nothing to plot. 🤷")
        return

    param_names = list(sweep_params.keys())
    num_params = len(param_names)

    if num_params == 0:
        print("No sweep parameters found to plot.")
        return

    # --- Set up the plot aesthetics ---
    sns.set_theme(style="whitegrid", palette="viridis")

    # --- Create a grid of subplots for each parameter ---
    # We'll use a two-column layout for the plots
    cols = 2 if num_params > 1 else 1
    rows = math.ceil(num_params / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 6), squeeze=False)
    axes = axes.flatten()  # Flatten for easy iteration

    print(f"📊 Generating {num_params} plots to analyze the sweep results...")

    # --- Generate a box plot for each swept parameter ---
    for i, param_name in enumerate(param_names):
        ax = axes[i]

        # Sort the categories for a cleaner plot (especially for numerical parameters)
        sorted_categories = sorted(results_df[param_name].unique())

        sns.boxplot(
            data=results_df,
            x=param_name,
            y="hardware_complexity",
            ax=ax,
            order=sorted_categories,
        )

        ax.set_title(f"Impact of '{param_name}'", fontsize=15, weight="bold")
        ax.set_xlabel(param_name.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Hardware Complexity Score", fontsize=12)
        ax.tick_params(axis="x", labelrotation=15)

    # --- Final Touches ---
    # Hide any unused subplots in the grid
    for j in range(num_params, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Sweep Analysis: Parameter Impact on Hardware Complexity",
        fontsize=20,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.show()
