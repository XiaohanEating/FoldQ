import itertools
import math
import random
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
from community import community_louvain
from qldpc.codes.common import CSSCode
from tqdm import tqdm

from hal.utils.grid_utils import distance


##################--------- Edge Intersection  -------######################
def ccw(A: tuple[float, float], B: tuple[float, float], C: tuple[float, float]) -> bool:
    """Return True if points A, B, C are in counterclockwise order.

    Args:
        A (tuple[float, float]): First point.
        B (tuple[float, float]): Second point.
        C (tuple[float, float]): Third point.

    Returns:
        bool: True if the points are in counterclockwise order.
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(
    A: tuple[float, float], B: tuple[float, float], C: tuple[float, float], D: tuple[float, float]
) -> bool:
    """Return True if segments AB and CD intersect.

    Args:
        A, B, C, D (tuple[float, float]): Segment endpoints.

    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def compute_crossing_pairs(
    G: nx.Graph,
) -> tuple[list[tuple], list[tuple[tuple, tuple]]]:
    """
    Given a graph G with node positions, compute a list of edge pairs
    that cross. Only consider pairs of edges that do not share a vertex.

    Returns:
      crossing_pairs: a list of tuples (e, f) where each e and f are edges
      represented as tuples of nodes (with nodes in sorted order).
    """
    # Represent each edge as a sorted tuple to be consistent
    edges = [tuple(sorted(e)) for e in G.edges()]
    crossing_pairs = []
    # Check every unordered pair of edges
    for e1, e2 in itertools.combinations(edges, 2):
        # Skip if they share a vertex
        if set(e1) & set(e2):
            continue

        # Get the coordinates for the endpoints
        p1, q1 = G.nodes[e1[0]]["pos"], G.nodes[e1[1]]["pos"]
        p2, q2 = G.nodes[e2[0]]["pos"], G.nodes[e2[1]]["pos"]

        if intersect(p1, q1, p2, q2):
            crossing_pairs.append((e1, e2))
    return edges, crossing_pairs


##################--------- Orthogonalize  -------######################
def remove_extra_edges(G: nx.Graph) -> list[tuple]:
    """Heuristically reduce node degrees to at most 4 by removing edges.

    Args:
        G (nx.Graph): Graph modified in-place.

    Returns:
        list[tuple]: Edges removed.
    """
    removed_edges = []

    nodes_4_plus_edges = {k: v for (k, v) in dict(G.degree).items() if v > 4}
    nodes_4_plus_edges_sorted = dict(sorted(nodes_4_plus_edges.items(), key=lambda item: item[1]))

    while True:
        node_list = list(nodes_4_plus_edges_sorted.keys())
        if len(node_list) == 0:
            break

        node = node_list[0]

        while G.degree[node] > 4:
            try:
                edge_to_remove = list(G.edges(node))[0]
            except IndexError as exc:
                print(G.nodes)
                raise ValueError(
                    f"Graph has {node} with degree > 4 but no edges to remove."
                ) from exc
            G.remove_edge(edge_to_remove[0], edge_to_remove[1])
            removed_edges.append(edge_to_remove)

        del nodes_4_plus_edges_sorted[node]

    return removed_edges


##################--------- Thickness  -------######################
def thickness(
    G: nx.Graph, ordered_edges: list[tuple], print_progress: bool = True
) -> list[nx.Graph]:
    """Partition edges into planar layers by greedy insertion order.

    Args:
        G (nx.Graph): Original graph (nodes copied to each layer).
        ordered_edges (list[tuple]): Edge insertion order.
        print_progress (bool): Show a progress bar.

    Returns:
        list[nx.Graph]: List of planar subgraphs (layers).
    """
    drawn_edges = []
    layers = []
    progress_bar = None
    if print_progress:
        progress_bar = tqdm(total=len(ordered_edges), desc="- Extract base layer edges")

    while True:
        layer = G.copy()
        layer.remove_edges_from(G.edges())

        for edge in ordered_edges:
            layer.add_edge(edge[0], edge[1])

            if not nx.is_planar(layer):
                layer.remove_edge(edge[0], edge[1])
            else:
                drawn_edges.append(edge)
                if progress_bar is not None:
                    progress_bar.update(1)

        if len(ordered_edges) == 0:
            break
        layers.append(layer)
        for e in drawn_edges:
            ordered_edges.remove(e)
        drawn_edges = []

    if progress_bar is not None:
        progress_bar.close()

    return layers


def greedy_mps_random(G: nx.Graph, seed: int | None = None) -> tuple[nx.Graph, nx.Graph]:
    """
    Greedy maximum planar subgraph extraction with randomized ordering of remaining edges.

    Parameters:
      - G: The original graph.
      - seed: Optional seed for reproducibility.

    Returns:
      - P: A planar subgraph of G.
    """
    if seed is not None:
        random.seed(seed)

    # Initialize the planar subgraph with all nodes
    P = nx.Graph()
    P.add_nodes_from(G.nodes())

    # Start with a spanning tree (planar by definition)
    spanning_tree = nx.minimum_spanning_tree(G)
    P.add_edges_from(spanning_tree.edges())

    # Get remaining edges (those not in the spanning tree)
    remaining_edges = list(set(G.edges()) - set(spanning_tree.edges()))
    # Randomize the order of the remaining edges
    random.shuffle(remaining_edges)

    # Greedily add edges while preserving planarity
    for edge in remaining_edges:
        u, v = edge
        P.add_edge(u, v)
        is_planar, _ = nx.check_planarity(P, counterexample=False)
        if not is_planar:
            P.remove_edge(u, v)

    # The rejected edges are those in G that are not in P.
    rejected_edges = set(G.edges()) - set(P.edges())
    R = nx.Graph()
    R.add_nodes_from(G.nodes())
    R.add_edges_from(rejected_edges)

    return P, R


def best_greedy_mps(
    G: nx.Graph, num_iter: int = 100, seed: int | None = None
) -> tuple[nx.Graph, nx.Graph]:
    """
    Run the randomized greedy MPS extraction multiple times and return the best result.

    Parameters:
      - G: The original graph.
      - num_iter: Number of iterations to perform.
      - seed: Optional seed for reproducibility.

    Returns:
      - best_P: The planar subgraph with the maximum number of edges found.
    """
    best_P = None
    best_edge_count = 0
    remaining_subgraph = None
    for i in range(num_iter):
        # Adjust seed per iteration if seed is provided
        current_seed = seed + i if seed is not None else None
        P, R = greedy_mps_random(G, seed=current_seed)
        if P.number_of_edges() > best_edge_count:
            best_edge_count = P.number_of_edges()
            best_P = P
            remaining_subgraph = R
    return best_P, remaining_subgraph


##################--------- Ordered Edges  -------######################


def get_ordered_edges(
    G: nx.Graph, edge_metric: str, edge_weights: dict | None = None
) -> list[tuple]:
    """
    Calculates a metric for each edge and returns a sorted list of edges.

    New metrics added:
    - "betweenness": Edge betweenness centrality. Measures the edge's importance
      as a bridge in the graph's shortest paths.
    - "sum_degrees": The sum of the degrees of an edge's two endpoints. A
      heuristic for routing in congested areas first.
    - "angle": The geometric angle of the edge. Useful for routing in a
      systematic sweeping order.
    """
    if edge_weights is None:
        edge_weights = {}

    if "random" in edge_metric:
        # Shuffle the edges and return
        edges = list(G.edges(data=True))
        random.shuffle(edges)
        return edges

    # --- Graph-wide pre-computation for expensive metrics ---
    if "betweenness" in edge_metric:
        # This is calculated once for the entire graph for efficiency.
        centrality = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
        nx.set_edge_attributes(G, centrality, edge_metric)

    # --- Per-edge calculations ---
    else:
        for edge in G.edges():
            source, target = edge
            source_pt = G.nodes[source]["pos"]
            target_pt = G.nodes[target]["pos"]

            if "custom_weights" in edge_metric:
                # Assigns a pre-defined weight to the edge.
                # (Original logic was corrected to assign per-edge)
                G.edges[edge][edge_metric] = edge_weights.get(edge) or edge_weights.get(
                    (target, source)
                )

            elif "crossings" in edge_metric:
                crossings = 0
                for other_edge in G.edges():
                    if edge == other_edge:
                        continue
                    other_source, other_target = other_edge
                    # Avoid checking crossings between edges sharing a vertex
                    if source in other_edge or target in other_edge:
                        continue

                    other_source_pt = G.nodes[other_source]["pos"]
                    other_target_pt = G.nodes[other_target]["pos"]

                    if intersect(source_pt, target_pt, other_source_pt, other_target_pt):
                        crossings += 1
                # Each crossing is found twice, so we divide by 2
                G.edges[edge][edge_metric] = crossings / 2

            elif "length" in edge_metric:
                length = distance(source_pt, target_pt)
                G.edges[edge][edge_metric] = length

            elif "closeness" in edge_metric:
                dist_to_middle_1 = distance((0, 0), target_pt)
                dist_to_middle_2 = distance((0, 0), source_pt)
                closeness_to_middle = min(dist_to_middle_1, dist_to_middle_2)
                G.edges[edge][edge_metric] = closeness_to_middle

            elif "length_orthogonal_first" in edge_metric:
                length = distance(source_pt, target_pt)
                orthogonal = source_pt[0] == target_pt[0] or source_pt[1] == target_pt[1]
                G.edges[edge][edge_metric] = (0, length) if orthogonal else (1, length)

            elif "sum_degrees" in edge_metric:
                degree_sum = G.degree(source) + G.degree(target)
                G.edges[edge][edge_metric] = degree_sum

            elif "angle" in edge_metric:
                delta_y = target_pt[1] - source_pt[1]
                delta_x = target_pt[0] - source_pt[0]
                angle = math.atan2(delta_y, delta_x)  # Radians from -pi to pi
                G.edges[edge][edge_metric] = angle

    # --- Sorting and Return ---
    def key_func(edge):
        return edge[2].get(edge_metric, 0)

    if "desc" in edge_metric:
        return sorted(G.edges(data=True), key=key_func, reverse=True)
    return sorted(G.edges(data=True), key=key_func, reverse=False)


##################--------- Community Layout -------######################
def community_layout(g: nx.Graph) -> dict:
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """
    partition = community_louvain.best_partition(g)
    pos_communities = _position_communities(g, partition, scale=3.0)

    pos_nodes = _position_nodes(g, partition, scale=1.0)

    pos = {}
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(g: nx.Graph, partition: dict, **kwargs) -> dict:

    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    pos = {}
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _find_between_community_edges(g: nx.Graph, partition: dict) -> dict:

    edges = {}

    for ni, nj in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            edges.setdefault((ci, cj), []).append((ni, nj))

    return edges


def _position_nodes(g: nx.Graph, partition: dict, **kwargs) -> dict:
    """
    Positions nodes within communities.
    """

    communities = {}
    for node, community in partition.items():
        communities.setdefault(community, []).append(node)

    pos = {}
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


##################--------- Edge Coloring -------######################


def generate_tanner_graph(parity_check_matrix: np.ndarray) -> nx.Graph:
    """
    Generate a Tanner graph from a parity check matrix.

    Args:
        parity_check_matrix (np.ndarray): Parity check matrix. Assumes he parity check matrix is structured as np.vstack((Hx, Hz)),

    Returns:
        nx.Graph: A bipartite Tanner graph.
    """
    hx, hz = parity_check_matrix_to_hx_hz(parity_check_matrix)
    code = CSSCode(code_x=hx.astype(int), code_z=hz.astype(int))

    # Iterate and clear attributes
    for _, _, data in code.graph.edges(data=True):
        data.clear()

    return code.graph


def tanner_graph_to_parity_check_matrix(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct Hx and Hz parity check matrices from a Tanner graph,
    assuming the following node order:
        - First quarter: Hx check nodes
        - Second quarter: Hz check nodes
        - Third quarter: variable nodes for Hx
        - Fourth quarter: variable nodes for Hz

    Args:
        G (nx.Graph): The Tanner graph.

    Returns:
        np.ndarray : Parity check matrix; vstack of (Hx, Hz).
    """
    sorted_nodes = sorted(G.nodes, key=lambda node: node.index)
    check_nodes = [node for node in sorted_nodes if node.is_data is False]
    data_nodes = [node for node in sorted_nodes if node.is_data is True]
    n = len(sorted_nodes) // 2

    x_check_nodes = check_nodes[: n // 2]
    z_check_nodes = check_nodes[n // 2 : n]

    Hx = np.zeros((n // 2, n), dtype=int)
    Hz = np.zeros((n // 2, n), dtype=int)

    for x_node in x_check_nodes:
        for d_node in G.neighbors(x_node):
            row = x_node.index
            col = d_node.index
            Hx[row, col] = 1

    for z_node in z_check_nodes:
        for d_node in G.neighbors(z_node):
            row = z_node.index - n // 2
            col = d_node.index
            Hz[row, col] = 1

    return np.vstack((Hx, Hz))


def parity_check_matrix_to_hx_hz(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a parity check matrix into Hx and Hz components.

    Args:
        P (np.ndarray): The parity check matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The Hx and Hz matrices.
    """
    n = P.shape[0]
    Hx = P[: n // 2, :]
    Hz = P[n // 2 :, :]
    return Hx, Hz


def dash_print(message: str, string_length: int = 50, char: str = "-") -> None:
    """
    Print a message with dashes around it for emphasis.

    Args:
        message (str): The message to print.
        string_length (int): The total length of the printed line, including dashes.
    """
    num_dashes = (string_length - len(message)) // 2
    print(char * num_dashes + message + char * num_dashes)
