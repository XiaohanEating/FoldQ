import ast
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def create_path(directory_path: str, measurement_name: str, create_directory: bool = False) -> Path:
    """Create a dated output directory for a measurement.

    The directory will be named like YYYYMMDD_HhMmSs_<measurement_name>
    and will contain `layers` and `grid_view` subfolders.
    """
    datetime_now = datetime.now()

    str_ymd = datetime_now.strftime("%Y%m%d_%Hh%Mm%Ss")
    str_ymd_measurement_name = f"{str_ymd}_{measurement_name}"

    path_to_experiment = Path(directory_path).joinpath(f"{str_ymd_measurement_name}")

    Path.mkdir(path_to_experiment, parents=True, exist_ok=True)

    Path.mkdir(path_to_experiment / "layers", exist_ok=True)
    Path.mkdir(path_to_experiment / "grid_view", exist_ok=True)

    return path_to_experiment


def save_grid(grid: np.ndarray, name: str, path: str) -> None:
    with open(f"{path}/{name}", "wb") as fh:
        pickle.dump(grid, fh)


def load_grid(path_to_file: str) -> np.ndarray:
    with open(path_to_file, "rb") as fh:
        return pickle.load(fh)


def save_graph(graph: nx.Graph, name: str, path: str) -> None:
    with open(f"{path}/{name}", "wb") as fh:
        pickle.dump(graph, fh)


def load_graph(path_to_file: str) -> nx.Graph:
    with open(path_to_file, "rb") as fh:
        return pickle.load(fh)


def save_pdf_multipage(
    filename: str | Path, figs: Sequence[plt.Figure] | None = None, dpi: int = 800
) -> None:
    with PdfPages(str(filename)) as pp:
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            pp.savefig(fig)
    plt.close("all")


def load_parity_check_matrices(
    root_dir: str,
) -> list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load parity check matrices from subfolders named like 'n_k_d' under root_dir.

    Each folder should contain 'hx.csv' and 'hz.csv'. The function loads these CSV files
    and returns a list of tuples: (n, k, d, hx, hz, parity_matrix).
    """
    matrices = []

    # List all entries in the given directory
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        # Check if the entry is a directory
        if os.path.isdir(folder_path):
            parts = folder.split("_")
            # Ensure the folder name has exactly three parts
            if len(parts) != 3:
                continue
            try:
                # Convert each part to an integer: n, k, d
                n, k, d = map(int, parts)
            except ValueError:
                # Skip folders that do not have integer parameters
                continue

            # Construct file paths for hx.csv and hz.csv
            hx_path = os.path.join(folder_path, "hx.csv")
            hz_path = os.path.join(folder_path, "hz.csv")

            # Load CSV files into numpy arrays (assumes comma-separated values)
            try:
                hx = np.loadtxt(hx_path, delimiter=",")
                hz = np.loadtxt(hz_path, delimiter=",")
            except Exception as e:
                print(f"Error loading files in {folder_path}: {e}")
                continue

            # Vertically stack the hx and hz matrices
            parity_matrix = np.vstack((hx, hz))
            # Append the parameters and the matrix as a tuple to the results list
            matrices.append((n, k, d, hx, hz, parity_matrix))

    return matrices


def create_tuple(tuple_str: str) -> tuple:
    """Convert string representation of a tuple to an actual tuple using ast.literal_eval."""
    return ast.literal_eval(tuple_str)


def extract_code_parameters(csv_file_path: str) -> list[dict[str, Any]]:
    """Extract code parameters from a CSV file used for website/demo generation."""
    df = pd.read_csv(csv_file_path)

    code_params_list = []

    for index, row in df.iterrows():
        code_params = {
            "name": row["[[n, k, d]]"],
            "a1": create_tuple(row["a1"]),
            "a2": create_tuple(row["a2"]),
            "a_coeffs": (int(row["f_x_exp"]), int(row["f_y_exp"])),
            "b_coeffs": (int(row["g_x_exp"]), int(row["g_y_exp"])),
        }

        code_params_list.append(code_params)

    return code_params_list


def process_single_code(code_params: dict[str, Any], x: Any, y: Any) -> dict[str, Any]:
    """Construct polynomials for a single code parameter set.

    Returns a dict with name, lattice vectors, coefficient tuples, and polynomials poly_a, poly_b.
    """
    name = code_params["name"]
    a1 = code_params["a1"]
    a2 = code_params["a2"]
    a_coeffs = code_params["a_coeffs"]
    b_coeffs = code_params["b_coeffs"]

    # Now you can easily construct the polynomials
    poly_a = 1 + x + x ** a_coeffs[0] * y ** a_coeffs[1]
    poly_b = 1 + y + x ** b_coeffs[0] * y ** b_coeffs[1]

    return {
        "name": name,
        "a1": a1,
        "a2": a2,
        "a_coeffs": a_coeffs,
        "b_coeffs": b_coeffs,
        "poly_a": poly_a,
        "poly_b": poly_b,
    }
