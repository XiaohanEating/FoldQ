from typing import Iterable, Tuple

import numpy as np


def generate_grid(
    x_list: Iterable[float],
    y_list: Iterable[float],
    num_faces: int = 2,
    grid_size: int = 500,
    margin: float = 0.2,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Create a grid and mark given (x, y) points on it.

    Args:
        x_list, y_list: Coordinates to mark.
        num_faces (int): Number of z-faces in the grid.
        grid_size (int): Size of grid in each axis.
        margin (float): Fractional margin used for spacing.

    Returns:
        tuple[np.ndarray, float, np.ndarray]: The grid, margin size, and grid spacing.
    """
    grid = np.zeros((grid_size, grid_size, num_faces))
    margin_size = grid_size * margin

    x_val_min = int(np.min(x_list))
    y_val_min = int(np.min(y_list))
    x_val_max = int(np.max(x_list))
    y_val_max = int(np.max(y_list))
    grid_spacing = np.array(
        [
            np.abs(x_val_max - x_val_min) / (grid_size - margin_size - 1),
            np.abs(y_val_max - y_val_min) / (grid_size - margin_size - 1),
        ]
    )

    pts = list(zip(x_list, y_list))

    for pt in pts:
        ind = position_to_grid(pt, margin_size=margin_size, grid_spacing=grid_spacing)
        if num_faces == 1:
            grid[int(ind[0]), int(ind[1])] = 10
        else:
            grid[int(ind[0]), int(ind[1]), 0] = 10

    return grid, margin_size, grid_spacing


def position_to_grid(
    point: Tuple[float, float], margin_size: float, grid_spacing: np.ndarray
) -> np.ndarray:
    """Map a continuous point to integer grid coordinates."""
    return np.round(point / grid_spacing) + margin_size / 2


def position_to_rectangular_grid(
    point: Tuple[float, float],
    margin_size_x: float,
    margin_size_y: float,
    grid_spacing: Iterable[float],
) -> np.ndarray:
    """Map a continuous point to non-square grid coordinates."""
    point = np.array(point)
    margin_size = np.array((margin_size_x, margin_size_y))
    grid_spacing = np.array(grid_spacing)
    return np.round(point / grid_spacing) + margin_size / 2


def grid_to_position(
    xy_idx: Tuple[int, int], margin_size: float, grid_spacing: float | np.ndarray
) -> list[float]:
    """Map grid coordinates back to continuous position."""
    return [
        (xy_idx[0] - margin_size / 2) * grid_spacing,
        (xy_idx[1] - margin_size / 2) * grid_spacing,
    ]


def distance(point1: tuple, point2: tuple) -> float:
    """Euclidean distance between 2D points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def isX(point1: tuple[int, int], point2: tuple[int, int]) -> bool:
    """Return True if the points share Y coordinate (horizontal edge)."""
    return point1[0] != point2[0] and point1[1] == point2[1]


def isY(point1: tuple[int, int], point2: tuple[int, int]) -> bool:
    """Return True if the points share X coordinate (vertical edge)."""
    return point1[0] == point2[0] and point1[1] != point2[1]
