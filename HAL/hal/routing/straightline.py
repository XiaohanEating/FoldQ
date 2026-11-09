from typing import Union

import numpy as np


def straightline(
    array: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int]
) -> Union[bool, list[tuple[int, int, int]]]:
    """Generate an axis-aligned straight path from start to goal if clear.

    Returns a list of (x, y, z) positions when the straight path is valid
    given face-switching rules; otherwise returns False.
    """
    x1 = goal[0]
    y1 = goal[1]

    x0 = start[0]
    y0 = start[1]

    x = []
    y = []

    if y1 - y0 == 0:
        x_step = 1 if x1 - x0 > 0 else -1

        for x_val in range(x0, x1, x_step):
            x.append(x_val)
            y.append(y0)

        x.append(x1)
        y.append(y1)

    elif x1 - x0 == 0:
        y_step = 1 if y1 - y0 > 0 else -1

        for y_val in range(y0, y1, y_step):
            x.append(x0)
            y.append(y_val)

        x.append(x1)
        y.append(y1)

    else:
        slope = (y1 - y0) / (x1 - x0)

        if np.abs(slope) > 1:
            x0temp = x0
            x1temp = x1
            x0 = y0
            y0 = x0temp
            x1 = y1
            y1 = x1temp
            slope = 1 / slope

            step = 1 if x0 < x1 else -1
            y = np.arange(x0, x1, step)
            x = slope * (y - x0) + y0
            y = list(y)
            x = [int(v) for v in x]

            x.append(goal[0])
            y.append(goal[1])
        else:
            step = 1 if x0 < x1 else -1
            x = np.arange(x0, x1 + step, step)
            y = slope * (x - x0) + y0
            y = [int(v) for v in y]

    route = [(x, y, 0) for x, y in zip(x, y)]

    curr_face = 0
    rerouted_remaineder_with_astar = False
    to_reroute_with_astar = []
    for ind in range(2, len(route) - 2):
        cell_val = array[route[ind][0]][route[ind][1]][curr_face]

        if cell_val in (2, 10):
            curr_face = 1
        elif cell_val == 4:
            curr_face = 0

        # route[ind] = (route[ind][0], route[ind][1], curr_face)

        # if (array[route[ind][0]][route[ind][1]][curr_face] == 10) or \
        #     array[route[ind][0]][route[ind][1]][curr_face] == blockage_val:
        #     curr_face = 1 - curr_face
        #     blockage_val = 2 if curr_face == 0 else 4

        if array[route[ind][0]][route[ind][1]][curr_face] in (2, 4, 10):
            return False
        route[ind] = (route[ind][0], route[ind][1], curr_face)
    # for ind in range(len(route)):
    #     if curr_face == 1 and array[route[ind][0]][route[ind][1]][0] != 2:
    #         route[ind] = (route[ind][0], route[ind][1], 0)

    # raised_route_inds = [ind for ind in range(len(route)) if route[ind][1] == 1]
    # inds_to_moveup = []
    # if len(raised_route_inds) > 0:
    #     for ind in range(raised_route_inds[0], raised_route_inds[-1]):
    #         if route[ind][2] == 0 and array[route[ind][0]][route[ind][1]][1] != 4:
    #                 inds_to_moveup.append(ind)

    # for inds_to_moveup_ind in inds_to_moveup:
    #     route[inds_to_moveup_ind] = (route[inds_to_moveup_ind][0], route[inds_to_moveup_ind][1], 1)

    # for reroute_range in to_reroute_with_astar:
    #     astar_route = astar_3d(array, reroute_range[0], reroute_range[-1])
    #     if type(astar_route) is bool:
    #         return False
    #     else:
    #         astar_route.append(reroute_range[0])
    #     index_to_inject = route.index(reroute_range[0])
    #     route = [r for r in route if r not in reroute_range]
    #     for ind, point in enumerate(astar_route):
    #         route.insert(index_to_inject+ind, point)

    # for reroute_range in to_reroute_with_astar_edge:
    #     astar_route = astar_3d(array, reroute_range[0], reroute_range[-1])
    #     if type(astar_route) is bool:
    #         return False
    #     else:
    #         astar_route.append(reroute_range[0])
    #     index_to_inject = route.index(reroute_range[0])
    #     route = [r for r in route if r not in reroute_range]
    #     for ind, point in enumerate(astar_route):
    #         route.insert(index_to_inject+ind, point)

    # if rerouted_remaineder_with_astar:
    #     new_route = route[:ind]

    #     for r in astar_route_mid:
    #         new_route.append(r)

    #     route = new_route

    # print(to_reroute_with_astar)
    return route


def straightline_no_face_switches(
    array: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int]
) -> Union[bool, list[tuple[int, int, int]]]:
    """Generate a straight line path without face switches; return False if blocked."""
    x1 = goal[0]
    y1 = goal[1]

    x0 = start[0]
    y0 = start[1]

    x = []
    y = []

    if y1 - y0 == 0:
        x_step = 1 if x1 - x0 > 0 else -1

        for x_val in range(x0, x1, x_step):
            x.append(x_val)
            y.append(y0)

        x.append(x1)
        y.append(y1)

    elif x1 - x0 == 0:
        y_step = 1 if y1 - y0 > 0 else -1

        for y_val in range(y0, y1, y_step):
            x.append(x0)
            y.append(y_val)

        x.append(x1)
        y.append(y1)
        # print([(x,y) for x,y in zip(x, y)])
    else:
        slope = (y1 - y0) / (x1 - x0)

        if np.abs(slope) > 1:
            x0temp = x0
            x1temp = x1
            x0 = y0
            y0 = x0temp
            x1 = y1
            y1 = x1temp
            slope = 1 / slope

            step = 1 if x0 < x1 else -1
            y = np.arange(x0, x1, step)
            x = slope * (y - x0) + y0
            y = list(y)
            x = [int(v) for v in x]

            x.append(goal[0])
            y.append(goal[1])
        else:
            step = 1 if x0 < x1 else -1
            x = np.arange(x0, x1 + step, step)
            y = slope * (x - x0) + y0
            y = [int(v) for v in y]

    route = [(x, y, 0) for x, y in zip(x, y)]

    # 2026-11-06 修复（FoldQ 复核 P0）：原 range(3, len(route)-3) 使 ≤6 顶点的
    # 短路径完全跳过占用检查，端点邻域也不检查，导致交叉漏检
    # （docs/reviews/2026-11-06-dac 反例 D4_3/GT_98）。改为检查全部内部顶点。
    for ind in range(1, len(route) - 1):
        cell_val = array[route[ind][0]][route[ind][1]][0]

        if cell_val in (2, 4, 10):  # FoldQ 2026-11-06: face1 占用标记为 4，原检查漏检
            return False

    return route


def is_consecutive(point_1: tuple[int, int, int], point_2: tuple[int, int, int]) -> bool:
    return bool(any(np.abs(np.array(point_1) - np.array(point_2)) == 1))


def connect(
    array: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int]
) -> Union[bool, list[tuple[int, int, int]]]:
    x1 = goal[0]
    y1 = goal[1]

    x0 = start[0]
    y0 = start[1]

    x = []
    y = []

    if y1 - y0 == 0:
        x_step = 1 if x1 - x0 > 0 else -1

        for x_val in range(x0, x1, x_step):
            x.append(x_val)
            y.append(y0)

        x.append(x1)
        y.append(y1)

    elif x1 - x0 == 0:
        y_step = 1 if y1 - y0 > 0 else -1

        for y_val in range(y0, y1, y_step):
            x.append(x0)
            y.append(y_val)

        x.append(x1)
        y.append(y1)

    else:
        raise ValueError("Cannot connect start and goal with a vertical or horizontal line")

    route = [(x, y, 0) for x, y in zip(x, y)]
    return route
