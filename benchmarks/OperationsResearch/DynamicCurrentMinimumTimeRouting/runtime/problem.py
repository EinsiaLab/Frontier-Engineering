from __future__ import annotations

import heapq
import math
from typing import Any


def _build_case(case_id: str, width: int, height: int, start: tuple[int, int], goal: tuple[int, int], land: tuple[int, int, int, int], bands: tuple[dict[str, float], ...], min_depth: float, shallow: tuple[tuple[int, int, float], ...]) -> dict[str, Any]:
    def is_land(cell: tuple[int, int]) -> bool:
        x, y = cell
        x0, x1, y0, y1 = land
        return x0 <= x <= x1 and y0 <= y <= y1

    def depth_at(cell: tuple[int, int]) -> float:
        if is_land(cell):
            return 0.0
        x, y = cell
        depth = 3.8
        for sx, sy, value in shallow:
            if x == sx and y == sy:
                depth = value
        return depth

    def is_navigable(cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < width and 0 <= y < height and not is_land(cell) and depth_at(cell) >= min_depth

    def current_at(cell: tuple[int, int]) -> tuple[float, float]:
        x, y = cell
        band = bands[min(len(bands) - 1, y // max(1, height // len(bands)))]
        east = band["east"] + band.get("east_amp", 0.0) * math.sin(band.get("east_freq", 0.4) * x)
        north = band.get("north", 0.0) * math.cos(band.get("north_freq", 0.3) * x)
        return (east, north)

    rows = []
    current_rows = []
    depth_rows = []
    for y in range(height):
        row = []
        current_row = []
        depth_row = []
        for x in range(width):
            cell = (x, y)
            if cell == start:
                row.append("S")
            elif cell == goal:
                row.append("G")
            elif is_land(cell):
                row.append("#")
            elif depth_at(cell) < min_depth:
                row.append("~")
            else:
                row.append(".")
            current_row.append(tuple(round(v, 4) for v in current_at(cell)))
            depth_row.append(round(depth_at(cell), 4))
        rows.append("".join(row))
        current_rows.append(tuple(current_row))
        depth_rows.append(tuple(depth_row))
    return {
        "case_id": case_id,
        "grid": tuple(rows),
        "start": start,
        "goal": goal,
        "current_field": tuple(current_rows),
        "depth_field": tuple(depth_rows),
        "min_depth": float(min_depth),
        "objective": "time",
        "max_hops": int(width * height),
    }


PUBLIC_CASES = (
    _build_case("public_mid_channel", 20, 10, (1, 4), (18, 4), (8, 12, 2, 6), ({"east": -0.32, "east_amp": 0.03, "north": 0.01}, {"east": -0.05, "east_amp": 0.03, "north": 0.02}, {"east": 0.42, "east_amp": 0.03, "north": -0.01}), 2.5, ((3, 7, 2.4), (4, 7, 2.4), (5, 7, 2.4), (2, 6, 2.2), (3, 6, 2.2), (4, 6, 2.2))),
    _build_case("public_top_bypass", 20, 10, (1, 2), (18, 2), (7, 11, 3, 7), ({"east": -0.25, "east_amp": 0.02, "north": 0.01}, {"east": 0.02, "east_amp": 0.03, "north": 0.01}, {"east": 0.35, "east_amp": 0.02, "north": -0.01}), 2.4, ((6, 1, 2.1), (7, 1, 2.1), (8, 1, 2.1))),
    _build_case("public_bottom_bypass", 20, 10, (1, 7), (18, 7), (8, 12, 1, 5), ({"east": -0.38, "east_amp": 0.02, "north": 0.01}, {"east": 0.00, "east_amp": 0.02, "north": 0.02}, {"east": 0.28, "east_amp": 0.02, "north": -0.01}), 2.5, ((5, 8, 2.3), (6, 8, 2.3), (7, 8, 2.3))),
)

HIDDEN_CASES = (
    _build_case("hidden_diagonal_access", 22, 10, (1, 3), (20, 6), (9, 13, 2, 6), ({"east": -0.34, "east_amp": 0.03, "north": 0.01}, {"east": -0.04, "east_amp": 0.03, "north": 0.02}, {"east": 0.40, "east_amp": 0.03, "north": -0.01}), 2.5, ((4, 1, 2.3), (5, 1, 2.3), (6, 1, 2.3), (3, 8, 2.4), (4, 8, 2.4))),
    _build_case("hidden_north_pressure", 20, 11, (1, 2), (18, 8), (8, 12, 3, 7), ({"east": -0.42, "east_amp": 0.03, "north": 0.02}, {"east": -0.08, "east_amp": 0.02, "north": 0.01}, {"east": 0.24, "east_amp": 0.02, "north": -0.02}), 2.4, ((2, 1, 2.2), (3, 1, 2.2), (6, 9, 2.3))),
    _build_case("hidden_central_bottleneck", 21, 10, (1, 5), (19, 4), (9, 12, 2, 7), ({"east": -0.30, "east_amp": 0.03, "north": 0.01}, {"east": -0.01, "east_amp": 0.03, "north": 0.02}, {"east": 0.33, "east_amp": 0.03, "north": -0.01}), 2.5, ((5, 7, 2.4), (6, 7, 2.4), (7, 7, 2.4), (6, 2, 2.3))),
    _build_case("hidden_long_east", 24, 10, (1, 4), (22, 4), (10, 14, 2, 6), ({"east": -0.36, "east_amp": 0.02, "north": 0.01}, {"east": -0.02, "east_amp": 0.02, "north": 0.01}, {"east": 0.45, "east_amp": 0.02, "north": -0.01}), 2.5, ((7, 7, 2.3), (8, 7, 2.3), (9, 7, 2.3), (4, 1, 2.2))),
    _build_case("hidden_dual_detour", 20, 12, (1, 5), (18, 5), (8, 12, 3, 8), ({"east": -0.33, "east_amp": 0.03, "north": 0.01}, {"east": -0.03, "east_amp": 0.03, "north": 0.01}, {"east": 0.37, "east_amp": 0.03, "north": -0.01}), 2.5, ((3, 10, 2.3), (4, 10, 2.3), (6, 1, 2.2), (7, 1, 2.2))),
)


def load_instance() -> dict[str, Any]:
    return dict(PUBLIC_CASES[0])


def _to_cell(value: Any) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("cell must be a length-2 sequence")
    return int(round(float(value[0]))), int(round(float(value[1])))


def extract_path(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, dict):
        if "path" not in value:
            raise ValueError("missing path")
        value = value["path"]
    path = [_to_cell(cell) for cell in value]
    if not path:
        raise ValueError("path is empty")
    return path


def is_navigable(instance: dict[str, Any], cell: tuple[int, int]) -> bool:
    x, y = cell
    rows = instance["grid"]
    return 0 <= y < len(rows) and 0 <= x < len(rows[0]) and rows[y][x] in {".", "S", "G"}


def neighbors(instance: dict[str, Any], cell: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = cell
    out = []
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nxt = (x + dx, y + dy)
        if is_navigable(instance, nxt):
            out.append(nxt)
    return out


def validate_path(instance: dict[str, Any], value: Any) -> list[tuple[int, int]]:
    path = extract_path(value)
    if path[0] != tuple(instance["start"]):
        raise ValueError("path must start at START")
    if path[-1] != tuple(instance["goal"]):
        raise ValueError("path must end at GOAL")
    if len(path) - 1 > int(instance["max_hops"]):
        raise ValueError("path exceeds hop budget")
    for cell in path:
        if not is_navigable(instance, cell):
            raise ValueError("path enters land or leaves the map")
    for prev, curr in zip(path, path[1:]):
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        if dx + dy != 1:
            raise ValueError("path contains a non-adjacent move")
    return path


def _leg_time(instance: dict[str, Any], prev: tuple[int, int], curr: tuple[int, int]) -> float:
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    current_u, current_v = instance["current_field"][prev[1]][prev[0]]
    current_along = current_u * dx + current_v * dy
    depth = float(instance["depth_field"][curr[1]][curr[0]])
    shallow_penalty = max(0.0, 3.0 - depth) * 0.22
    speed = max(0.25, 1.0 + 0.9 * current_along - shallow_penalty)
    return 1.0 / speed


def route_metrics(instance: dict[str, Any], value: Any) -> dict[str, float]:
    path = validate_path(instance, value)
    total_time_h = 0.0
    for prev, curr in zip(path, path[1:]):
        total_time_h += _leg_time(instance, prev, curr)
    return {"time_h": float(total_time_h), "hops": float(len(path) - 1)}


def shortest_time_path(instance: dict[str, Any]) -> list[tuple[int, int]]:
    start = tuple(instance["start"])
    goal = tuple(instance["goal"])
    frontier = [(0.0, start)]
    parent = {start: None}
    best = {start: 0.0}
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        if current_cost > best[current]:
            continue
        for nxt in neighbors(instance, current):
            next_cost = current_cost + _leg_time(instance, current, nxt)
            if next_cost < best.get(nxt, float("inf")):
                best[nxt] = next_cost
                parent[nxt] = current
                heapq.heappush(frontier, (next_cost, nxt))
    raise RuntimeError("no feasible path found")
