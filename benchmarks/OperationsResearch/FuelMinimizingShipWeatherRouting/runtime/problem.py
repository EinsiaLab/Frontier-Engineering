from __future__ import annotations

import heapq
import math
from typing import Any


def _build_case(case_id: str, width: int, height: int, start: tuple[int, int], goal: tuple[int, int], land: tuple[int, int, int, int], wind_bands: tuple[dict[str, float], ...], current_bands: tuple[dict[str, float], ...], latest_arrival_h: float) -> dict[str, Any]:
    def is_land(cell: tuple[int, int]) -> bool:
        x, y = cell
        x0, x1, y0, y1 = land
        return x0 <= x <= x1 and y0 <= y <= y1

    def current_at(cell: tuple[int, int]) -> tuple[float, float]:
        x, y = cell
        band = current_bands[min(len(current_bands) - 1, y // max(1, height // len(current_bands)))]
        east = band["east"] + band.get("east_amp", 0.0) * math.sin(band.get("east_freq", 0.45) * x)
        north = band.get("north", 0.0) * math.cos(band.get("north_freq", 0.35) * x)
        return (east, north)

    def wind_at(cell: tuple[int, int]) -> tuple[float, float]:
        x, y = cell
        band = wind_bands[min(len(wind_bands) - 1, y // max(1, height // len(wind_bands)))]
        east = band["east"] + band.get("east_amp", 0.0) * math.sin(band.get("east_freq", 0.3) * x)
        north = band.get("north", 0.0) * math.cos(band.get("north_freq", 0.2) * x)
        return (east, north)

    rows = []
    current_rows = []
    wind_rows = []
    for y in range(height):
        row = []
        current_row = []
        wind_row = []
        for x in range(width):
            cell = (x, y)
            if cell == start:
                row.append("S")
            elif cell == goal:
                row.append("G")
            elif is_land(cell):
                row.append("#")
            else:
                row.append(".")
            current_row.append(tuple(round(v, 4) for v in current_at(cell)))
            wind_row.append(tuple(round(v, 4) for v in wind_at(cell)))
        rows.append("".join(row))
        current_rows.append(tuple(current_row))
        wind_rows.append(tuple(wind_row))
    return {
        "case_id": case_id,
        "grid": tuple(rows),
        "start": start,
        "goal": goal,
        "current_field": tuple(current_rows),
        "wind_field": tuple(wind_rows),
        "objective": "fuel",
        "latest_arrival_h": float(latest_arrival_h),
    }


PUBLIC_CASES = (
    _build_case("public_mid_channel", 20, 10, (1, 4), (18, 4), (8, 12, 2, 6), ({"east": -0.60, "east_amp": 0.04, "north": 0.04}, {"east": -0.08, "east_amp": 0.02, "north": 0.02}, {"east": 0.22, "east_amp": 0.04, "north": -0.04}), ({"east": -0.32, "east_amp": 0.04, "north": 0.02}, {"east": 0.04, "east_amp": 0.03, "north": 0.01}, {"east": 0.26, "east_amp": 0.04, "north": -0.02}), 46.0),
    _build_case("public_top_route", 20, 10, (1, 2), (18, 2), (7, 11, 3, 7), ({"east": -0.55, "east_amp": 0.03, "north": 0.03}, {"east": -0.06, "east_amp": 0.02, "north": 0.01}, {"east": 0.20, "east_amp": 0.03, "north": -0.02}), ({"east": -0.26, "east_amp": 0.03, "north": 0.02}, {"east": 0.02, "east_amp": 0.02, "north": 0.01}, {"east": 0.21, "east_amp": 0.03, "north": -0.02}), 44.0),
    _build_case("public_bottom_route", 20, 10, (1, 7), (18, 7), (8, 12, 1, 5), ({"east": -0.52, "east_amp": 0.04, "north": 0.02}, {"east": -0.04, "east_amp": 0.02, "north": 0.01}, {"east": 0.18, "east_amp": 0.04, "north": -0.02}), ({"east": -0.30, "east_amp": 0.03, "north": 0.01}, {"east": 0.03, "east_amp": 0.02, "north": 0.01}, {"east": 0.28, "east_amp": 0.03, "north": -0.01}), 44.0),
)

HIDDEN_CASES = (
    _build_case("hidden_diagonal_access", 22, 10, (1, 3), (20, 6), (9, 13, 2, 6), ({"east": -0.62, "east_amp": 0.04, "north": 0.03}, {"east": -0.07, "east_amp": 0.02, "north": 0.01}, {"east": 0.24, "east_amp": 0.04, "north": -0.03}), ({"east": -0.29, "east_amp": 0.03, "north": 0.01}, {"east": 0.03, "east_amp": 0.02, "north": 0.01}, {"east": 0.24, "east_amp": 0.03, "north": -0.01}), 49.0),
    _build_case("hidden_central_pressure", 20, 11, (1, 5), (18, 8), (8, 12, 3, 7), ({"east": -0.58, "east_amp": 0.03, "north": 0.03}, {"east": -0.10, "east_amp": 0.02, "north": 0.01}, {"east": 0.16, "east_amp": 0.03, "north": -0.03}), ({"east": -0.31, "east_amp": 0.03, "north": 0.01}, {"east": 0.01, "east_amp": 0.02, "north": 0.01}, {"east": 0.26, "east_amp": 0.03, "north": -0.01}), 47.0),
    _build_case("hidden_long_east", 24, 10, (1, 4), (22, 4), (10, 14, 2, 6), ({"east": -0.57, "east_amp": 0.04, "north": 0.02}, {"east": -0.08, "east_amp": 0.02, "north": 0.01}, {"east": 0.20, "east_amp": 0.04, "north": -0.02}), ({"east": -0.33, "east_amp": 0.03, "north": 0.02}, {"east": 0.02, "east_amp": 0.02, "north": 0.01}, {"east": 0.27, "east_amp": 0.03, "north": -0.02}), 52.0),
    _build_case("hidden_top_stress", 20, 10, (1, 2), (18, 6), (8, 12, 3, 6), ({"east": -0.64, "east_amp": 0.03, "north": 0.03}, {"east": -0.12, "east_amp": 0.02, "north": 0.01}, {"east": 0.12, "east_amp": 0.03, "north": -0.02}), ({"east": -0.28, "east_amp": 0.03, "north": 0.02}, {"east": 0.01, "east_amp": 0.02, "north": 0.01}, {"east": 0.19, "east_amp": 0.03, "north": -0.02}), 46.0),
    _build_case("hidden_bottom_stress", 20, 12, (1, 8), (18, 5), (8, 12, 3, 8), ({"east": -0.55, "east_amp": 0.04, "north": 0.02}, {"east": -0.07, "east_amp": 0.02, "north": 0.01}, {"east": 0.19, "east_amp": 0.04, "north": -0.03}), ({"east": -0.34, "east_amp": 0.03, "north": 0.01}, {"east": -0.01, "east_amp": 0.02, "north": 0.01}, {"east": 0.25, "east_amp": 0.03, "north": -0.01}), 48.0),
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


def is_water(instance: dict[str, Any], cell: tuple[int, int]) -> bool:
    x, y = cell
    rows = instance["grid"]
    return 0 <= y < len(rows) and 0 <= x < len(rows[0]) and rows[y][x] in {".", "S", "G"}


def neighbors(instance: dict[str, Any], cell: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = cell
    out = []
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nxt = (x + dx, y + dy)
        if is_water(instance, nxt):
            out.append(nxt)
    return out


def validate_path(instance: dict[str, Any], value: Any) -> list[tuple[int, int]]:
    path = extract_path(value)
    if path[0] != tuple(instance["start"]):
        raise ValueError("path must start at START")
    if path[-1] != tuple(instance["goal"]):
        raise ValueError("path must end at GOAL")
    for cell in path:
        if not is_water(instance, cell):
            raise ValueError("path enters land or leaves the map")
    for prev, curr in zip(path, path[1:]):
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        if dx + dy != 1:
            raise ValueError("path contains a non-adjacent move")
    return path


def _leg_metrics(instance: dict[str, Any], prev: tuple[int, int], curr: tuple[int, int]) -> tuple[float, float]:
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    current_u, current_v = instance["current_field"][prev[1]][prev[0]]
    wind_u, wind_v = instance["wind_field"][prev[1]][prev[0]]
    current_along = current_u * dx + current_v * dy
    wind_along = wind_u * dx + wind_v * dy
    headwind = max(0.0, -wind_along)
    crosswind = abs(-dy * wind_u + dx * wind_v)
    speed = max(0.35, 1.0 + 0.65 * current_along - 0.45 * headwind)
    leg_time_h = 1.0 / speed
    fuel_rate = 1.05 + 0.55 * headwind + 0.20 * crosswind + 0.25 * max(0.0, -current_along)
    return leg_time_h * fuel_rate, leg_time_h


def route_metrics(instance: dict[str, Any], value: Any) -> dict[str, float]:
    path = validate_path(instance, value)
    total_fuel = 0.0
    total_time_h = 0.0
    for prev, curr in zip(path, path[1:]):
        leg_fuel, leg_time = _leg_metrics(instance, prev, curr)
        total_fuel += leg_fuel
        total_time_h += leg_time
    if total_time_h > float(instance["latest_arrival_h"]):
        raise ValueError("path misses the latest-arrival constraint")
    return {"fuel": float(total_fuel), "time_h": float(total_time_h), "hops": float(len(path) - 1)}


def minimum_fuel_path(instance: dict[str, Any]) -> list[tuple[int, int]]:
    start = tuple(instance["start"])
    goal = tuple(instance["goal"])
    frontier = [(0.0, 0.0, start)]
    parent = {start: None}
    best = {start: (0.0, 0.0)}
    while frontier:
        current_fuel, current_time, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        if current_fuel > best[current][0]:
            continue
        for nxt in neighbors(instance, current):
            leg_fuel, leg_time = _leg_metrics(instance, current, nxt)
            next_fuel = current_fuel + leg_fuel
            next_time = current_time + leg_time
            if next_time > float(instance["latest_arrival_h"]):
                continue
            best_cost = best.get(nxt)
            if best_cost is None or next_fuel < best_cost[0]:
                best[nxt] = (next_fuel, next_time)
                parent[nxt] = current
                heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(frontier, (next_fuel + 0.01 * heuristic, next_time, nxt))
    raise RuntimeError("no feasible path found")
