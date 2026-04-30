from __future__ import annotations

from heapq import heappop, heappush
from typing import Any


PUBLIC_CASES = (
    {"case_id": "public_cross", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
    {"case_id": "public_cross_repeat_1", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
    {"case_id": "public_cross_repeat_2", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
    {"case_id": "public_cross_repeat_3", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
)

HIDDEN_CASES = (
    {"case_id": "hidden_cross_repeat_1", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
    {"case_id": "hidden_cross_repeat_2", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
    {"case_id": "hidden_cross_repeat_3", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
    {"case_id": "hidden_cross_repeat_4", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
    {"case_id": "hidden_cross_repeat_5", "grid": ("##########", "#....#..##", "#..#..#.##", "#..B##.bC#", "#...c#...#", "#......#A#", "#.a..#...#", "##########")},
)


def _parse_grid(grid):
    start_map: dict[str, tuple[int, int]] = {}
    goal_map: dict[str, tuple[int, int]] = {}
    rows = []
    for y, row in enumerate(grid):
        new_row = []
        for x, cell in enumerate(row):
            if cell in "ABC":
                start_map[cell] = (x, y)
                new_row.append(".")
            elif cell in "abc":
                goal_map[cell.upper()] = (x, y)
                new_row.append(".")
            else:
                new_row.append(cell)
        rows.append("".join(new_row))
    robot_ids = tuple(sorted(start_map))
    starts = tuple(start_map[robot_id] for robot_id in robot_ids)
    goals = tuple(goal_map[robot_id] for robot_id in robot_ids)
    return tuple(rows), robot_ids, starts, goals


def load_instance() -> dict[str, Any]:
    grid, robot_ids, starts, goals = _parse_grid(PUBLIC_CASES[0]["grid"])
    return {"grid": grid, "robot_ids": robot_ids, "starts": starts, "goals": goals}


def _to_cell(value: Any) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("cell must be a length-2 sequence")
    return int(round(float(value[0]))), int(round(float(value[1])))


def _extract_paths(value: Any) -> list[list[tuple[int, int]]]:
    if isinstance(value, dict):
        if "paths" not in value:
            raise ValueError("missing paths")
        value = value["paths"]
    paths = []
    for raw_path in value:
        path = [_to_cell(cell) for cell in raw_path]
        if not path:
            raise ValueError("robot path is empty")
        paths.append(path)
    return paths


def is_free(grid: tuple[str, ...], cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] != "#"


def neighbors(grid: tuple[str, ...], cell: tuple[int, int], allow_wait: bool = False) -> list[tuple[int, int]]:
    x, y = cell
    result = []
    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    if allow_wait:
        candidates.append((x, y))
    for candidate in candidates:
        if is_free(grid, candidate):
            result.append(candidate)
    return result


def _retrace(parent, node):
    path = []
    current = node
    while current is not None:
        path.append(current[0])
        current = parent[current]
    return path[::-1]


def space_time_astar(grid, start, goal, reserved_vertices, reserved_edges, max_time=60):
    def heuristic(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    frontier = [(heuristic(start), 0, start)]
    parent = {(start, 0): None}
    best_time = {(start, 0): 0}
    while frontier:
        _, current_time, current = heappop(frontier)
        if best_time[(current, current_time)] != current_time:
            continue
        if current == goal:
            return _retrace(parent, (current, current_time))
        if current_time >= max_time:
            continue
        for nxt in neighbors(grid, current, allow_wait=True):
            next_time = current_time + 1
            if (nxt, next_time) in reserved_vertices:
                continue
            if ((current, nxt), next_time) in reserved_edges or ((nxt, current), next_time) in reserved_edges:
                continue
            state = (nxt, next_time)
            if state in best_time and next_time >= best_time[state]:
                continue
            best_time[state] = next_time
            parent[state] = (current, current_time)
            heappush(frontier, (next_time + heuristic(nxt), next_time, nxt))
    return None


def reserve_path(path, reserved_vertices, reserved_edges, horizon=60):
    for t, cell in enumerate(path):
        reserved_vertices.add((cell, t))
        if t > 0:
            reserved_edges.add(((path[t - 1], cell), t))
        if t == len(path) - 1:
            for future in range(t + 1, horizon + 1):
                reserved_vertices.add((cell, future))


def prioritized_plan(instance: dict[str, Any], order: tuple[int, ...]) -> list[list[tuple[int, int]]]:
    grid = tuple(instance["grid"])
    starts = tuple(instance["starts"])
    goals = tuple(instance["goals"])
    reserved_vertices = set()
    reserved_edges = set()
    paths: list[list[tuple[int, int]] | None] = [None] * len(starts)
    for robot_idx in order:
        path = space_time_astar(grid, starts[robot_idx], goals[robot_idx], reserved_vertices, reserved_edges)
        if path is None:
            raise RuntimeError("prioritized planning failed")
        paths[robot_idx] = path
        reserve_path(path, reserved_vertices, reserved_edges)
    return [path for path in paths if path is not None]


def validate_paths(instance: dict[str, Any], value: Any) -> list[list[tuple[int, int]]]:
    grid = tuple(instance["grid"])
    starts = tuple(instance["starts"])
    goals = tuple(instance["goals"])
    paths = _extract_paths(value)
    if len(paths) != len(starts):
        raise ValueError("incorrect number of robot paths")
    for idx, path in enumerate(paths):
        if path[0] != starts[idx] or path[-1] != goals[idx]:
            raise ValueError("robot path endpoints are invalid")
        for cell in path:
            if not is_free(grid, cell):
                raise ValueError("robot path enters obstacle")
        for previous, current in zip(path, path[1:]):
            if abs(previous[0] - current[0]) + abs(previous[1] - current[1]) not in {0, 1}:
                raise ValueError("robot path contains a non-adjacent move")

    horizon = max(len(path) for path in paths)
    previous_positions = [path[0] for path in paths]
    for t in range(horizon):
        positions = [path[t] if t < len(path) else path[-1] for path in paths]
        if len(set(positions)) != len(positions):
            raise ValueError("vertex collision detected")
        if t > 0:
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    if previous_positions[i] == positions[j] and previous_positions[j] == positions[i]:
                        raise ValueError("edge-swap collision detected")
        previous_positions = positions
    return paths


def total_cost(instance: dict[str, Any], value: Any) -> int:
    return sum(len(path) - 1 for path in validate_paths(instance, value))


def makespan(instance: dict[str, Any], value: Any) -> int:
    return max(len(path) - 1 for path in validate_paths(instance, value))
