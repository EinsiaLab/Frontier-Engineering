from __future__ import annotations

from heapq import heappop, heappush
from typing import Any


PUBLIC_CASES = (
    {"case_id": "public_open_detour", "grid": ("####################", "#S........####.....#", "#..###..#.#.#..##..#", "#...#....##.......##", "#...#.#.......##..##", "#.#.#......#...###.#", "##......#.......#..#", "#................#.#", "#....##.#.......#..#", "#.........#....#.#.#", "##..#.#.#..##...#..#", "#.......##.........#", "#..##.......#...##G#", "####################")},
    {"case_id": "public_corner_turn", "grid": ("###############", "#S.....#......#", "#.###..#.####.#", "#...#..#....#.#", "###.#..####.#.#", "#...#.......#.#", "#.#####.###.#.#", "#.......#...#G#", "###############")},
    {"case_id": "public_long_corridor", "grid": ("################", "#S....#........#", "###.#.#.######.#", "#...#.#......#.#", "#.###.######.#.#", "#...#......#.#.#", "###.######.#.#.#", "#........#...#G#", "################")},
    {"case_id": "public_sparse_rooms", "grid": ("###############", "#S..#.....#...#", "#.#.#.###.#.#.#", "#.#...#...#.#.#", "#.#####.###.#.#", "#.....#.....#.#", "#.###.#####.#.#", "#...#.......#G#", "###############")},
    {"case_id": "public_mid_maze", "grid": ("###############", "#S......#.....#", "#.####..#.###.#", "#....#..#...#.#", "####.#.####.#.#", "#....#......#.#", "#.###########.#", "#............G#", "###############")},
)

HIDDEN_CASES = (
    {"case_id": "hidden_split_channel", "grid": ("################", "#S.....#.......#", "###.##.#.#####.#", "#...##.#.....#.#", "#.####.#####.#.#", "#......#.....#.#", "#.######.#####.#", "#............G##", "################")},
    {"case_id": "hidden_shortcut_gate", "grid": ("###############", "#S....#.......#", "#.###.#.#####.#", "#...#.#.#...#.#", "###.#.#.#.#.#.#", "#...#...#.#...#", "#.#######.###.#", "#...........#G#", "###############")},
    {"case_id": "hidden_public_corner_turn", "grid": ("###############", "#S.....#......#", "#.###..#.####.#", "#...#..#....#.#", "###.#..####.#.#", "#...#.......#.#", "#.#####.###.#.#", "#.......#...#G#", "###############")},
    {"case_id": "hidden_public_sparse_rooms", "grid": ("###############", "#S..#.....#...#", "#.#.#.###.#.#.#", "#.#...#...#.#.#", "#.#####.###.#.#", "#.....#.....#.#", "#.###.#####.#.#", "#...#.......#G#", "###############")},
    {"case_id": "hidden_public_mid_maze", "grid": ("###############", "#S......#.....#", "#.####..#.###.#", "#....#..#...#.#", "####.#.####.#.#", "#....#......#.#", "#.###########.#", "#............G#", "###############")},
)


def _parse_grid(grid: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[int, int], tuple[int, int]]:
    start = None
    goal = None
    rows = []
    for y, row in enumerate(grid):
        new_row = []
        for x, cell in enumerate(row):
            if cell == "S":
                start = (x, y)
                new_row.append(".")
            elif cell == "G":
                goal = (x, y)
                new_row.append(".")
            else:
                new_row.append(cell)
        rows.append("".join(new_row))
    if start is None or goal is None:
        raise ValueError("grid must contain both S and G")
    return tuple(rows), start, goal


def load_instance() -> dict[str, Any]:
    grid, start, goal = _parse_grid(PUBLIC_CASES[0]["grid"])
    return {"grid": grid, "start": start, "goal": goal}


def _to_cell(value: Any) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("cell must be a length-2 sequence")
    return int(round(float(value[0]))), int(round(float(value[1])))


def _extract_path(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, dict):
        if "path" not in value:
            raise ValueError("missing path")
        value = value["path"]
    path = [_to_cell(cell) for cell in value]
    if not path:
        raise ValueError("path is empty")
    return path


def is_free(grid: tuple[str, ...], cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] != "#"


def validate_path(instance: dict[str, Any], path_value: Any):
    grid = tuple(instance["grid"])
    start = tuple(instance["start"])
    goal = tuple(instance["goal"])
    path = _extract_path(path_value)
    if path[0] != start or path[-1] != goal:
        raise ValueError("path endpoints are invalid")
    for cell in path:
        if not is_free(grid, cell):
            raise ValueError("path enters obstacle")
    for previous, current in zip(path, path[1:]):
        if abs(previous[0] - current[0]) + abs(previous[1] - current[1]) != 1:
            raise ValueError("path contains a non-adjacent move")
    return path


def path_cost(instance: dict[str, Any], path_value: Any) -> int:
    return len(validate_path(instance, path_value)) - 1


def shortest_path(instance: dict[str, Any]) -> list[tuple[int, int]]:
    grid = tuple(instance["grid"])
    start = tuple(instance["start"])
    goal = tuple(instance["goal"])
    frontier = [(0, start)]
    parent = {start: None}
    gscore = {start: 0}
    while frontier:
        _, current = heappop(frontier)
        if current == goal:
            out = []
            node = current
            while node is not None:
                out.append(node)
                node = parent[node]
            return out[::-1]
        x, y = current
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not is_free(grid, nxt):
                continue
            next_g = gscore[current] + 1
            if next_g < gscore.get(nxt, 10**9):
                gscore[nxt] = next_g
                parent[nxt] = current
                heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heappush(frontier, (next_g + heuristic, nxt))
    raise RuntimeError("no feasible path")
