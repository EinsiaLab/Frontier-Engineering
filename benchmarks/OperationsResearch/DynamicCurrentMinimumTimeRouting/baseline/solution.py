from __future__ import annotations

import heapq


def solve(instance):
    start = tuple(instance["start"])
    goal = tuple(instance["goal"])
    rows = instance["grid"]
    depth_field = instance["depth_field"]
    current_field = instance["current_field"]

    def is_navigable(cell):
        x, y = cell
        return 0 <= y < len(rows) and 0 <= x < len(rows[0]) and rows[y][x] in {".", "S", "G"}

    def leg_time(prev, curr):
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        current_u, current_v = current_field[prev[1]][prev[0]]
        current_along = current_u * dx + current_v * dy
        depth = float(depth_field[curr[1]][curr[0]])
        shallow_penalty = max(0.0, 3.0 - depth) * 0.22
        speed = max(0.25, 1.0 + 0.9 * current_along - shallow_penalty)
        return 1.0 / speed

    frontier = [(0.0, start)]
    best = {start: 0.0}
    parent = {start: None}
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return {"path": path[::-1]}
        if current_cost > best[current]:
            continue
        x, y = current
        for nxt in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if not is_navigable(nxt):
                continue
            next_cost = current_cost + leg_time(current, nxt)
            if next_cost < best.get(nxt, float("inf")):
                best[nxt] = next_cost
                parent[nxt] = current
                heapq.heappush(frontier, (next_cost, nxt))
    raise RuntimeError("no feasible path found")
