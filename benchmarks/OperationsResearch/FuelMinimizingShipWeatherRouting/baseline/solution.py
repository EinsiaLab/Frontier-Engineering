from __future__ import annotations

import heapq


def solve(instance):
    start = tuple(instance["start"])
    goal = tuple(instance["goal"])
    rows = instance["grid"]
    current_field = instance["current_field"]
    wind_field = instance["wind_field"]
    latest_arrival_h = float(instance["latest_arrival_h"])

    def is_water(cell):
        x, y = cell
        return 0 <= y < len(rows) and 0 <= x < len(rows[0]) and rows[y][x] in {".", "S", "G"}

    def leg_metrics(prev, curr):
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        current_u, current_v = current_field[prev[1]][prev[0]]
        wind_u, wind_v = wind_field[prev[1]][prev[0]]
        current_along = current_u * dx + current_v * dy
        wind_along = wind_u * dx + wind_v * dy
        headwind = max(0.0, -wind_along)
        crosswind = abs(-dy * wind_u + dx * wind_v)
        speed = max(0.35, 1.0 + 0.65 * current_along - 0.45 * headwind)
        leg_time_h = 1.0 / speed
        fuel_rate = 1.05 + 0.55 * headwind + 0.20 * crosswind + 0.25 * max(0.0, -current_along)
        return leg_time_h * fuel_rate, leg_time_h

    frontier = [(0.0, 0.0, start)]
    best = {start: (0.0, 0.0)}
    parent = {start: None}
    while frontier:
        current_score, current_time, current = heapq.heappop(frontier)
        current_fuel = best[current][0]
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return {"path": path[::-1]}
        x, y = current
        for nxt in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if not is_water(nxt):
                continue
            leg_fuel, leg_time = leg_metrics(current, nxt)
            next_fuel = current_fuel + leg_fuel
            next_time = best[current][1] + leg_time
            if next_time > latest_arrival_h:
                continue
            best_cost = best.get(nxt)
            if best_cost is None or next_fuel < best_cost[0]:
                best[nxt] = (next_fuel, next_time)
                parent[nxt] = current
                heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(frontier, (next_fuel + 0.01 * heuristic, next_time, nxt))
    raise RuntimeError("no feasible path found")
