from __future__ import annotations

from heapq import heappop, heappush


def plan_paths(grid, starts, goals):
    def is_free(cell):
        x, y = cell
        return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] != "#"

    def neighbors(cell, allow_wait=False):
        x, y = cell
        out = []
        cands = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if allow_wait:
            cands.append((x, y))
        for cand in cands:
            if is_free(cand):
                out.append(cand)
        return out

    def retrace(parent, node):
        path = []
        current = node
        while current is not None:
            path.append(current[0])
            current = parent[current]
        return path[::-1]

    def astar(start, goal, reserved_vertices, reserved_edges, max_time=60):
        frontier = [(0, 0, start)]
        parent = {(start, 0): None}
        best_time = {(start, 0): 0}
        while frontier:
            _, current_time, current = heappop(frontier)
            if best_time[(current, current_time)] != current_time:
                continue
            if current == goal:
                return retrace(parent, (current, current_time))
            if current_time >= max_time:
                continue
            for nxt in neighbors(current, allow_wait=True):
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
                heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heappush(frontier, (next_time + heuristic, next_time, nxt))
        raise RuntimeError("prioritized planning failed")

    def reserve(path, reserved_vertices, reserved_edges, horizon=60):
        for t, cell in enumerate(path):
            reserved_vertices.add((cell, t))
            if t > 0:
                reserved_edges.add(((path[t - 1], cell), t))
            if t == len(path) - 1:
                for future in range(t + 1, horizon + 1):
                    reserved_vertices.add((cell, future))

    reserved_vertices = set()
    reserved_edges = set()
    order = sorted(range(len(starts)), key=lambda idx: abs(starts[idx][0] - goals[idx][0]) + abs(starts[idx][1] - goals[idx][1]), reverse=True)
    paths = [None] * len(starts)
    for idx in order:
        path = astar(tuple(starts[idx]), tuple(goals[idx]), reserved_vertices, reserved_edges)
        paths[idx] = path
        reserve(path, reserved_vertices, reserved_edges)
    return {"paths": paths}
