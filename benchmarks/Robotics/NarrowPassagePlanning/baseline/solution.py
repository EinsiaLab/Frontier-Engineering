from __future__ import annotations

from heapq import heappop, heappush


def plan_path(grid, start, goal):
    frontier = [(0, tuple(start))]
    parent = {tuple(start): None}
    gscore = {tuple(start): 0}
    while frontier:
        _, current = heappop(frontier)
        if current == tuple(goal):
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return {"path": path[::-1]}
        x, y = current
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nx, ny = nxt
            if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])) or grid[ny][nx] == "#":
                continue
            next_g = gscore[current] + 1
            if next_g < gscore.get(nxt, 10**9):
                gscore[nxt] = next_g
                parent[nxt] = current
                heuristic = abs(nx - goal[0]) + abs(ny - goal[1])
                heappush(frontier, (next_g + heuristic, nxt))
    raise RuntimeError("no feasible path")
