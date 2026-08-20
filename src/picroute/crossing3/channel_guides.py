from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Any

from .case_io import InstanceGeometry, NetGeometry


Cell = tuple[int, int]
Point = tuple[float, float]


@dataclass
class GuidePlan:
    paths: dict[str, list[Point]]
    cell_size_um: float
    columns: int
    rows: int
    failed_nets: list[str]
    maximum_usage: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": "coarse_capacity_aware_8_neighbor_astar",
            "cell_size_um": self.cell_size_um,
            "columns": self.columns,
            "rows": self.rows,
            "failed_nets": self.failed_nets,
            "maximum_usage": self.maximum_usage,
            "paths": {
                name: [[point[0], point[1]] for point in path]
                for name, path in self.paths.items()
            },
        }


def _orientation_vector(degrees: float) -> Point:
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def build_channel_guides(
    nets: dict[str, NetGeometry],
    obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    config: dict[str, Any],
) -> GuidePlan:
    """Create coarse routing guides without producing detailed waveguides."""

    target_columns = max(32, int(config.get("guide_target_columns", 128)))
    crossing_envelope = float(config.get("crossing_body_um", 8.0)) + 2.0 * float(
        config.get("crossing_halo_um", 4.5)
    )
    width = max(die[2] - die[0], 1.0)
    height = max(die[3] - die[1], 1.0)
    cell_size = max(crossing_envelope, width / target_columns)
    columns = max(2, int(math.ceil(width / cell_size)))
    rows = max(2, int(math.ceil(height / cell_size)))
    access = float(config.get("minimum_access_um", 10.0))

    def cell(point: Point) -> Cell:
        return (
            min(columns - 1, max(0, int((point[0] - die[0]) / cell_size))),
            min(rows - 1, max(0, int((point[1] - die[1]) / cell_size))),
        )

    def point(value: Cell) -> Point:
        return (
            die[0] + (value[0] + 0.5) * cell_size,
            die[1] + (value[1] + 0.5) * cell_size,
        )

    blocked: set[Cell] = set()
    for obstacle in obstacles:
        min_x, min_y, max_x, max_y = obstacle.bbox
        first = cell((min_x, min_y))
        second = cell((max_x, max_y))
        for x in range(first[0], second[0] + 1):
            for y in range(first[1], second[1] + 1):
                blocked.add((x, y))

    usage: dict[Cell, int] = {}
    paths: dict[str, list[Point]] = {}
    failed: list[str] = []
    neighbours = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    ordered_nets = sorted(nets.values(), key=lambda net: (-net.length, net.name))
    for net in ordered_nets:
        source_direction = _orientation_vector(net.source.orientation)
        target_direction = _orientation_vector(net.target.orientation)
        source_access = (
            net.source.x + access * source_direction[0],
            net.source.y + access * source_direction[1],
        )
        target_access = (
            net.target.x + access * target_direction[0],
            net.target.y + access * target_direction[1],
        )
        start, goal = cell(source_access), cell(target_access)
        local_blocked = blocked - {start, goal}
        serial = itertools.count()
        queue: list[tuple[float, float, int, Cell, tuple[int, int] | None]] = [
            (0.0, 0.0, next(serial), start, None)
        ]
        best: dict[tuple[Cell, tuple[int, int] | None], float] = {(start, None): 0.0}
        parent: dict[
            tuple[Cell, tuple[int, int] | None],
            tuple[Cell, tuple[int, int] | None] | None,
        ] = {(start, None): None}
        goal_state: tuple[Cell, tuple[int, int] | None] | None = None
        while queue:
            _priority, cost, _serial, current, previous_direction = heapq.heappop(
                queue
            )
            state = (current, previous_direction)
            if cost > best.get(state, math.inf) + 1e-12:
                continue
            if current == goal:
                goal_state = state
                break
            for direction in neighbours:
                candidate = (current[0] + direction[0], current[1] + direction[1])
                if (
                    candidate[0] < 0
                    or candidate[0] >= columns
                    or candidate[1] < 0
                    or candidate[1] >= rows
                    or candidate in local_blocked
                ):
                    continue
                step = math.sqrt(2.0) if direction[0] and direction[1] else 1.0
                congestion = float(usage.get(candidate, 0))
                bend = 0.35 if previous_direction not in (None, direction) else 0.0
                next_cost = cost + step + bend + 0.20 * congestion * congestion
                next_state = (candidate, direction)
                if next_cost >= best.get(next_state, math.inf) - 1e-12:
                    continue
                best[next_state] = next_cost
                parent[next_state] = state
                heuristic = math.hypot(goal[0] - candidate[0], goal[1] - candidate[1])
                heapq.heappush(
                    queue,
                    (
                        next_cost + heuristic,
                        next_cost,
                        next(serial),
                        candidate,
                        direction,
                    ),
                )
        if goal_state is None:
            failed.append(net.name)
            paths[net.name] = [
                (net.source.x, net.source.y),
                (net.target.x, net.target.y),
            ]
            continue
        cells = []
        state = goal_state
        while state is not None:
            cells.append(state[0])
            state = parent[state]
        cells.reverse()
        for item in cells:
            usage[item] = usage.get(item, 0) + 1
        polyline = [(net.source.x, net.source.y)]
        polyline.extend(point(item) for item in cells)
        polyline.append((net.target.x, net.target.y))
        paths[net.name] = polyline
    return GuidePlan(
        paths=paths,
        cell_size_um=cell_size,
        columns=columns,
        rows=rows,
        failed_nets=sorted(failed),
        maximum_usage=max(usage.values(), default=0),
    )


def point_to_polyline_distance(point: Point, polyline: list[Point]) -> float:
    if not polyline:
        return 0.0
    if len(polyline) == 1:
        return math.dist(point, polyline[0])
    best = math.inf
    for first, second in zip(polyline, polyline[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            best = min(best, math.dist(point, first))
            continue
        parameter = max(
            0.0,
            min(
                1.0,
                ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy)
                / length_squared,
            ),
        )
        projection = (first[0] + parameter * dx, first[1] + parameter * dy)
        best = min(best, math.dist(point, projection))
    return best
