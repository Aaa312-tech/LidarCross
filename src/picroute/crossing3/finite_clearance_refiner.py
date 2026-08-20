from __future__ import annotations

import math
from typing import Any

from .case_io import InstanceGeometry, NetGeometry
from .model import CrossingEvent, PlacedCrossing, Prediction
from .placement_legal import placement_is_legal
from .predictor import _assign_net_orders, _orientation_assignment


Point = tuple[float, float]


def _cross(first: Point, second: Point) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _line_parameters(first: NetGeometry, second: NetGeometry) -> tuple[float, float, Point] | None:
    p = (first.source.x, first.source.y)
    q = (second.source.x, second.source.y)
    r = first.vector
    s = second.vector
    denominator = _cross(r, s)
    if abs(denominator) <= 1e-9:
        return None
    q_minus_p = (q[0] - p[0], q[1] - p[1])
    first_parameter = _cross(q_minus_p, s) / denominator
    second_parameter = _cross(q_minus_p, r) / denominator
    return (
        first_parameter,
        second_parameter,
        (
            p[0] + first_parameter * r[0],
            p[1] + first_parameter * r[1],
        ),
    )


def _proper_segment_intersection(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> Point | None:
    r = (first_end[0] - first_start[0], first_end[1] - first_start[1])
    s = (second_end[0] - second_start[0], second_end[1] - second_start[1])
    denominator = _cross(r, s)
    if abs(denominator) <= 1e-9:
        return None
    delta = (
        second_start[0] - first_start[0],
        second_start[1] - first_start[1],
    )
    first_parameter = _cross(delta, s) / denominator
    second_parameter = _cross(delta, r) / denominator
    epsilon = 1e-9
    if not (
        epsilon < first_parameter < 1.0 - epsilon
        and epsilon < second_parameter < 1.0 - epsilon
    ):
        return None
    return (
        first_start[0] + first_parameter * r[0],
        first_start[1] + first_parameter * r[1],
    )


def _polyline_intersections(first: list[Point], second: list[Point]) -> list[Point]:
    result: list[Point] = []
    for first_start, first_end in zip(first, first[1:]):
        for second_start, second_end in zip(second, second[1:]):
            point = _proper_segment_intersection(
                first_start, first_end, second_start, second_end
            )
            if point is None:
                continue
            if not any(math.dist(point, existing) <= 1e-6 for existing in result):
                result.append(point)
    return result


def _placed_polyline(
    net_name: str,
    prediction: Prediction,
    placed_by_id: dict[str, PlacedCrossing],
    nets: dict[str, NetGeometry],
) -> list[Point]:
    net = nets[net_name]
    return [
        (net.source.x, net.source.y),
        *[
            placed_by_id[event_id].center_um
            for event_id in prediction.net_orders.get(net_name, [])
            if event_id in placed_by_id
        ],
        (net.target.x, net.target.y),
    ]


def _clearance_center(
    physical_intersection: Point,
    near_net: NetGeometry,
    terminal_parameter: float,
    grid_um: float,
) -> Point:
    dx, dy = near_net.vector
    length_squared = max(dx * dx + dy * dy, 1e-12)
    projection = (
        (physical_intersection[0] - near_net.source.x) * dx
        + (physical_intersection[1] - near_net.source.y) * dy
    ) / length_squared
    projected = (
        near_net.source.x + projection * dx,
        near_net.source.y + projection * dy,
    )
    length = math.sqrt(length_squared)
    inward = (dx / length, dy / length)
    if terminal_parameter > 1.0:
        inward = (-inward[0], -inward[1])
    raw = (
        projected[0] + grid_um * inward[0],
        projected[1] + grid_um * inward[1],
    )
    return (
        round(raw[0] / grid_um) * grid_um,
        round(raw[1] / grid_um) * grid_um,
    )


def add_finite_clearance_crossings(
    prediction: Prediction,
    placed: list[PlacedCrossing],
    nets: dict[str, NetGeometry],
    fixed_obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[PlacedCrossing], dict[str, Any]]:
    """Materialize near-terminal intersections hidden by infinite chords.

    A pair is eligible only when its terminal chords miss by a technology
    clearance at exactly one endpoint *and* the already placed crossing paths
    have one proper physical intersection.  This second condition prevents a
    terminal-line extrapolation from inventing an unnecessary crossing.
    """

    enabled = bool(config.get("finite_clearance_refinement", True))
    if not enabled:
        return placed, {
            "enabled": False,
            "algorithm": "finite_terminal_clearance_path_intersection_v1",
            "crossings": [],
        }
    grid = float(config.get("grid_um", 2.0))
    clearance = float(config.get("crossing_halo_um", 4.5))
    existing_pairs = {event.pair() for event in prediction.events}
    placed_by_id = {item.event.event_id: item for item in placed}
    polylines = {
        name: _placed_polyline(name, prediction, placed_by_id, nets)
        for name in nets
    }
    inherited_orders = {
        name: list(order) for name, order in prediction.net_orders.items()
    }
    inherited_ranks = {
        event.event_id: (event.order_on_net_a, event.order_on_net_b)
        for event in prediction.events
    }
    added: list[PlacedCrossing] = []
    audits = []
    names = sorted(nets)
    for index, first_name in enumerate(names):
        first = nets[first_name]
        for second_name in names[index + 1 :]:
            pair = (first_name, second_name)
            if pair in existing_pairs:
                continue
            second = nets[second_name]
            if {
                first.source_name,
                first.target_name,
            } & {second.source_name, second.target_name}:
                continue
            parameters = _line_parameters(first, second)
            if parameters is None:
                continue
            first_parameter, second_parameter, terminal_intersection = parameters
            first_proper = 0.0 < first_parameter < 1.0
            second_proper = 0.0 < second_parameter < 1.0
            if first_proper == second_proper:
                continue
            near_name, near_net, near_parameter = (
                (second_name, second, second_parameter)
                if first_proper
                else (first_name, first, first_parameter)
            )
            extension = (
                -near_parameter * near_net.length
                if near_parameter < 0.0
                else (near_parameter - 1.0) * near_net.length
            )
            if extension < -1e-9 or extension > clearance + 1e-9:
                continue
            physical = _polyline_intersections(
                polylines[first_name], polylines[second_name]
            )
            if len(physical) != 1:
                continue
            center = _clearance_center(physical[0], near_net, near_parameter, grid)
            rotation, pair_a, pair_b = _orientation_assignment(first, second)
            event = CrossingEvent(
                event_id=f"xc__{first_name}__{second_name}__clearance",
                net_a=first_name,
                net_b=second_name,
                ideal_center_um=center,
                evidence=[
                    "finite_terminal_clearance",
                    "placed_path_proper_intersection",
                    "odd_pair_parity",
                    "technology_grid_inward_anchor",
                ],
                confidence="high",
                preferred_rotation_deg=rotation,
                net_a_ports=pair_a,
                net_b_ports=pair_b,
            )
            candidate = PlacedCrossing(
                event=event,
                center_um=center,
                rotation_deg=rotation,
                legal=True,
                displacement_um=0.0,
            )
            legal = placement_is_legal(
                candidate,
                [*placed, *added],
                fixed_obstacles,
                die,
                manifest,
            )
            audits.append(
                {
                    "pair": list(pair),
                    "near_terminal_net": near_name,
                    "terminal_line_intersection_um": list(terminal_intersection),
                    "terminal_parameter": near_parameter,
                    "extension_um": extension,
                    "placed_path_intersection_um": list(physical[0]),
                    "candidate_center_um": list(center),
                    "legal": legal,
                }
            )
            if not legal:
                continue
            added.append(candidate)
            existing_pairs.add(pair)

    if added:
        affected_nets = {
            name
            for item in added
            for name in (item.event.net_a, item.event.net_b)
        }
        prediction.events.extend(item.event for item in added)
        prediction.net_orders = _assign_net_orders(prediction.events, nets)
        for name, order in inherited_orders.items():
            if name not in affected_nets:
                prediction.net_orders[name] = order
        for event in prediction.events:
            inherited = inherited_ranks.get(event.event_id)
            if inherited is None:
                continue
            if event.net_a not in affected_nets:
                event.order_on_net_a = inherited[0]
            if event.net_b not in affected_nets:
                event.order_on_net_b = inherited[1]
        for item in added:
            pair = item.event.pair()
            prediction.parity_contract["|".join(pair)] = {
                "nets": list(pair),
                "required_parity": "odd",
                "predicted_count": 1,
                "parity_satisfied": True,
            }
        prediction.diagnostics.append(
            "finite_clearance_crossings_added=" + str(len(added))
        )

    return [*placed, *added], {
        "enabled": True,
        "algorithm": "finite_terminal_clearance_path_intersection_v1",
        "crossing_count": len(added),
        "crossings": audits,
    }

