from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .access_contract import effective_hard_access, port_connection_feasibility
from .case_io import InstanceGeometry, NetGeometry
from .channel_guides import point_to_polyline_distance
from .model import CrossingEvent, PlacedCrossing, Prediction
from .pcell_geometry import BASE_PORT_PAIRS, absolute_port, crossing_views
from .routing_lattice import snap_to_routing_track


@dataclass(frozen=True)
class Candidate:
    event_id: str
    center_um: tuple[float, float]
    rotation_deg: float
    net_a_ports: tuple[str, str]
    net_b_ports: tuple[str, str]
    bbox: tuple[float, float, float, float]
    halo_bbox: tuple[float, float, float, float]
    hard_access_violations: int
    cost: float


def _overlaps(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    epsilon: float = 1e-9,
) -> bool:
    return (
        min(first[2], second[2]) - max(first[0], second[0]) > epsilon
        and min(first[3], second[3]) - max(first[1], second[1]) > epsilon
    )


def _inflate(
    box: tuple[float, float, float, float], amount: float
) -> tuple[float, float, float, float]:
    return box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount


def _snap(value: float, grid: float, origin: float = 0.0) -> float:
    return snap_to_routing_track(value, origin, grid)


def _ring_offsets(radius: float, step: float) -> list[tuple[float, float]]:
    limit = max(0, int(math.ceil(radius / step)))
    result = [(0.0, 0.0)]
    for ring in range(1, limit + 1):
        values = []
        for x in range(-ring, ring + 1):
            values.append((x, -ring))
            values.append((x, ring))
        for y in range(-ring + 1, ring):
            values.append((-ring, y))
            values.append((ring, y))
        result.extend((x * step, y * step) for x, y in sorted(set(values)))
    return result


def _axis_distance(first: float, second: float) -> float:
    delta = abs((first - second) % 180.0)
    return min(delta, 180.0 - delta)


def _terminal_access_proxy(
    event: CrossingEvent,
    center: tuple[float, float],
    view: dict[str, Any],
    net: NetGeometry,
    pair: tuple[str, str],
    rank: int,
    last_rank: int,
    minimum_access_um: float,
    minimum_radius_um: float,
    direct_threshold_um: float,
    samples: int,
) -> tuple[float, bool]:
    states = (pair, (pair[1], pair[0]))
    alternatives = []
    for entry, exit_name in states:
        cost = 0.0
        legal = True
        if rank == 0:
            entry_point, entry_angle = absolute_port(center, view, entry)
            feasibility = port_connection_feasibility(
                (net.source.x, net.source.y),
                net.source.orientation,
                entry_point,
                entry_angle,
                minimum_access_um,
                direct_threshold_um,
                minimum_radius_um,
                samples=samples,
            )
            cost += float(feasibility["penalty"])
            legal = legal and not effective_hard_access(feasibility)
        if rank == last_rank:
            exit_point, exit_angle = absolute_port(center, view, exit_name)
            feasibility = port_connection_feasibility(
                exit_point,
                exit_angle,
                (net.target.x, net.target.y),
                net.target.orientation,
                minimum_access_um,
                direct_threshold_um,
                minimum_radius_um,
                samples=samples,
            )
            cost += float(feasibility["penalty"])
            legal = legal and not effective_hard_access(feasibility)
        alternatives.append((cost, legal))
    legal_costs = [cost for cost, legal in alternatives if legal]
    return (min(legal_costs), True) if legal_costs else (min(cost for cost, _ in alternatives), False)


def _candidate_domains(
    events: list[CrossingEvent],
    nets: dict[str, NetGeometry],
    obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
    config: dict[str, Any],
    committed: list[Candidate],
    guide_paths: dict[str, list[tuple[float, float]]] | None,
) -> tuple[dict[str, list[Candidate]], dict[str, dict[str, Any]]]:
    grid = float(config.get("grid_um", 2.0))
    limit = int(config.get("candidate_limit", 48))
    initial_radius = float(config.get("candidate_radius_um", 80.0))
    maximum_radius = max(
        initial_radius,
        float(config.get("candidate_max_radius_um", 2.0 * initial_radius)),
    )
    expansion_factor = max(1.1, float(config.get("candidate_expansion_factor", 1.5)))
    minimum_domain_size = min(
        limit, max(1, int(config.get("minimum_legal_candidates", 12)))
    )
    ring_step = max(grid, float(config.get("candidate_ring_step_um", 4.0)))
    halo = float(manifest.get("halo_um", 0.0))
    fixed_clearance = max(
        halo, float(config.get("minimum_access_um", 10.0))
    )
    views = crossing_views(manifest)
    obstacle_boxes = [geometry.bbox for geometry in obstacles]
    minimum_access = float(config.get("minimum_access_um", 10.0))
    minimum_radius = float(config.get("bend_radius_um", 5.0))
    direct_threshold = float(
        config.get(
            "short_direct_access_threshold_um",
            2.0 * (minimum_access + minimum_radius),
        )
    )
    access_samples = max(16, int(config.get("placement_access_samples", 32)))
    expansion_radii = [initial_radius]
    while expansion_radii[-1] < maximum_radius - 1e-9:
        expanded = max(
            expansion_radii[-1] + ring_step,
            expansion_radii[-1] * expansion_factor,
        )
        expansion_radii.append(min(maximum_radius, expanded))
    maximum_rank: dict[str, int] = defaultdict(int)
    for event in events:
        maximum_rank[event.net_a] = max(maximum_rank[event.net_a], event.order_on_net_a)
        maximum_rank[event.net_b] = max(maximum_rank[event.net_b], event.order_on_net_b)
    domains: dict[str, list[Candidate]] = {}
    domain_audits: dict[str, dict[str, Any]] = {}
    for event in events:
        net_a = nets[event.net_a]
        net_b = nets[event.net_b]
        angle_a = math.degrees(math.atan2(net_a.vector[1], net_a.vector[0])) % 180.0
        angle_b = math.degrees(math.atan2(net_b.vector[1], net_b.vector[0])) % 180.0
        raw: list[Candidate] = []
        hard_rejected = 0
        geometry_rejected = 0
        radius_used = initial_radius
        for radius_used in expansion_radii:
            raw = []
            hard_rejected = 0
            geometry_rejected = 0
            for dx, dy in _ring_offsets(radius_used, ring_step):
                center = (
                    _snap(event.ideal_center_um[0] + dx, grid, die[0]),
                    _snap(event.ideal_center_um[1] + dy, grid, die[1]),
                )
                for rotation, view in views.items():
                    local = [float(value) for value in view["bbox_centered_um"]]
                    box = (
                        center[0] + local[0],
                        center[1] + local[1],
                        center[0] + local[2],
                        center[1] + local[3],
                    )
                    halo_box = _inflate(box, halo)
                    fixed_clearance_box = _inflate(box, fixed_clearance)
                    if (
                        box[0] < die[0]
                        or box[1] < die[1]
                        or box[2] > die[2]
                        or box[3] > die[3]
                        or any(
                            _overlaps(fixed_clearance_box, fixed)
                            for fixed in obstacle_boxes
                        )
                        or any(
                            _overlaps(halo_box, other.halo_bbox)
                            for other in committed
                        )
                    ):
                        geometry_rejected += 2
                        continue
                    pair_axes = [
                        float(view["ports"][pair[0]]["orientation_deg"]) % 180.0
                        for pair in BASE_PORT_PAIRS
                    ]
                    for a_pair_index in (0, 1):
                        b_pair_index = 1 - a_pair_index
                        orientation_cost = (
                            _axis_distance(angle_a, pair_axes[a_pair_index]) ** 2
                            + _axis_distance(angle_b, pair_axes[b_pair_index]) ** 2
                        )
                        displacement = math.dist(center, event.ideal_center_um)
                        preferred = (
                            0.0
                            if math.isclose(rotation, event.preferred_rotation_deg)
                            else 1.0
                        )
                        access_a, access_a_legal = _terminal_access_proxy(
                            event,
                            center,
                            view,
                            net_a,
                            BASE_PORT_PAIRS[a_pair_index],
                            event.order_on_net_a,
                            maximum_rank[event.net_a],
                            minimum_access,
                            minimum_radius,
                            direct_threshold,
                            access_samples,
                        )
                        access_b, access_b_legal = _terminal_access_proxy(
                            event,
                            center,
                            view,
                            net_b,
                            BASE_PORT_PAIRS[b_pair_index],
                            event.order_on_net_b,
                            maximum_rank[event.net_b],
                            minimum_access,
                            minimum_radius,
                            direct_threshold,
                            access_samples,
                        )
                        if not access_a_legal or not access_b_legal:
                            hard_rejected += 1
                            continue
                        guide_cost = 0.0
                        if guide_paths:
                            guide_cost = point_to_polyline_distance(
                                center, guide_paths.get(event.net_a, [])
                            ) + point_to_polyline_distance(
                                center, guide_paths.get(event.net_b, [])
                            )
                        physical_cost = (
                            displacement * displacement
                            + 0.05 * orientation_cost
                            + 50.0 * (access_a + access_b)
                            + 0.25 * guide_cost * guide_cost
                            + preferred
                        )
                        raw.append(
                            Candidate(
                                event_id=event.event_id,
                                center_um=center,
                                rotation_deg=float(rotation),
                                net_a_ports=BASE_PORT_PAIRS[a_pair_index],
                                net_b_ports=BASE_PORT_PAIRS[b_pair_index],
                                bbox=box,
                                halo_bbox=halo_box,
                                hard_access_violations=0,
                                cost=physical_cost,
                            )
                        )
            if len(raw) >= minimum_domain_size:
                break
        raw.sort(
            key=lambda item: (
                item.hard_access_violations,
                round(item.cost, 12),
                item.center_um,
                abs(item.rotation_deg),
                item.net_a_ports,
            )
        )
        domains[event.event_id] = raw[:limit]
        domain_audits[event.event_id] = {
            "initial_radius_um": initial_radius,
            "radius_used_um": radius_used,
            "maximum_radius_um": maximum_radius,
            "legal_candidate_count_before_limit": len(raw),
            "selected_domain_size": len(domains[event.event_id]),
            "hard_access_rejected": hard_rejected,
            "geometry_rejected": geometry_rejected,
        }
    return domains, domain_audits


def _event_components(events: list[CrossingEvent]) -> list[list[CrossingEvent]]:
    events_by_net: dict[str, set[str]] = defaultdict(set)
    by_id = {event.event_id: event for event in events}
    for event in events:
        events_by_net[event.net_a].add(event.event_id)
        events_by_net[event.net_b].add(event.event_id)
    adjacency: dict[str, set[str]] = {event.event_id: set() for event in events}
    for identifiers in events_by_net.values():
        for identifier in identifiers:
            adjacency[identifier].update(identifiers - {identifier})
    remaining = set(adjacency)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = []
        while stack:
            current = stack.pop()
            component.append(by_id[current])
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        result.append(sorted(component, key=lambda item: item.event_id))
    result.sort(key=lambda values: (-len(values), values[0].event_id))
    return result


def _rank(event: CrossingEvent, net_name: str) -> int:
    return event.order_on_net_a if event.net_a == net_name else event.order_on_net_b


def _pair_for_net(candidate: Candidate, event: CrossingEvent, net_name: str) -> tuple[str, str]:
    return candidate.net_a_ports if event.net_a == net_name else candidate.net_b_ports


def _polyline_coordinate(
    point: tuple[float, float], path: list[tuple[float, float]]
) -> float:
    if len(path) < 2:
        return 0.0
    best_distance = math.inf
    best_coordinate = 0.0
    cumulative = 0.0
    for first, second in zip(path, path[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-18:
            continue
        fraction = max(
            0.0,
            min(
                1.0,
                ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy)
                / length_squared,
            ),
        )
        projection = (first[0] + fraction * dx, first[1] + fraction * dy)
        distance = math.dist(point, projection)
        segment_length = math.sqrt(length_squared)
        coordinate = cumulative + fraction * segment_length
        if (distance, coordinate) < (best_distance, best_coordinate):
            best_distance = distance
            best_coordinate = coordinate
        cumulative += segment_length
    return best_coordinate


def _order_conflict(
    first_event: CrossingEvent,
    first: Candidate,
    second_event: CrossingEvent,
    second: Candidate,
    nets: dict[str, NetGeometry],
    guide_paths: dict[str, list[tuple[float, float]]] | None,
    order_spacing_um: float,
) -> bool:
    shared = {first_event.net_a, first_event.net_b} & {
        second_event.net_a,
        second_event.net_b,
    }
    for net_name in shared:
        net = nets[net_name]
        path = (guide_paths or {}).get(net_name, [])
        if len(path) >= 2:
            first_projection = _polyline_coordinate(first.center_um, path)
            second_projection = _polyline_coordinate(second.center_um, path)
        else:
            dx, dy = net.vector
            length = max(math.hypot(dx, dy), 1e-9)
            direction = (dx / length, dy / length)
            first_projection = (
                first.center_um[0] * direction[0]
                + first.center_um[1] * direction[1]
            )
            second_projection = (
                second.center_um[0] * direction[0]
                + second.center_um[1] * direction[1]
            )
        first_rank = _rank(first_event, net_name)
        second_rank = _rank(second_event, net_name)
        if (
            first_rank < second_rank
            and first_projection > second_projection - order_spacing_um
        ):
            return True
        if (
            second_rank < first_rank
            and second_projection > first_projection - order_spacing_um
        ):
            return True
    return False


def _adjacent_access_conflict(
    first_event: CrossingEvent,
    first: Candidate,
    second_event: CrossingEvent,
    second: Candidate,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    shared = {first_event.net_a, first_event.net_b} & {
        second_event.net_a,
        second_event.net_b,
    }
    if not shared:
        return False
    views = crossing_views(manifest)
    minimum_access = float(config.get("minimum_access_um", 10.0))
    minimum_radius = float(config.get("bend_radius_um", 5.0))
    direct_threshold = float(
        config.get(
            "short_direct_access_threshold_um",
            2.0 * (minimum_access + minimum_radius),
        )
    )
    samples = max(16, int(config.get("placement_access_samples", 32)))
    for net_name in shared:
        first_rank = _rank(first_event, net_name)
        second_rank = _rank(second_event, net_name)
        if abs(first_rank - second_rank) != 1:
            continue
        if first_rank < second_rank:
            earlier_event, earlier = first_event, first
            later_event, later = second_event, second
        else:
            earlier_event, earlier = second_event, second
            later_event, later = first_event, first
        earlier_pair = _pair_for_net(earlier, earlier_event, net_name)
        later_pair = _pair_for_net(later, later_event, net_name)
        earlier_view = views[float(earlier.rotation_deg)]
        later_view = views[float(later.rotation_deg)]
        feasible = False
        for _earlier_entry, earlier_exit in (
            earlier_pair,
            (earlier_pair[1], earlier_pair[0]),
        ):
            first_point, first_angle = absolute_port(
                earlier.center_um, earlier_view, earlier_exit
            )
            for later_entry, _later_exit in (
                later_pair,
                (later_pair[1], later_pair[0]),
            ):
                second_point, second_angle = absolute_port(
                    later.center_um, later_view, later_entry
                )
                audit = port_connection_feasibility(
                    first_point,
                    first_angle,
                    second_point,
                    second_angle,
                    minimum_access,
                    direct_threshold,
                    minimum_radius,
                    samples=samples,
                )
                if not effective_hard_access(audit):
                    feasible = True
                    break
            if feasible:
                break
        if not feasible:
            return True
    return False


def _solve_component(
    events: list[CrossingEvent],
    domains: dict[str, list[Candidate]],
    nets: dict[str, NetGeometry],
    manifest: dict[str, Any],
    config: dict[str, Any],
    guide_paths: dict[str, list[tuple[float, float]]] | None,
    time_limit_s: float,
) -> tuple[list[Candidate], dict[str, Any]]:
    variables: list[Candidate] = []
    event_ranges: dict[str, list[int]] = {}
    for event in events:
        values = domains[event.event_id]
        if not values:
            raise RuntimeError(f"No legal candidate for {event.event_id}")
        indices = list(range(len(variables), len(variables) + len(values)))
        variables.extend(values)
        event_ranges[event.event_id] = indices

    conflict_pairs: list[tuple[int, int]] = []
    access_conflict_count = 0
    order_spacing_um = max(0.0, float(config.get("topology_order_spacing_um", 0.0)))
    for first_index, first_event in enumerate(events):
        for second_event in events[first_index + 1 :]:
            shared = bool(
                {first_event.net_a, first_event.net_b}
                & {second_event.net_a, second_event.net_b}
            )
            for first_variable in event_ranges[first_event.event_id]:
                first = variables[first_variable]
                for second_variable in event_ranges[second_event.event_id]:
                    second = variables[second_variable]
                    geometry_conflict = _overlaps(
                        first.bbox if shared else first.halo_bbox,
                        second.bbox if shared else second.halo_bbox,
                    )
                    order_conflict = _order_conflict(
                            first_event,
                            first,
                            second_event,
                            second,
                            nets,
                            guide_paths,
                            order_spacing_um,
                        )
                    access_conflict = False
                    if not geometry_conflict and not order_conflict:
                        access_conflict = _adjacent_access_conflict(
                            first_event,
                            first,
                            second_event,
                            second,
                            manifest,
                            config,
                        )
                    if geometry_conflict or order_conflict or access_conflict:
                        conflict_pairs.append((first_variable, second_variable))
                        access_conflict_count += int(access_conflict)

    rows = len(events) + len(conflict_pairs)
    matrix = lil_matrix((rows, len(variables)), dtype=float)
    lower = np.full(rows, -np.inf)
    upper = np.ones(rows)
    row = 0
    for event in events:
        for variable in event_ranges[event.event_id]:
            matrix[row, variable] = 1.0
        lower[row] = 1.0
        upper[row] = 1.0
        row += 1
    for first, second in conflict_pairs:
        matrix[row, first] = 1.0
        matrix[row, second] = 1.0
        upper[row] = 1.0
        row += 1

    objective = np.array([candidate.cost for candidate in variables], dtype=float)
    result = milp(
        objective,
        integrality=np.ones(len(variables)),
        bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": float(time_limit_s), "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(
            "HCTP component MILP failed: " + str(getattr(result, "message", "unknown"))
        )
    selected = []
    for event in events:
        best = max(event_ranges[event.event_id], key=lambda index: result.x[index])
        if result.x[best] < 0.5:
            raise RuntimeError(f"MILP did not select a candidate for {event.event_id}")
        selected.append(variables[best])
    return selected, {
        "event_count": len(events),
        "variable_count": len(variables),
        "conflict_count": len(conflict_pairs),
        "access_conflict_count": access_conflict_count,
        "topology_order_coordinate": "guide_polyline_arclength",
        "topology_order_spacing_um": order_spacing_um,
        "objective": float(result.fun),
        "status": int(result.status),
        "message": str(result.message),
    }


def place_crossings(
    prediction: Prediction,
    nets: dict[str, NetGeometry],
    obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
    config: dict[str, Any],
    guide_paths: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[PlacedCrossing], dict[str, Any]]:
    committed: list[Candidate] = []
    reports = []
    event_by_id = {event.event_id: event for event in prediction.events}
    for component in _event_components(prediction.events):
        domains, domain_audits = _candidate_domains(
            component,
            nets,
            obstacles,
            die,
            manifest,
            config,
            committed,
            guide_paths,
        )
        selected, report = _solve_component(
            component,
            domains,
            nets,
            manifest,
            config,
            guide_paths,
            float(config.get("solver_time_limit_s", 120.0)),
        )
        report["topology_spacing_relaxed"] = False
        report["projection_order_constraint"] = True
        committed.extend(selected)
        report["events"] = [event.event_id for event in component]
        report["domain_sizes"] = {
            event.event_id: len(domains[event.event_id]) for event in component
        }
        report["candidate_domain_audit"] = domain_audits
        report["hard_access_violation_candidates"] = {
            event.event_id: 0 for event in component
        }
        reports.append(report)

    placed = []
    for candidate in sorted(committed, key=lambda item: item.event_id):
        event = event_by_id[candidate.event_id]
        event.net_a_ports = candidate.net_a_ports
        event.net_b_ports = candidate.net_b_ports
        placed.append(
            PlacedCrossing(
                event=event,
                center_um=candidate.center_um,
                rotation_deg=candidate.rotation_deg,
                legal=True,
                displacement_um=math.dist(candidate.center_um, event.ideal_center_um),
            )
        )
    return placed, {
        "schema_version": 2,
        "algorithm": "hctp_access_constrained_component_milp_v2",
        "component_count": len(reports),
        "crossing_count": len(placed),
        "components": reports,
        "original_instances_moved": 0,
    }
