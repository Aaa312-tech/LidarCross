from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import traceback
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from .audit import audit_frontend
from .benchmark_source import BenchmarkRef, sha256_file
from .case_io import NetGeometry, absolute_ports, die_area, fixed_obstacles, load_case
from .direction_solver import solve_net_directions
from .materialize import materialize_case, write_json
from .model import PlacedCrossing, Prediction
from .pcell_geometry import BASE_PORT_PAIRS, crossing_views
from .pdk_angle_contract import ACCESS_LATERAL_TOLERANCE_UM
from .routing_lattice import snap_to_routing_track


def _snap_backend_center(
    x: float, y: float, config: dict[str, Any]
) -> tuple[float, float]:
    technology = config["technology"]
    grid = float(technology.get("grid_um", 2.0))
    origin = technology.get("routing_grid_origin_um", [0.0, 0.0])
    return (
        snap_to_routing_track(x, float(origin[0]), grid),
        snap_to_routing_track(y, float(origin[1]), grid),
    )


def _axis_distance(first: float, second: float) -> float:
    delta = abs((first - second) % 180.0)
    return min(delta, 180.0 - delta)


def _orientation_states(
    item: PlacedCrossing,
    nets: dict[str, NetGeometry],
    manifest: dict[str, Any],
) -> list[tuple[float, tuple[str, str], tuple[str, str]]]:
    event = item.event
    angles = []
    for name in (event.net_a, event.net_b):
        dx, dy = nets[name].vector
        angles.append(math.degrees(math.atan2(dy, dx)) % 180.0)
    values = []
    for rotation, view in crossing_views(manifest).items():
        axes = [
            float(view["ports"][pair[0]]["orientation_deg"]) % 180.0
            for pair in BASE_PORT_PAIRS
        ]
        for a_index in (0, 1):
            b_index = 1 - a_index
            score = _axis_distance(angles[0], axes[a_index]) ** 2 + _axis_distance(
                angles[1], axes[b_index]
            ) ** 2
            values.append(
                (
                    score,
                    abs(float(rotation)),
                    float(rotation),
                    BASE_PORT_PAIRS[a_index],
                    BASE_PORT_PAIRS[b_index],
                )
            )
    return [(rotation, pair_a, pair_b) for _s, _a, rotation, pair_a, pair_b in sorted(values)]


def _state(item: PlacedCrossing) -> tuple[float, tuple[str, str], tuple[str, str]]:
    return (
        float(item.rotation_deg),
        tuple(item.event.net_a_ports),
        tuple(item.event.net_b_ports),
    )


def _center_states(
    item: PlacedCrossing,
    nets: dict[str, NetGeometry],
    config: dict[str, Any],
    failing_points: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Generate deterministic technology-scale escape moves in several axes."""

    event = item.event
    terminals = []
    for name in (event.net_a, event.net_b):
        terminals.extend(
            [
                (nets[name].source.x, nets[name].source.y),
                (nets[name].target.x, nets[name].target.y),
            ]
        )
    reference_points = failing_points or terminals
    unit_escapes = []
    for point in reference_points:
        escape = (item.center_um[0] - point[0], item.center_um[1] - point[1])
        norm = math.hypot(*escape)
        if norm > 1e-9:
            unit_escapes.append((escape[0] / norm, escape[1] / norm))
    average = (
        sum(value[0] for value in unit_escapes),
        sum(value[1] for value in unit_escapes),
    )
    length = math.hypot(*average)
    if length <= 1e-9:
        mean = (
            nets[event.net_a].vector[0] + nets[event.net_b].vector[0],
            nets[event.net_a].vector[1] + nets[event.net_b].vector[1],
        )
        mean_length = max(math.hypot(*mean), 1e-9)
        average = (-mean[1] / mean_length, mean[0] / mean_length)
    else:
        average = (average[0] / length, average[1] / length)
    technology = config["technology"]
    step = float(technology.get("bend_radius_um", 5.0)) + float(
        technology.get("waveguide_spacing_um", 1.0)
    )
    grid = float(technology.get("grid_um", 2.0))
    configured_radius = float(
        config["placement"].get(
            "candidate_max_radius_um",
            config["placement"].get("candidate_radius_um", 80.0),
        )
    )
    radius = max(configured_radius, item.displacement_um + 2.0 * step)
    vectors = [average]
    for name in (event.net_a, event.net_b):
        dx, dy = nets[name].vector
        norm = math.hypot(dx, dy)
        if norm > 1e-9:
            vectors.extend(((-dy / norm, dx / norm), (dy / norm, -dx / norm)))
    vectors.extend(
        (
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (math.sqrt(0.5), math.sqrt(0.5)),
            (math.sqrt(0.5), -math.sqrt(0.5)),
            (-math.sqrt(0.5), math.sqrt(0.5)),
            (-math.sqrt(0.5), -math.sqrt(0.5)),
        )
    )
    result = []
    seen = {item.center_um}
    for scale in (1.0, 2.0):
        for vector in vectors:
            center = _snap_backend_center(
                item.center_um[0] + scale * step * vector[0],
                item.center_um[1] + scale * step * vector[1],
                config,
            )
            if (
                center not in seen
                and math.dist(center, event.ideal_center_um) <= radius + 1e-9
            ):
                seen.add(center)
                result.append(center)
    return result


def _cluster_center_states(
    items: list[PlacedCrossing], config: dict[str, Any]
) -> list[tuple[tuple[str, tuple[float, float]], ...]]:
    if len(items) < 2:
        return []
    technology = config["technology"]
    step = float(technology.get("bend_radius_um", 5.0)) + float(
        technology.get("waveguide_spacing_um", 1.0)
    )
    grid = float(technology.get("grid_um", 2.0))
    configured_radius = float(
        config["placement"].get(
            "candidate_max_radius_um",
            config["placement"].get("candidate_radius_um", 80.0),
        )
    )
    centroid = (
        sum(item.center_um[0] for item in items) / len(items),
        sum(item.center_um[1] for item in items) / len(items),
    )

    def snapped(item: PlacedCrossing, vector: tuple[float, float]) -> tuple[float, float]:
        center = _snap_backend_center(
            item.center_um[0] + step * vector[0],
            item.center_um[1] + step * vector[1],
            config,
        )
        return center

    states: list[tuple[tuple[str, tuple[float, float]], ...]] = []
    outward = []
    for index, item in enumerate(items):
        vector = (item.center_um[0] - centroid[0], item.center_um[1] - centroid[1])
        length = math.hypot(*vector)
        if length <= 1e-9:
            angle = 2.0 * math.pi * index / len(items)
            vector = (math.cos(angle), math.sin(angle))
        else:
            vector = (vector[0] / length, vector[1] / length)
        outward.append((item.event.event_id, snapped(item, vector)))
    states.append(tuple(sorted(outward)))

    for vector in (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (math.sqrt(0.5), math.sqrt(0.5)),
        (math.sqrt(0.5), -math.sqrt(0.5)),
        (-math.sqrt(0.5), math.sqrt(0.5)),
        (-math.sqrt(0.5), -math.sqrt(0.5)),
    ):
        translated = tuple(
            sorted((item.event.event_id, snapped(item, vector)) for item in items)
        )
        states.append(translated)
    result = []
    seen = set()
    for state in states:
        if state in seen:
            continue
        if all(
            math.dist(
                center,
                next(
                    item.event.ideal_center_um
                    for item in items
                    if item.event.event_id == event_id
                ),
            )
            <= max(
                configured_radius,
                next(
                    item.displacement_um
                    for item in items
                    if item.event.event_id == event_id
                )
                + 2.0 * step,
            )
            + 1e-9
            for event_id, center in state
        ):
            seen.add(state)
            result.append(state)
    return result


def _geometry_legal_after_centers(
    items: list[PlacedCrossing],
    state: tuple[tuple[str, tuple[float, float]], ...],
    fixed_boxes: list[tuple[float, float, float, float]],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
) -> bool:
    updates = dict(state)
    views = crossing_views(manifest)
    halo = float(manifest.get("halo_um", 0.0))
    boxes: dict[str, tuple[float, float, float, float]] = {}
    by_id = {item.event.event_id: item for item in items}

    def overlaps(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> bool:
        return (
            min(first[2], second[2]) - max(first[0], second[0]) > 1e-9
            and min(first[3], second[3]) - max(first[1], second[1]) > 1e-9
        )

    def inflate(
        box: tuple[float, float, float, float], amount: float
    ) -> tuple[float, float, float, float]:
        return (
            box[0] - amount,
            box[1] - amount,
            box[2] + amount,
            box[3] + amount,
        )

    for item in items:
        center = updates.get(item.event.event_id, item.center_um)
        local = views[float(item.rotation_deg)]["bbox_centered_um"]
        box = (
            center[0] + float(local[0]),
            center[1] + float(local[1]),
            center[0] + float(local[2]),
            center[1] + float(local[3]),
        )
        if (
            box[0] < die[0]
            or box[1] < die[1]
            or box[2] > die[2]
            or box[3] > die[3]
            or any(overlaps(inflate(box, halo), fixed) for fixed in fixed_boxes)
        ):
            return False
        boxes[item.event.event_id] = box
    identifiers = sorted(boxes)
    for index, first_id in enumerate(identifiers):
        first_nets = {
            by_id[first_id].event.net_a,
            by_id[first_id].event.net_b,
        }
        for second_id in identifiers[index + 1 :]:
            shared = bool(
                first_nets
                & {
                    by_id[second_id].event.net_a,
                    by_id[second_id].event.net_b,
                }
            )
            first = boxes[first_id]
            second = boxes[second_id]
            if overlaps(
                first if shared else inflate(first, halo),
                second if shared else inflate(second, halo),
            ):
                return False
    return True


def _terminal_lateral_correction_is_actionable(
    correction_um: float,
    segment: str,
    pdk_segments: set[str],
    grid_um: float,
) -> bool:
    minimum = (
        ACCESS_LATERAL_TOLERANCE_UM
        if segment in pdk_segments
        else 0.5 * grid_um - 1e-6
    )
    return abs(correction_um) > minimum


def _terminal_lateral_states(
    report: dict[str, Any],
    assignment: dict[str, Any],
    items: list[PlacedCrossing],
    config: dict[str, Any],
) -> list[tuple[str, tuple[tuple[float, float], str, float]]]:
    """Align a failed crossing-to-terminal leg with the terminal tangent line."""

    routing_case = report.get("routing_case")
    if not routing_case or not Path(routing_case).is_file():
        return []
    ports = absolute_ports(load_case(Path(routing_case)))
    crossing_ids = {item.event.event_id for item in items}
    by_id = {item.event.event_id: item for item in items}
    technology = config["technology"]
    grid = float(technology.get("grid_um", 2.0))
    maximum_shift = float(
        config.get("feedback", {}).get(
            "terminal_lateral_max_um",
            technology.get("minimum_access_um", 10.0),
        )
    )
    configured_radius = float(
        config["placement"].get(
            "candidate_max_radius_um",
            config["placement"].get("candidate_radius_um", 80.0),
        )
    )
    states: list[tuple[str, tuple[tuple[float, float], str, float]]] = []
    seen: set[tuple[str, tuple[float, float]]] = set()
    pdk_segments = set(report.get("pdk_angle_violation_segments", []))
    feedback_segments = sorted(
        set(report.get("abnormal_segments", [])) | pdk_segments
    )
    for segment in feedback_segments:
        endpoints = [
            str(value)
            for value in assignment.get("segment_endpoints", {}).get(segment, [])
        ]
        if len(endpoints) != 2 or any(endpoint not in ports for endpoint in endpoints):
            continue
        crossing_endpoints = [
            endpoint
            for endpoint in endpoints
            if endpoint.split(",", 1)[0] in crossing_ids
        ]
        if len(crossing_endpoints) != 1:
            continue
        crossing_endpoint = crossing_endpoints[0]
        terminal_endpoint = next(
            endpoint for endpoint in endpoints if endpoint != crossing_endpoint
        )
        if terminal_endpoint.split(",", 1)[0] in crossing_ids:
            continue
        crossing_port = ports[crossing_endpoint]
        terminal_port = ports[terminal_endpoint]
        angle = math.radians(float(terminal_port.orientation))
        normal = (-math.sin(angle), math.cos(angle))
        lateral_error = (
            (crossing_port.x - terminal_port.x) * normal[0]
            + (crossing_port.y - terminal_port.y) * normal[1]
        )
        correction = max(-maximum_shift, min(maximum_shift, -lateral_error))
        # Native route failures should move by at least half a routing cell;
        # smaller snapped perturbations cannot change the A* start cell.  A
        # post-GDS PDK failure is different: every resolvable (>1 nm)
        # transverse offset must be repaired exactly, even when it is only a
        # fraction of the routing grid.
        if not _terminal_lateral_correction_is_actionable(
            correction, segment, pdk_segments, grid
        ):
            continue
        event_id = crossing_endpoint.split(",", 1)[0]
        item = by_id[event_id]
        raw_center = (
            item.center_um[0] + correction * normal[0],
            item.center_um[1] + correction * normal[1],
        )
        # A fixed device port can live on the opposite half-grid parity from
        # native A* cell centres. For a PDK-angle violation, exact terminal
        # axis alignment is the repair; snapping it back recreates the S-bend.
        center = (
            raw_center
            if segment in pdk_segments
            else _snap_backend_center(raw_center[0], raw_center[1], config)
        )
        key = (event_id, center)
        if (
            key in seen
            or center == item.center_um
            or math.dist(center, item.event.ideal_center_um)
            > max(configured_radius, item.displacement_um + maximum_shift) + 1e-9
        ):
            continue
        seen.add(key)
        states.append((event_id, (center, str(segment), float(lateral_error))))
    return states


def _orthogonal_terminal_clearance_shifts(
    terminal_port: Any,
    crossing_port: Any,
    minimum_access_um: float,
    grid_um: float,
) -> list[tuple[float, float, float]]:
    """Return bounded crossing shifts for a congested 90-degree terminal arm.

    One minimum-access span lets a dense terminal arm leave its sibling port
    bank; a second span separates that escape bend from the crossing approach
    bend.  The crossing's outward ray must independently retain one access
    span.  This is a sufficient PDK construction corridor, not a case-specific
    coordinate rule.
    """

    terminal_angle = math.radians(float(terminal_port.orientation))
    crossing_angle = math.radians(float(crossing_port.orientation))
    forward = (math.cos(terminal_angle), math.sin(terminal_angle))
    outward = (math.cos(crossing_angle), math.sin(crossing_angle))
    determinant = forward[0] * outward[1] - forward[1] * outward[0]
    if abs(determinant) <= 1e-9:
        return []
    delta = (
        float(crossing_port.x) - float(terminal_port.x),
        float(crossing_port.y) - float(terminal_port.y),
    )
    first_ray = (delta[0] * outward[1] - delta[1] * outward[0]) / determinant
    second_ray = -(
        forward[0] * delta[1] - forward[1] * delta[0]
    ) / determinant
    if first_ray < -1e-6 or second_ray < -1e-6:
        return []

    shifts: list[tuple[float, float, float]] = []
    for forward_spans in (2.0, 3.0):
        forward_deficit = max(
            0.0, forward_spans * minimum_access_um - first_ray
        )
        outward_deficit = max(0.0, minimum_access_um - second_ray)
        if forward_deficit <= 0.5 * grid_um - 1e-9 and outward_deficit <= 0.5 * grid_um - 1e-9:
            continue
        forward_shift = (
            0.0
            if forward_deficit <= 1e-9
            else grid_um
            * math.ceil((forward_deficit + 0.5 * grid_um) / grid_um)
        )
        outward_shift = (
            0.0
            if outward_deficit <= 1e-9
            else grid_um
            * math.ceil((outward_deficit + 0.5 * grid_um) / grid_um)
        )
        shifts.append(
            (
                forward_shift * forward[0] - outward_shift * outward[0],
                forward_shift * forward[1] - outward_shift * outward[1],
                max(forward_deficit, outward_deficit),
            )
        )
    return list(dict.fromkeys(shifts))


def _terminal_radius_clearance_states(
    report: dict[str, Any],
    assignment: dict[str, Any],
    items: list[PlacedCrossing],
    config: dict[str, Any],
) -> list[tuple[str, tuple[tuple[float, float], str, float]]]:
    """Move a crossing by the minimum radius-derived terminal escape deficit.

    Parallel, same-heading terminal/crossing legs require a two-45-degree
    lateral transfer.  Its forward span must carry both lateral displacement
    and both radius trims.  A fixed small feedback step cannot discover the
    required move when the crossing is tens of microns too close.
    """

    routing_case = report.get("routing_case")
    if not routing_case or not Path(routing_case).is_file():
        return []
    ports = absolute_ports(load_case(Path(routing_case)))
    by_id = {item.event.event_id: item for item in items}
    crossing_ids = set(by_id)
    technology = config["technology"]
    grid = float(technology.get("grid_um", 2.0))
    bend_radius = float(technology.get("bend_radius_um", 5.0))
    minimum_access = float(technology.get("minimum_access_um", 10.0))
    tangent_45 = bend_radius * math.tan(math.pi / 8.0)
    configured_radius = float(
        config["placement"].get(
            "candidate_max_radius_um",
            config["placement"].get("candidate_radius_um", 80.0),
        )
    )
    states: list[tuple[str, tuple[tuple[float, float], str, float]]] = []
    seen: set[tuple[str, tuple[float, float]]] = set()
    for segment in sorted(set(report.get("abnormal_segments", []))):
        endpoints = [
            str(value)
            for value in assignment.get("segment_endpoints", {}).get(segment, [])
        ]
        if len(endpoints) != 2 or any(endpoint not in ports for endpoint in endpoints):
            continue
        crossing_endpoints = [
            endpoint
            for endpoint in endpoints
            if endpoint.split(",", 1)[0] in crossing_ids
        ]
        if len(crossing_endpoints) != 1:
            continue
        crossing_endpoint = crossing_endpoints[0]
        terminal_endpoint = next(
            endpoint for endpoint in endpoints if endpoint != crossing_endpoint
        )
        if terminal_endpoint.split(",", 1)[0] in crossing_ids:
            continue
        crossing_port = ports[crossing_endpoint]
        terminal_port = ports[terminal_endpoint]
        terminal_angle = math.radians(float(terminal_port.orientation))
        start_forward = (math.cos(terminal_angle), math.sin(terminal_angle))
        end_heading = (float(crossing_port.orientation) + 180.0) % 360.0
        heading_difference = abs(
            ((end_heading - float(terminal_port.orientation) + 180.0) % 360.0)
            - 180.0
        )
        if abs(heading_difference - 90.0) <= 1e-6:
            event_id = crossing_endpoint.split(",", 1)[0]
            item = by_id[event_id]
            for shift_x, shift_y, deficit in _orthogonal_terminal_clearance_shifts(
                terminal_port, crossing_port, minimum_access, grid
            ):
                center = _snap_backend_center(
                    item.center_um[0] + shift_x,
                    item.center_um[1] + shift_y,
                    config,
                )
                key = (event_id, center)
                if (
                    key in seen
                    or center == item.center_um
                    or math.dist(center, item.event.ideal_center_um)
                    > max(configured_radius, item.displacement_um + math.hypot(shift_x, shift_y))
                    + 1e-9
                ):
                    continue
                seen.add(key)
                states.append((event_id, (center, str(segment), float(deficit))))
            continue
        if heading_difference > 1e-6:
            continue
        delta = (
            float(crossing_port.x) - float(terminal_port.x),
            float(crossing_port.y) - float(terminal_port.y),
        )
        normal = (-start_forward[1], start_forward[0])
        longitudinal = delta[0] * start_forward[0] + delta[1] * start_forward[1]
        lateral = delta[0] * normal[0] + delta[1] * normal[1]
        required = abs(lateral) + 2.0 * tangent_45
        deficit = required - longitudinal
        if deficit <= 0.5 * grid - 1e-9:
            continue
        # One extra half-cell prevents an analytically tangent solution from
        # falling back below the limit after cell-centre quantization.
        snapped_shift = grid * math.ceil((deficit + 0.5 * grid) / grid)
        event_id = crossing_endpoint.split(",", 1)[0]
        item = by_id[event_id]
        center = _snap_backend_center(
            item.center_um[0] + snapped_shift * start_forward[0],
            item.center_um[1] + snapped_shift * start_forward[1],
            config,
        )
        key = (event_id, center)
        if (
            key in seen
            or center == item.center_um
            or math.dist(center, item.event.ideal_center_um)
            > max(configured_radius, item.displacement_um + snapped_shift) + 1e-9
        ):
            continue
        seen.add(key)
        states.append((event_id, (center, str(segment), float(deficit))))
    return states


def _terminal_radius_feedback_candidates(
    radius_options: list[
        tuple[str, tuple[tuple[float, float], str, float]]
    ],
    items: list[PlacedCrossing],
    fixed_boxes: list[tuple[float, float, float, float]],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
) -> list[tuple[str, str, Any]]:
    """Build a bounded, fair portfolio of radius-derived crossing moves.

    A dense bank can report several failing arms at the same crossing.  Queueing
    the raw states segment-by-segment lets one crossing consume the complete
    detailed-routing budget while equally causal crossings are never tested.
    Offer one jointly prechecked centre-only move first, then interleave the
    individual fallbacks by crossing.  Rotation and port assignment are not
    part of these states and therefore remain invariant.
    """

    by_event: dict[
        str, list[tuple[tuple[float, float], str, float]]
    ] = {}
    for event_id, state in radius_options:
        by_event.setdefault(event_id, []).append(state)
    if not by_event:
        return []

    event_order = list(by_event)
    candidates: list[tuple[str, str, Any]] = []
    primary_cluster = tuple(
        (event_id, by_event[event_id][0]) for event_id in event_order
    )
    primary_centers = tuple(
        (event_id, state[0]) for event_id, state in primary_cluster
    )
    if len(primary_cluster) >= 2 and _geometry_legal_after_centers(
        items,
        primary_centers,
        fixed_boxes,
        die,
        manifest,
    ):
        candidates.append(
            (
                "terminal_radius_cluster",
                "|".join(event_order),
                primary_cluster,
            )
        )

    maximum_options = max(len(states) for states in by_event.values())
    for option_index in range(maximum_options):
        for event_id in event_order:
            states = by_event[event_id]
            if option_index < len(states):
                candidates.append(
                    ("terminal_radius", event_id, states[option_index])
                )
    return candidates


def _octilinear_access_deficit(
    audit: dict[str, Any], bend_radius_um: float
) -> float:
    """Return the missing local length for a straight/45/90 port connection.

    This is a sufficient local-access test, not a declaration that a longer
    global detour is impossible.  It is used to rank feedback candidates so a
    crossing move cannot repair one arm by silently removing the bend room
    from another arm.
    """

    if bool(audit.get("coincident_abutment")) or bool(
        audit.get("direct_straight_feasible")
    ):
        return 0.0
    first_forward = float(audit.get("first_forward_um", 0.0))
    second_forward = float(audit.get("second_forward_um", 0.0))
    orientation_delta = float(audit.get("orientation_delta_deg", 0.0))
    if abs(orientation_delta - 180.0) <= 1e-6:
        lateral = abs(float(audit.get("direct_lateral_offset_um", 0.0)))
        tangent_45 = bend_radius_um * math.tan(math.pi / 8.0)
        required = lateral + 2.0 * tangent_45
        return max(0.0, required - min(first_forward, second_forward))
    if orientation_delta <= 90.0 + 1e-6:
        required = float(audit.get("corner_required_um", bend_radius_um))
        rays = (
            float(audit.get("ray_intersection_first_um", math.inf)),
            float(audit.get("ray_intersection_second_um", math.inf)),
        )
        return max(0.0, required - min(rays))
    # A 135-degree endpoint change needs at least two legal bends.  Preserve
    # it as a possible global detour, but rank candidates with forward room on
    # both ends ahead of candidates that immediately reverse at a port.
    return max(0.0, bend_radius_um - min(first_forward, second_forward))


def _joint_access_states(
    items: list[PlacedCrossing],
    nets: dict[str, NetGeometry],
    manifest: dict[str, Any],
    config: dict[str, Any],
    fixed_boxes: list[tuple[float, float, float, float]],
    die: tuple[float, float, float, float],
    implicated: list[str],
) -> list[
    tuple[
        str,
        tuple[
            tuple[float, float],
            float,
            tuple[str, str],
            tuple[str, str],
            tuple[int, float, float, float],
        ],
    ]
]:
    """Rank bounded center/orientation moves using all four crossing arms."""

    technology = config["technology"]
    feedback = config.get("feedback", {})
    grid = float(technology.get("grid_um", 2.0))
    bend_radius = float(technology.get("bend_radius_um", 5.0))
    minimum_access = float(technology.get("minimum_access_um", 10.0))
    direct_threshold = float(
        technology.get(
            "short_direct_access_threshold_um",
            2.0 * (minimum_access + bend_radius),
        )
    )
    search_radius = max(
        2.0 * grid,
        float(
            feedback.get(
                "joint_access_search_radius_um",
                4.0
                * (
                    bend_radius
                    + float(technology.get("waveguide_spacing_um", 1.0))
                ),
            )
        ),
    )
    step = max(grid, float(feedback.get("joint_access_step_um", 2.0 * grid)))
    limit = max(1, int(feedback.get("joint_access_candidate_limit", 12)))
    rings = max(1, int(math.ceil(search_radius / step)))
    by_id = {item.event.event_id: item for item in items}

    def metric(
        audits: dict[str, Any], names: tuple[str, str]
    ) -> tuple[int, float, float, float]:
        selected = [
            segment
            for name in names
            for segment in audits[name]["selected_segment_access"]
        ]
        hard = sum(bool(segment.get("hard_infeasible")) for segment in selected)
        deficits = [
            _octilinear_access_deficit(segment, bend_radius) for segment in selected
        ]
        return (
            int(hard),
            float(max(deficits, default=0.0)),
            float(sum(deficits)),
            float(sum(audits[name]["connection_proxy_cost"] for name in names)),
        )

    candidates: list[
        tuple[
            tuple[Any, ...],
            str,
            tuple[
                tuple[float, float],
                float,
                tuple[str, str],
                tuple[str, str],
                tuple[int, float, float, float],
            ],
        ]
    ] = []
    for event_id in implicated:
        item = by_id[event_id]
        selected_nets = (item.event.net_a, item.event.net_b)
        try:
            _directions, current_audit = solve_net_directions(
                items,
                nets,
                manifest,
                minimum_access,
                bend_radius,
                direct_threshold,
                selected_nets=set(selected_nets),
            )
        except RuntimeError:
            continue
        current_metric = metric(current_audit, selected_nets)
        offsets = [(0, 0)]
        for ring in range(1, rings + 1):
            offsets.extend(
                sorted(
                    {
                        (x, y)
                        for x in range(-ring, ring + 1)
                        for y in range(-ring, ring + 1)
                        if max(abs(x), abs(y)) == ring
                    }
                )
            )
        orientation_states = list(dict.fromkeys(_orientation_states(item, nets, manifest)))
        for dx, dy in offsets:
            center = _snap_backend_center(
                item.center_um[0] + dx * step,
                item.center_um[1] + dy * step,
                config,
            )
            for rotation, pair_a, pair_b in orientation_states:
                state_identity = (
                    center,
                    float(rotation),
                    tuple(pair_a),
                    tuple(pair_b),
                )
                if state_identity == (
                    item.center_um,
                    float(item.rotation_deg),
                    tuple(item.event.net_a_ports),
                    tuple(item.event.net_b_ports),
                ):
                    continue
                trial = copy.deepcopy(items)
                changed = next(
                    value for value in trial if value.event.event_id == event_id
                )
                changed.center_um = center
                changed.rotation_deg = float(rotation)
                changed.event.net_a_ports = tuple(pair_a)
                changed.event.net_b_ports = tuple(pair_b)
                changed.displacement_um = math.dist(
                    center, changed.event.ideal_center_um
                )
                if not _geometry_legal_after_centers(
                    trial, (), fixed_boxes, die, manifest
                ):
                    continue
                try:
                    _directions, audits = solve_net_directions(
                        trial,
                        nets,
                        manifest,
                        minimum_access,
                        bend_radius,
                        direct_threshold,
                        selected_nets=set(selected_nets),
                    )
                except RuntimeError:
                    continue
                candidate_metric = metric(audits, selected_nets)
                if candidate_metric[0] > current_metric[0]:
                    continue
                hard_weight = 4.0 * bend_radius
                current_risk = current_metric[1] + hard_weight * current_metric[0]
                candidate_risk = (
                    candidate_metric[1] + hard_weight * candidate_metric[0]
                )
                if (candidate_risk, candidate_metric[2]) >= (
                    current_risk,
                    current_metric[2],
                ):
                    continue
                state = (
                    center,
                    float(rotation),
                    tuple(pair_a),
                    tuple(pair_b),
                    candidate_metric,
                )
                rank = (
                    -(current_risk - candidate_risk),
                    -(current_metric[2] - candidate_metric[2]),
                    candidate_risk,
                    *candidate_metric,
                    round(changed.displacement_um, 12),
                    center,
                    abs(float(rotation)),
                    tuple(pair_a),
                )
                candidates.append((rank, event_id, state))
    candidates.sort(key=lambda value: value[0])
    per_event: dict[str, list[tuple[tuple[Any, ...], tuple[Any, ...]]]] = {}
    seen: set[tuple[Any, ...]] = set()
    for rank, event_id, state in candidates:
        key = (event_id, state[:4])
        if key in seen:
            continue
        seen.add(key)
        per_event.setdefault(event_id, []).append((rank, state))
    event_order = sorted(
        per_event,
        key=lambda event_id: (per_event[event_id][0][0], event_id),
    )
    result = []
    option_index = 0
    while len(result) < limit:
        added = False
        for event_id in event_order:
            values = per_event[event_id]
            if option_index >= len(values):
                continue
            result.append((event_id, values[option_index][1]))
            added = True
            if len(result) >= limit:
                break
        if not added:
            break
        option_index += 1
    return result


def _failing_fixed_points(
    event_id: str,
    report: dict[str, Any],
    assignment: dict[str, Any],
    ports: dict[str, Any],
) -> list[tuple[float, float]]:
    points = []
    for segment in sorted(
        set(report.get("abnormal_segments", []))
        | set(report.get("pdk_angle_violation_segments", []))
    ):
        endpoints = assignment.get("segment_endpoints", {}).get(segment, [])
        if not any(str(value).split(",", 1)[0] == event_id for value in endpoints):
            continue
        for endpoint in endpoints:
            name = str(endpoint)
            if name.split(",", 1)[0] == event_id or name not in ports:
                continue
            port = ports[name]
            points.append((float(port.x), float(port.y)))
    return points


def _run_logged(
    command: list[str], log: Path, env: dict[str, str] | None = None
) -> int:
    try:
        process = subprocess.run(
            command,
            cwd=log.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        log.write_text(process.stdout, encoding="utf-8")
        return int(process.returncode)
    except (OSError, subprocess.SubprocessError, Exception) as error:
        message = (
            "Failed to launch command.\n"
            + str(error)
            + "\n"
            + "".join(traceback.format_exception(type(error), error, error.__traceback__))
        )
        log.write_text(message, encoding="utf-8")
        return -1000


def _key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for field in raw.split("\t"):
            if "=" in field:
                key, value = field.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def _integer(value: str | None, fallback: int = 10**9) -> int:
    try:
        return int(value or "")
    except ValueError:
        return fallback


def _route_result_failure_evidence(path: Path) -> dict[str, Any]:
    """Read only the small failure header, not multi-megabyte routed paths."""

    if not path.is_file():
        return {}
    header_keys = {
        "schema",
        "schema_version",
        "success",
        "entries",
        "abnormal_nets",
        "start_blockers",
        "start_blocker_types",
        "search_blockers",
        "search_blocker_types",
        "search_blocker_evidence",
    }
    lines = []
    evidence_seen = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line and not line[0].isspace() and ":" in line:
                key = line.split(":", 1)[0].strip()
                if evidence_seen and key not in header_keys:
                    break
                evidence_seen = evidence_seen or key == "search_blocker_evidence"
            lines.append(line)
    header = yaml.safe_load("".join(lines)) or {}
    return {
        key: copy.deepcopy(header.get(key) or {})
        for key in (
            "start_blockers",
            "start_blocker_types",
            "search_blockers",
            "search_blocker_types",
            "search_blocker_evidence",
        )
    }


def _route_result_first_pass_order(path: Path) -> list[str]:
    """Recover the actual first-pass net order without loading routed paths."""

    if not path.is_file():
        return []
    in_flow = False
    iteration: int | None = None
    result: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line == "flow:\n":
                in_flow = True
                continue
            if not in_flow:
                continue
            if line.startswith("instances:"):
                break
            if line.startswith("  - iteration:"):
                try:
                    iteration = int(line.split(":", 1)[1].strip())
                except ValueError:
                    iteration = None
                if iteration is not None and iteration > 1:
                    break
                continue
            if iteration == 1 and line.startswith("    net:"):
                name = line.split(":", 1)[1].strip()
                if name and name not in seen:
                    seen.add(name)
                    result.append(name)
    return result


def _failure_directed_priority(
    report: dict[str, Any], all_design_nets: Iterable[str] = ()
) -> tuple[str, ...]:
    """Apply only observed victim-before-start-blocker precedence repairs."""

    order = [str(value) for value in report.get("native_first_pass_order", [])]
    seen = set(order)
    # Native first-pass traces contain only nets that reached the ordinary
    # routing loop.  Route-group handling can therefore omit valid design nets,
    # while the C++ explicit-priority contract intentionally requires a full
    # permutation.  Complete the trace deterministically before making the
    # local victim/blocker precedence repair.
    order.extend(
        sorted(str(value) for value in all_design_nets if str(value) not in seen)
    )
    abnormal = set(str(value) for value in report.get("abnormal_segments", []))
    start_blockers = (
        report.get("native_failure_evidence", {}).get("start_blockers", {}) or {}
    )
    if not order or not abnormal or not abnormal.issubset(order):
        return ()
    priority = list(order)
    changed = False
    # Move a failed victim only as far as the earliest directly observed
    # routed blocker.  This preserves the established order everywhere else
    # and avoids turning a local two-net conflict into a global reshuffle.
    for victim in order:
        if victim not in abnormal:
            continue
        blockers = [
            str(value)
            for value in start_blockers.get(victim, [])
            if str(value) in priority and str(value) != victim
        ]
        if not blockers:
            continue
        victim_index = priority.index(victim)
        blocker_index = min(priority.index(value) for value in blockers)
        if victim_index > blocker_index:
            priority.pop(victim_index)
            priority.insert(blocker_index, victim)
            changed = True
    if not changed:
        return ()
    result = tuple(priority)
    # Leave ample room under Windows' CreateProcess command-line limit.
    if len("--explicit-net-priority=" + ",".join(result)) > 24000:
        return ()
    return result


def _renderer_lateral_access_failures(log_path: Path) -> list[dict[str, Any]]:
    """Recover strict PDK access evidence when rendering stops before JSON."""

    if not log_path.is_file():
        return []
    pattern = re.compile(
        r"radius-correct render failed for (?P<net>[^/\s]+)/\d+: "
        r"PDK forbids lateral access S-bend: "
        r"longitudinal=(?P<longitudinal>[-+0-9.eE]+)um, "
        r"lateral=(?P<lateral>[-+0-9.eE]+)um"
    )
    failures: dict[str, dict[str, Any]] = {}
    for match in pattern.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        net = match.group("net")
        failures[net] = {
            "net": net,
            "longitudinal_um": float(match.group("longitudinal")),
            "lateral_um": float(match.group("lateral")),
            "reason": "pdk_forbids_lateral_access_sbend",
        }
    return [failures[name] for name in sorted(failures)]


def _attempt_backend(
    attempt_dir: Path,
    routing_case: Path,
    config: dict[str, Any],
    explicit_net_priority: tuple[str, ...] | None = None,
    max_search_expanded: int | None = None,
    mfot_mode: str | None = None,
    physical_bank_channels: bool = False,
) -> dict[str, Any]:
    paths = config["paths"]
    attempt_dir.mkdir(parents=True, exist_ok=False)
    converted = attempt_dir / "converted"
    converter_command = [
        sys.executable,
        str(Path(paths["frozen_converter"]).resolve(strict=True)),
        str(routing_case),
        str(converted),
    ]
    converter_started = time.perf_counter()
    converter_code = _run_logged(converter_command, attempt_dir / "converter.log")
    converter_elapsed = time.perf_counter() - converter_started
    report: dict[str, Any] = {
        "converter_command": converter_command,
        "converter_exit_code": converter_code,
        "converter_elapsed_s": converter_elapsed,
    }
    if converter_code != 0:
        report.update(
            status=(
                "CONVERTER_LAUNCH_FAIL"
                if converter_code < 0
                else "CONVERTER_FAIL"
            ),
            accepted=False,
        )
        return report

    # Route against the same realized PCell bbox and optical-port geometry
    # that the immutable GDS renderer instantiates.  The former preserved
    # LiDAR-library abstracts differed from real ring ports by up to microns,
    # which forced the renderer to hide the mismatch with endpoint S-bends.
    conversion_manifest_path = converted / "conversion_manifest.yml"
    conversion_manifest = yaml.safe_load(
        conversion_manifest_path.read_text(encoding="utf-8")
    ) or {}
    macro_sources = conversion_manifest.get("macro_sources", {}) or {}
    unrealized_macros = sorted(
        str(name)
        for name, source in macro_sources.items()
        if str(source) != "gdsfactory"
        and not str(name).startswith("picroute_crossing_")
    )
    realized_conversion = bool(
        conversion_manifest.get("realize_gdsfactory_lef", False)
    )
    report.update(
        conversion_manifest=str(conversion_manifest_path.resolve()),
        realized_pcell_geometry=realized_conversion,
        unrealized_non_crossing_macros=unrealized_macros,
    )
    if not realized_conversion or unrealized_macros:
        report.update(status="CONVERTER_GEOMETRY_FAIL", accepted=False)
        return report

    native = attempt_dir / "native"
    search_limit = int(
        max_search_expanded
        if max_search_expanded is not None
        else config.get("feedback", {}).get(
            "initial_max_search_expanded", 250000
        )
    )
    configured_mfot_mode = (
        mfot_mode
        if mfot_mode is not None
        else config.get("feedback", {}).get("mfot_mode", "off")
    )
    # YAML 1.1 treats the unquoted word ``off`` as boolean false.  Preserve
    # compatibility with existing configs while emitting the native CLI's
    # canonical string value.
    selected_mfot_mode = (
        "off" if configured_mfot_mode is False else str(configured_mfot_mode)
    )
    allowed_mfot_modes = {
        "off",
        "uniform",
        "static",
        "geometry",
        "rudy",
        "allnet2d",
        "directional",
        "no-access",
        "full",
    }
    if selected_mfot_mode not in allowed_mfot_modes:
        raise ValueError(f"Unsupported MFOT mode: {selected_mfot_mode}")
    router_command = [
        str(Path(paths["frozen_router"]).resolve(strict=True)),
        str(converted / "converted_lef.yml"),
        str(converted / "converted_def.yml"),
        str(native),
        "--skip-render",
        "--deterministic-order",
        "--max-iteration=20",
        f"--mfot-mode={selected_mfot_mode}",
        "--strict-preplaced-crossings",
        f"--max-search-expanded={search_limit}",
    ]
    if explicit_net_priority:
        router_command.append(
            "--explicit-net-priority=" + ",".join(explicit_net_priority)
        )
    router_env = os.environ.copy()
    if physical_bank_channels:
        router_env["PICDB_LIDAR_ENABLE_PHYSICAL_BANK_CHANNELS"] = "1"
        router_env.pop("PICDB_LIDAR_DISABLE_PHYSICAL_BANK_CHANNELS", None)
    else:
        router_env["PICDB_LIDAR_DISABLE_PHYSICAL_BANK_CHANNELS"] = "1"
        router_env.pop("PICDB_LIDAR_ENABLE_PHYSICAL_BANK_CHANNELS", None)
    router_started = time.perf_counter()
    router_code = _run_logged(
        router_command, attempt_dir / "router.log", env=router_env
    )
    router_elapsed = time.perf_counter() - router_started
    flow = _key_values(native / "lidar_grid_route_flow_summary.txt")
    drc = _key_values(native / "db_drc_summary.txt")
    native_timing = _key_values(attempt_dir / "router.log")
    native_evidence: dict[str, Any] = {}
    route_result_path = native / "lidar_route_result.yml"
    if route_result_path.is_file():
        native_evidence = _route_result_failure_evidence(route_result_path)
    native_first_pass_order = _route_result_first_pass_order(route_result_path)
    abnormal = sorted(value for value in flow.get("abnormal", "").split(",") if value)
    report.update(
        router_command=router_command,
        router_exit_code=router_code,
        router_elapsed_s=router_elapsed,
        native_runtime_init_s=native_timing.get("timing_cpp_runtime_init_s"),
        native_route_core_s=native_timing.get("timing_cpp_route_core_s"),
        strict_preplaced_crossings=True,
        flow=flow,
        db_drc=drc,
        abnormal_segments=abnormal,
        missing_routes=_integer(drc.get("missing_route")),
        db_drc_violations=_integer(drc.get("violations")),
        native_failure_evidence=native_evidence,
        native_first_pass_order=native_first_pass_order,
        explicit_net_priority_count=len(explicit_net_priority or ()),
        max_search_expanded=search_limit,
        mfot_mode=selected_mfot_mode,
        physical_bank_channels=physical_bank_channels,
    )
    route_clean = (
        router_code == 0 and flow.get("success") == "1" and drc.get("clean") == "1"
    )
    if router_code < 0:
        report.update(status="ROUTER_LAUNCH_FAIL", accepted=False)
        return report
    if not route_clean:
        report.update(status="STRICT_ROUTE_FAIL", accepted=False)
        return report

    renderer_command = [
        sys.executable,
        str(Path(paths["frozen_renderer"]).resolve(strict=True)),
        str(Path(paths["lidar_python_source"]).resolve(strict=True)),
        str(native / "lidar_route_result.yml"),
        str(attempt_dir / "final.gds"),
        "--strict-geometry",
        "--base-lidar-yml",
        str(routing_case),
        "--invalid-access-report",
        str(attempt_dir / "invalid_access.txt"),
        "--geometry-recovery-report",
        str(attempt_dir / "geometry_recovery.json"),
    ]
    renderer_started = time.perf_counter()
    renderer_code = _run_logged(renderer_command, attempt_dir / "renderer.log")
    renderer_elapsed = time.perf_counter() - renderer_started
    renderer_metrics = _key_values(attempt_dir / "renderer.log")
    report.update(
        renderer_command=renderer_command,
        renderer_exit_code=renderer_code,
        renderer_elapsed_s=renderer_elapsed,
        waveguide_length_um=renderer_metrics.get("length"),
    )
    if renderer_code < 0:
        report.update(status="RENDER_LAUNCH_FAIL", accepted=False)
        return report
    if renderer_code != 0:
        renderer_lateral_failures = _renderer_lateral_access_failures(
            attempt_dir / "renderer.log"
        )
        if renderer_lateral_failures:
            report.update(
                status="PDK_ANGLE_FAIL",
                accepted=False,
                pdk_angle_clean=False,
                pdk_angle_violation_counts={
                    "access_lateral_offsets": len(renderer_lateral_failures),
                    "prohibited_access_sbends": len(renderer_lateral_failures),
                },
                pdk_angle_violation_segments=[
                    item["net"] for item in renderer_lateral_failures
                ],
                renderer_lateral_access_failures=renderer_lateral_failures,
            )
        else:
            report.update(status="RENDER_FAIL", accepted=False)
        return report

    pdk_angle_command = [
        sys.executable,
        str(Path(paths["pdk_angle_audit"]).resolve(strict=True)),
        "--route-result",
        str(native / "lidar_route_result.yml"),
        "--geometry-report",
        str(attempt_dir / "geometry_recovery.json"),
        "--gds",
        str(attempt_dir / "final.gds"),
        "--out",
        str(attempt_dir / "pdk_angle_audit.json"),
    ]
    pdk_angle_started = time.perf_counter()
    pdk_angle_code = _run_logged(
        pdk_angle_command, attempt_dir / "pdk_angle_audit.log"
    )
    pdk_angle_elapsed = time.perf_counter() - pdk_angle_started
    pdk_angle_report = {}
    pdk_angle_path = attempt_dir / "pdk_angle_audit.json"
    if pdk_angle_path.exists():
        pdk_angle_report = json.loads(pdk_angle_path.read_text(encoding="utf-8"))
    pdk_angle_clean = pdk_angle_code == 0 and bool(
        pdk_angle_report.get("clean", False)
    )
    report.update(
        pdk_angle_command=pdk_angle_command,
        pdk_angle_exit_code=pdk_angle_code,
        pdk_angle_elapsed_s=pdk_angle_elapsed,
        pdk_angle_clean=pdk_angle_clean,
        pdk_angle_violation_counts={
            "short_sbends": int(
                pdk_angle_report.get("route_result_audit", {}).get(
                    "short_sbend_count", 0
                )
            ),
            "prohibited_short_sbends": int(
                pdk_angle_report.get("route_result_audit", {}).get(
                    "prohibited_short_sbend_count", 0
                )
            ),
            "access_lateral_offsets": int(
                pdk_angle_report.get("route_result_audit", {}).get(
                    "access_lateral_offset_count", 0
                )
            ),
            "off_grid_segments": int(
                pdk_angle_report.get("route_result_audit", {}).get(
                    "off_grid_segment_count", 0
                )
            ),
            "unsupported_turns": int(
                pdk_angle_report.get("route_result_audit", {}).get(
                    "unsupported_turn_count", 0
                )
            ),
            "unsupported_recoveries": int(
                pdk_angle_report.get("renderer_audit", {}).get(
                    "unsupported_recovery_count", 0
                )
            ),
            "prohibited_access_sbends": int(
                pdk_angle_report.get("renderer_audit", {}).get(
                    "prohibited_access_sbend_count", 0
                )
            ),
        },
        pdk_angle_violation_segments=sorted(
            {
                str(item.get("net"))
                for item in pdk_angle_report.get("route_result_audit", {}).get(
                    "access_lateral_offsets", []
                )
                if item.get("net")
            }
            | {
                str(item.get("net"))
                for item in pdk_angle_report.get("route_result_audit", {}).get(
                    "prohibited_short_sbends", []
                )
                if item.get("net")
            }
            | {
                str(item.get("net"))
                for item in pdk_angle_report.get("renderer_audit", {}).get(
                    "prohibited_access_sbends", []
                )
                if item.get("net")
            }
        ),
    )
    if pdk_angle_code < 0:
        report.update(status="PDK_ANGLE_LAUNCH_FAIL", accepted=False)
        return report

    continuity_command = [
        sys.executable,
        str(Path(paths["frozen_continuity_audit"]).resolve(strict=True)),
        "--case",
        str(routing_case),
        "--route-result",
        str(native / "lidar_route_result.yml"),
        "--gds",
        str(attempt_dir / "final.gds"),
        "--out",
        str(attempt_dir / "continuity.json"),
    ]
    continuity_started = time.perf_counter()
    continuity_code = _run_logged(continuity_command, attempt_dir / "continuity.log")
    continuity_elapsed = time.perf_counter() - continuity_started
    continuity = {}
    continuity_path = attempt_dir / "continuity.json"
    if continuity_path.exists():
        continuity = json.loads(continuity_path.read_text(encoding="utf-8"))
    if continuity_code < 0:
        report.update(
            continuity_command=continuity_command,
            continuity_exit_code=continuity_code,
            continuity_elapsed_s=continuity_elapsed,
            continuity_clean=False,
            status="CONTINUITY_LAUNCH_FAIL",
            accepted=False,
        )
        return report
    continuity_clean = continuity_code == 0 and not any(
        int(continuity.get(section, {}).get(key, 0))
        for section, key in (
            ("gds_audit", "disconnected_route_cell_count"),
            ("gds_audit", "empty_route_cell_count"),
            ("route_result_audit", "endpoint_tangent_violation_count"),
            ("route_result_audit", "zero_length_segment_count"),
        )
    )
    accepted = bool(pdk_angle_clean and continuity_clean)
    report.update(
        continuity_command=continuity_command,
        continuity_exit_code=continuity_code,
        continuity_elapsed_s=continuity_elapsed,
        continuity_clean=continuity_clean,
        status=(
            "ACCEPTED"
            if accepted
        else "PDK_ANGLE_FAIL"
            if not pdk_angle_clean
            else "CONTINUITY_FAIL"
        ),
        accepted=accepted,
    )
    gds_path = attempt_dir / "final.gds"
    if gds_path.exists():
        report.update(
            gds_bytes=gds_path.stat().st_size,
            gds_sha256=sha256_file(gds_path),
        )
    report["attempt_elapsed_s"] = (
        converter_elapsed
        + router_elapsed
        + renderer_elapsed
        + pdk_angle_elapsed
        + continuity_elapsed
    )
    return report


def _pdk_hard_violation_count(report: dict[str, Any]) -> int:
    """Return the number of geometry defects that make a GDS unacceptable.

    ``short_sbends`` is deliberately excluded: the renderer also records
    collinear short connections under that implementation name, and those
    connections produce a straight centreline.  Every other counter below is
    a hard PDK-angle failure.
    """

    counts = report.get("pdk_angle_violation_counts", {}) or {}
    return sum(
        int(counts.get(name, 0) or 0)
        for name in (
            "access_lateral_offsets",
            "off_grid_segments",
            "prohibited_short_sbends",
            "prohibited_access_sbends",
            "unsupported_recoveries",
            "unsupported_turns",
        )
    )


def _score(report: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        0 if report.get("accepted") else 1,
        int(report.get("missing_routes", 10**9)),
        len(report.get("abnormal_segments", [])),
        int(report.get("db_drc_violations", 10**9)),
        _pdk_hard_violation_count(report),
    )


def _implicated_crossings(
    report: dict[str, Any], assignment: dict[str, Any]
) -> list[str]:
    crossings = {str(item["id"]) for item in assignment["crossings"]}
    endpoints = assignment["segment_endpoints"]
    counts: dict[str, int] = {}
    feedback_segments = sorted(
        set(report.get("abnormal_segments", []))
        | set(report.get("pdk_angle_violation_segments", []))
    )
    for segment in feedback_segments:
        for endpoint in endpoints.get(segment, []):
            identifier = str(endpoint).split(",", 1)[0]
            if identifier in crossings:
                counts[identifier] = counts.get(identifier, 0) + 1
        blockers = (
            report.get("native_failure_evidence", {})
            .get("start_blockers", {})
            .get(segment, [])
        )
        for blocker in blockers:
            for endpoint in endpoints.get(str(blocker), []):
                identifier = str(endpoint).split(",", 1)[0]
                if identifier in crossings:
                    counts[identifier] = counts.get(identifier, 0) + 1
    return sorted(counts, key=lambda identifier: (-counts[identifier], identifier))


def _failure_crossing_clusters(
    report: dict[str, Any], assignment: dict[str, Any]
) -> list[tuple[str, ...]]:
    crossing_ids = {str(item["id"]) for item in assignment["crossings"]}
    endpoints = assignment["segment_endpoints"]

    def owners(segment: str) -> set[str]:
        return {
            str(endpoint).split(",", 1)[0]
            for endpoint in endpoints.get(segment, [])
            if str(endpoint).split(",", 1)[0] in crossing_ids
        }

    start_blockers = (
        report.get("native_failure_evidence", {}).get("start_blockers", {}) or {}
    )
    clusters = set()
    for segment in report.get("abnormal_segments", []):
        members = owners(str(segment))
        for blocker in start_blockers.get(segment, []):
            members.update(owners(str(blocker)))
        if members:
            clusters.add(tuple(sorted(members)))
    return sorted(clusters, key=lambda value: (-len(value), value))


def _merged_failure_components(
    report: dict[str, Any], assignment: dict[str, Any]
) -> list[tuple[str, ...]]:
    """Merge overlapping victim/blocker pairs into independent components."""

    pending = [
        set(cluster)
        for cluster in _failure_crossing_clusters(report, assignment)
        if len(cluster) >= 2
    ]
    merged: list[set[str]] = []
    while pending:
        component = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remaining = []
            for other in pending:
                if component & other:
                    component.update(other)
                    changed = True
                else:
                    remaining.append(other)
            pending = remaining
        merged.append(component)
    return sorted(
        (tuple(sorted(component)) for component in merged),
        key=lambda value: (-len(value), value),
    )


def run_strict_backend(
    benchmark: BenchmarkRef,
    normalized_path: Path,
    prediction: Prediction,
    placed: list[PlacedCrossing],
    nets: dict[str, NetGeometry],
    manifest: dict[str, Any],
    config: dict[str, Any],
    case_dir: Path,
    results_case_dir: Path,
) -> dict[str, Any]:
    """Run frozen routing and bounded orientation-state no-good feedback."""

    backend_started = time.perf_counter()
    backend_root = case_dir / "09_backend"
    backend_root.mkdir(parents=True, exist_ok=False)
    maximum = max(1, int(config.get("feedback", {}).get("maximum_attempts", 10)))
    current = copy.deepcopy(placed)
    best = copy.deepcopy(current)
    best_report: dict[str, Any] | None = None
    best_assignment: dict[str, Any] | None = None
    best_attempt = -1
    current_priority: tuple[str, ...] | None = None
    current_physical_bank_channels = False
    best_physical_bank_channels = False
    attempted_states: set[tuple[Any, ...]] = set()
    attempt_reports = []
    candidate_queue: list[tuple[str, str, Any]] = []
    original_ports = absolute_ports(load_case(normalized_path))
    plateau_accepts = 0
    accepted_causal_signatures: set[tuple[Any, ...]] = set()
    maximum_plateau_accepts = int(
        config.get("feedback", {}).get("maximum_plateau_accepts_at_score", 4)
    )
    normalized_case = load_case(normalized_path)
    feedback_fixed_boxes = [item.bbox for item in fixed_obstacles(normalized_case)]
    feedback_die = die_area(normalized_case)
    minimum_access_um = float(
        config["technology"].get("minimum_access_um", 10.0)
    )
    minimum_radius_um = float(config["technology"].get("bend_radius_um", 5.0))
    direct_threshold_um = float(
        config["technology"].get(
            "short_direct_access_threshold_um",
            2.0 * (minimum_access_um + minimum_radius_um),
        )
    )

    def solve_trial_directions(
        trial_items: list[PlacedCrossing],
    ) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
        return solve_net_directions(
            trial_items,
            nets,
            manifest,
            minimum_access_um,
            minimum_radius_um,
            direct_threshold_um,
        )

    rejected_feedback_states: list[dict[str, Any]] = []

    for attempt_index in range(maximum):
        trial = copy.deepcopy(current if attempt_index else best)
        trial_priority = current_priority
        trial_physical_bank_channels = (
            current_physical_bank_channels
            if attempt_index
            else best_physical_bank_channels
        )
        changed = None
        prechecked_directions: tuple[
            dict[str, dict[str, list[str]]], dict[str, Any]
        ] | None = None
        if attempt_index:
            while candidate_queue:
                kind, event_id, state = candidate_queue.pop(0)
                # A route-order repair that was ineffective at one crossing
                # geometry can become decisive after a center/orientation
                # move.  Include the current physical state in its no-good
                # identity so the same observed precedence is allowed to be
                # re-evaluated after geometry changes, while remaining
                # deterministic at an unchanged state.
                geometry_context = (
                    tuple(
                        sorted(
                            (
                                item.event.event_id,
                                tuple(item.center_um),
                                _state(item),
                            )
                            for item in trial
                        )
                    )
                    if kind == "priority"
                    else ()
                )
                key = (kind, event_id, state, geometry_context)
                if key in attempted_states:
                    continue
                attempted_states.add(key)
                if kind in {"cluster_center", "terminal_lateral_cluster"}:
                    trial_by_id = {
                        value.event.event_id: value for value in trial
                    }
                    changes = []
                    for crossing_id, center in state:
                        item = trial_by_id[crossing_id]
                        item.center_um = (float(center[0]), float(center[1]))
                        item.displacement_um = math.dist(
                            item.center_um, item.event.ideal_center_um
                        )
                        changes.append(
                            {"crossing": crossing_id, "center_um": list(item.center_um)}
                        )
                    changed = {
                        "kind": (
                            "missing_terminal_lateral_cluster_ripple"
                            if kind == "terminal_lateral_cluster"
                            else "blocker_cluster_ripple"
                        ),
                        "crossings": changes,
                    }
                elif kind == "terminal_radius_cluster":
                    trial_by_id = {
                        value.event.event_id: value for value in trial
                    }
                    changes = []
                    for crossing_id, radius_state in state:
                        center, trigger_segment, clearance_deficit = radius_state
                        item = trial_by_id[crossing_id]
                        item.center_um = (float(center[0]), float(center[1]))
                        item.displacement_um = math.dist(
                            item.center_um, item.event.ideal_center_um
                        )
                        changes.append(
                            {
                                "crossing": crossing_id,
                                "center_um": list(item.center_um),
                                "trigger_segment": trigger_segment,
                                "clearance_deficit_um": clearance_deficit,
                            }
                        )
                    changed = {
                        "kind": "missing_terminal_radius_clearance_cluster",
                        "crossings": changes,
                    }
                elif kind == "joint_access_cluster":
                    trial_by_id = {
                        value.event.event_id: value for value in trial
                    }
                    changes = []
                    for crossing_id, joint_state in state:
                        center, rotation, pair_a, pair_b, access_metric = joint_state
                        item = trial_by_id[crossing_id]
                        item.center_um = (float(center[0]), float(center[1]))
                        item.rotation_deg = float(rotation)
                        item.event.net_a_ports = tuple(pair_a)
                        item.event.net_b_ports = tuple(pair_b)
                        item.displacement_um = math.dist(
                            item.center_um, item.event.ideal_center_um
                        )
                        changes.append(
                            {
                                "crossing": crossing_id,
                                "center_um": list(item.center_um),
                                "rotation_deg": item.rotation_deg,
                                "net_a_ports": list(item.event.net_a_ports),
                                "net_b_ports": list(item.event.net_b_ports),
                                "access_metric": list(access_metric),
                            }
                        )
                    changed = {
                        "kind": "joint_four_arm_access_cluster",
                        "crossings": changes,
                    }
                elif kind == "priority":
                    trial_priority = tuple(str(value) for value in state)
                    changed = {
                        "kind": "failure_directed_net_priority",
                        "promoted_nets": list(event_id.split("|")),
                        "net_count": len(trial_priority),
                    }
                elif kind == "physical_bank_rescue":
                    trial_physical_bank_channels = bool(state)
                    changed = {
                        "kind": "failure_directed_physical_bank_rescue",
                        "trigger": event_id,
                        "enabled": trial_physical_bank_channels,
                    }
                else:
                    item = next(
                        value for value in trial if value.event.event_id == event_id
                    )
                if kind == "center":
                    item.center_um = (float(state[0]), float(state[1]))
                    item.displacement_um = math.dist(
                        item.center_um, item.event.ideal_center_um
                    )
                    changed = {
                        "kind": "center_no_good",
                        "crossing": event_id,
                        "center_um": list(item.center_um),
                    }
                elif kind == "terminal_lateral":
                    center, trigger_segment, lateral_error = state
                    item.center_um = (float(center[0]), float(center[1]))
                    item.displacement_um = math.dist(
                        item.center_um, item.event.ideal_center_um
                    )
                    changed = {
                        "kind": "missing_terminal_lateral_ripple",
                        "crossing": event_id,
                        "center_um": list(item.center_um),
                        "trigger_segment": trigger_segment,
                        "current_lateral_error_um": lateral_error,
                    }
                elif kind == "terminal_radius":
                    center, trigger_segment, clearance_deficit = state
                    item.center_um = (float(center[0]), float(center[1]))
                    item.displacement_um = math.dist(
                        item.center_um, item.event.ideal_center_um
                    )
                    changed = {
                        "kind": "missing_terminal_radius_clearance",
                        "crossing": event_id,
                        "center_um": list(item.center_um),
                        "trigger_segment": trigger_segment,
                        "clearance_deficit_um": clearance_deficit,
                    }
                elif kind == "joint_access":
                    center, rotation, pair_a, pair_b, access_metric = state
                    item.center_um = (float(center[0]), float(center[1]))
                    item.rotation_deg = float(rotation)
                    item.event.net_a_ports = tuple(pair_a)
                    item.event.net_b_ports = tuple(pair_b)
                    item.displacement_um = math.dist(
                        item.center_um, item.event.ideal_center_um
                    )
                    changed = {
                        "kind": "joint_four_arm_access_search",
                        "crossing": event_id,
                        "center_um": list(item.center_um),
                        "rotation_deg": item.rotation_deg,
                        "net_a_ports": list(item.event.net_a_ports),
                        "net_b_ports": list(item.event.net_b_ports),
                        "access_metric": list(access_metric),
                    }
                elif kind == "orientation":
                    item.rotation_deg, item.event.net_a_ports, item.event.net_b_ports = state
                    changed = {
                        "kind": "orientation_no_good",
                        "crossing": event_id,
                        "state": [state[0], list(state[1]), list(state[2])],
                    }
                try:
                    prechecked_directions = solve_trial_directions(trial)
                except RuntimeError as error:
                    # A feedback state that breaks the four-port direction
                    # chain can never reach the detailed router.  Reject it
                    # inside the candidate-selection loop instead of burning
                    # one of the bounded physical routing attempts.
                    rejected_feedback_states.append(
                        {
                            "attempt": attempt_index,
                            "changed_state": changed,
                            "reason": str(error),
                        }
                    )
                    trial = copy.deepcopy(current)
                    trial_priority = current_priority
                    trial_physical_bank_channels = current_physical_bank_channels
                    changed = None
                    prechecked_directions = None
                    continue
                break
            if changed is None:
                break

        attempt_dir = backend_root / f"attempt_{attempt_index:02d}"
        frontend_dir = attempt_dir / "frontend"
        try:
            if prechecked_directions is None:
                directions, direction_audit = solve_trial_directions(trial)
            else:
                directions, direction_audit = prechecked_directions
        except RuntimeError as error:
            report = {
                "attempt": attempt_index,
                "changed_state": changed,
                "status": "FRONTEND_DIRECTION_REJECT",
                "accepted": False,
                "missing_routes": 10**9,
                "db_drc_violations": 10**9,
                "abnormal_segments": [],
                "frontend_audit_passed": False,
                "pre_route_access_gate_passed": False,
                "frontend_error": str(error),
                "routing_case": None,
                "orientation_states": {
                    item.event.event_id: _state(item) for item in trial
                },
            }
            write_json(attempt_dir / "attempt_report.json", report)
            attempt_reports.append(report)
            continue
        routing_case, assignment = materialize_case(
            normalized_path,
            trial,
            prediction,
            directions,
            manifest,
            frontend_dir,
            float(config["technology"].get("minimum_access_um", 10.0)),
            float(config["technology"].get("bend_radius_um", 5.0)),
            float(
                config["technology"].get(
                    "short_direct_access_threshold_um",
                    2.0
                    * (
                        float(config["technology"].get("minimum_access_um", 10.0))
                        + float(config["technology"].get("bend_radius_um", 5.0))
                    ),
                )
            ),
        )
        write_json(frontend_dir / "direction_audit.json", direction_audit)
        frontend_audit = audit_frontend(
            benchmark.path,
            benchmark.sha256,
            normalized_path,
            routing_case,
            prediction,
            trial,
            manifest,
            die_area(load_case(normalized_path)),
        )
        write_json(frontend_dir / "frontend_audit.json", frontend_audit)
        access_gate_passed = not assignment["hard_invalid_segments"]
        if frontend_audit["passed"] and access_gate_passed:
            try:
                report = _attempt_backend(
                    attempt_dir / "frozen_backend",
                    routing_case,
                    config,
                    explicit_net_priority=trial_priority,
                    physical_bank_channels=trial_physical_bank_channels,
                )
            except Exception as error:
                report = {
                    "status": "BACKEND_EXCEPTION",
                    "accepted": False,
                    "missing_routes": 10**9,
                    "db_drc_violations": 10**9,
                    "abnormal_segments": [],
                    "frontend_audit_passed": frontend_audit["passed"],
                    "pre_route_access_gate_passed": access_gate_passed,
                    "backend_exception": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    ),
                }
        else:
            # A rejected frontend is not a routed design and must never win
            # the backend score merely because no router DRC was produced.
            frontend_failure_count = 10**9
            report = {
                "status": (
                    "FRONTEND_ACCESS_REJECT"
                    if not access_gate_passed
                    else "FRONTEND_REJECT"
                ),
                "accepted": False,
                "missing_routes": frontend_failure_count,
                "db_drc_violations": frontend_failure_count,
                "abnormal_segments": assignment["hard_invalid_segments"],
            }
        report.update(
            attempt=attempt_index,
            changed_state=changed,
            frontend_audit_passed=frontend_audit["passed"],
            pre_route_access_gate_passed=access_gate_passed,
            routing_case=str(routing_case.resolve()),
            orientation_states={item.event.event_id: _state(item) for item in trial},
        )
        attempt_reports.append(report)

        improved = best_report is None or _score(report) < _score(best_report)
        same_score = best_report is not None and _score(report) == _score(best_report)
        route_causal_plateau = bool(
            same_score
            and changed is not None
            and report.get("status") == "STRICT_ROUTE_FAIL"
            and report.get("pre_route_access_gate_passed")
            and set(report.get("abnormal_segments", []))
            != set(best_report.get("abnormal_segments", []))
        )
        pdk_causal_plateau = bool(
            same_score
            and changed is not None
            and report.get("status") == "PDK_ANGLE_FAIL"
            and set(report.get("pdk_angle_violation_segments", []))
            != set(best_report.get("pdk_angle_violation_segments", []))
        )
        causal_signature = (
            str(report.get("status")),
            _score(report),
            tuple(sorted(str(value) for value in report.get("abnormal_segments", []))),
            tuple(
                sorted(
                    str(value)
                    for value in report.get("pdk_angle_violation_segments", [])
                )
            ),
        )
        causal_plateau = bool(
            (route_causal_plateau or pdk_causal_plateau)
            and plateau_accepts < maximum_plateau_accepts
            and causal_signature not in accepted_causal_signatures
        )
        if improved or causal_plateau:
            best_report = report
            best = copy.deepcopy(trial)
            best_assignment = assignment
            best_attempt = attempt_index
            current = copy.deepcopy(trial)
            current_priority = trial_priority
            current_physical_bank_channels = trial_physical_bank_channels
            best_physical_bank_channels = trial_physical_bank_channels
            accepted_causal_signatures.add(causal_signature)
            if improved:
                plateau_accepts = 0
                report["feedback_decision"] = (
                    "baseline" if changed is None else "accepted_improvement"
                )
            else:
                plateau_accepts += 1
                report["feedback_decision"] = "accepted_causal_plateau"
            if attempt_index:
                candidate_queue.clear()
        else:
            report["feedback_decision"] = "rejected_no_improvement"
        report["plateau_accepts_at_score"] = plateau_accepts
        report["maximum_plateau_accepts_at_score"] = maximum_plateau_accepts
        write_json(attempt_dir / "attempt_report.json", report)
        if report.get("accepted"):
            break
        if attempt_index + 1 >= maximum:
            # There is no consumer for another candidate queue after the last
            # allowed attempt.  Dense failure analysis can take minutes, so do
            # not solve directions and enumerate geometry states that the loop
            # can never execute.
            break

        # Generate the next causal action only from the accepted working
        # state.  A rejected trial must not poison later candidate geometry or
        # blocker evidence.
        feedback_report = best_report
        feedback_assignment = best_assignment
        assert feedback_report is not None and feedback_assignment is not None
        implicated = _implicated_crossings(feedback_report, feedback_assignment)
        if not implicated:
            implicated = sorted(item.event.event_id for item in current)
        by_id = {item.event.event_id: item for item in current}
        cluster_options: list[tuple[str, Any]] = []
        for cluster in _merged_failure_components(
            feedback_report, feedback_assignment
        ):
            for state in _cluster_center_states(
                [by_id[event_id] for event_id in cluster], config
            ):
                if _geometry_legal_after_centers(
                    list(by_id.values()),
                    state,
                    feedback_fixed_boxes,
                    feedback_die,
                    manifest,
                ):
                    cluster_options.append(("|".join(cluster), state))
        first_cluster_states: list[tuple[str, Any]] = []
        seen_cluster_ids: set[str] = set()
        for event_id, state in cluster_options:
            identifiers = set(event_id.split("|"))
            if identifiers & seen_cluster_ids:
                continue
            seen_cluster_ids.update(identifiers)
            first_cluster_states.append((event_id, state))
        joint_cluster_state = tuple(
            sorted(
                item
                for _event_id, state in first_cluster_states
                for item in state
            )
        )
        center_options: dict[str, list[tuple[float, float]]] = {}
        for event_id in implicated:
            item = by_id[event_id]
            failing_points = _failing_fixed_points(
                event_id,
                feedback_report,
                feedback_assignment,
                original_ports,
            )
            center_options[event_id] = _center_states(
                item, nets, config, failing_points=failing_points
            )
        maximum_center_options = max(
            (len(values) for values in center_options.values()), default=0
        )
        orientation_options = {
            event_id: [
                state
                for state in _orientation_states(by_id[event_id], nets, manifest)
                if state != _state(by_id[event_id])
            ]
            for event_id in implicated
        }
        terminal_options = _terminal_lateral_states(
            feedback_report, feedback_assignment, list(by_id.values()), config
        )
        terminal_options = [
            (event_id, state)
            for event_id, state in terminal_options
            if _geometry_legal_after_centers(
                list(by_id.values()),
                ((event_id, state[0]),),
                feedback_fixed_boxes,
                feedback_die,
                manifest,
            )
        ]
        radius_options = _terminal_radius_clearance_states(
            feedback_report, feedback_assignment, list(by_id.values()), config
        )
        radius_options = [
            (event_id, state)
            for event_id, state in radius_options
            if _geometry_legal_after_centers(
                list(by_id.values()),
                ((event_id, state[0]),),
                feedback_fixed_boxes,
                feedback_die,
                manifest,
            )
        ]
        radius_candidates = _terminal_radius_feedback_candidates(
            radius_options,
            list(by_id.values()),
            feedback_fixed_boxes,
            feedback_die,
            manifest,
        )
        joint_access_options = _joint_access_states(
            list(by_id.values()),
            nets,
            manifest,
            config,
            feedback_fixed_boxes,
            feedback_die,
            implicated,
        )
        # Each joint-access option is solved against the same accepted
        # physical state.  Take the best option for every distinct crossing
        # and offer their deterministic combination first.  Direction-chain
        # prechecking below rejects coupled combinations, while the original
        # one-crossing options remain queued as a safe fallback.  This avoids
        # spending one expensive detailed-routing attempt per independent
        # crossing in a dense failure component.
        joint_access_by_event: dict[str, Any] = {}
        for event_id, state in joint_access_options:
            joint_access_by_event.setdefault(event_id, state)
        joint_access_cluster = tuple(
            (event_id, joint_access_by_event[event_id])
            for event_id in sorted(joint_access_by_event)
        )
        joint_access_cluster_centers = tuple(
            (event_id, state[0]) for event_id, state in joint_access_cluster
        )
        joint_access_candidates: list[tuple[str, str, Any]] = []
        if len(joint_access_cluster) >= 2 and _geometry_legal_after_centers(
            list(by_id.values()),
            joint_access_cluster_centers,
            feedback_fixed_boxes,
            feedback_die,
            manifest,
        ):
            joint_access_candidates.append(
                (
                    "joint_access_cluster",
                    "|".join(event_id for event_id, _state in joint_access_cluster),
                    joint_access_cluster,
                )
            )
        joint_access_candidates.extend(
            ("joint_access", event_id, state)
            for event_id, state in joint_access_options
        )
        start_blockers = (
            feedback_report.get("native_failure_evidence", {}).get(
                "start_blockers", {}
            )
            or {}
        )
        direct_terminal_options = [
            (event_id, state)
            for event_id, state in terminal_options
            if state[1] in start_blockers
            and any(str(value) != state[1] for value in start_blockers[state[1]])
        ]
        direct_terminal_keys = {
            (event_id, state[1]) for event_id, state in direct_terminal_options
        }
        remaining_terminal_options = [
            (event_id, state)
            for event_id, state in terminal_options
            if (event_id, state[1]) not in direct_terminal_keys
        ]
        terminal_by_event: dict[str, tuple[tuple[float, float], str, float]] = {}
        for event_id, state in remaining_terminal_options:
            terminal_by_event.setdefault(event_id, state)
        terminal_cluster = tuple(
            sorted(
                (event_id, state[0])
                for event_id, state in terminal_by_event.items()
            )
        )
        pdk_angle_failure = bool(
            feedback_report.get("pdk_angle_violation_segments")
        )
        lone_missing_route = int(
            feedback_report.get("missing_routes", 10**9)
        ) == 1

        # With one missing terminal arm, the radius-derived longitudinal
        # deficit is direct causal evidence: it computes the minimum forward
        # crossing move needed for a legal straight/45/90 connection.  Try it
        # before another priority permutation or broad joint-access search so
        # a bounded attempt budget cannot be consumed by proxy candidates.
        if lone_missing_route:
            candidate_queue.extend(
                candidate
                for candidate in radius_candidates
                if candidate[0] == "terminal_radius"
            )

        # The physical-bank channelizer is intentionally a bounded rescue
        # portfolio member.  Running it unconditionally can reserve most of a
        # large bank before a few unrecognised siblings are routed.  Invoke it
        # once a meaningful multi-net residue remains; MMI16 demonstrates that
        # a four-net bank can still defeat the generic small-cluster search.
        # The resulting attempt must improve the normal immutable score or it
        # is rejected wholesale, so unrelated cases cannot retain a bad bank
        # reservation.
        physical_bank_threshold = max(
            2,
            int(
                config.get("feedback", {}).get(
                    "physical_bank_rescue_missing_threshold", 4
                )
            ),
        )
        physical_bank_candidate: tuple[str, str, Any] | None = None
        if (
            feedback_report.get("status") == "STRICT_ROUTE_FAIL"
            and int(feedback_report.get("missing_routes", 10**9))
            >= physical_bank_threshold
            and not current_physical_bank_channels
        ):
            physical_bank_candidate = (
                "physical_bank_rescue",
                f"missing_ge_{physical_bank_threshold}",
                True,
            )

        priority_state = _failure_directed_priority(
            feedback_report,
            feedback_assignment.get("segment_endpoints", {}).keys(),
        )
        deferred_priority_after_joint: tuple[str, str, Any] | None = None
        if (
            bool(config.get("feedback", {}).get("failure_directed_priority", True))
            and priority_state
            and priority_state != current_priority
        ):
            promoted = [
                name
                for name in feedback_report.get("abnormal_segments", [])
                if priority_state.index(name)
                < feedback_report.get("native_first_pass_order", []).index(name)
            ]
            priority_candidate = (
                "priority",
                "|".join(promoted),
                priority_state,
            )
            # Native start-blocker evidence is an exact causal observation,
            # whereas center/orientation candidates are geometric proxies.
            # Try the local precedence repair first so a small attempt budget
            # is not exhausted by speculative crossing moves.  After an
            # accepted priority action, however, force one geometry action
            # before another ordering action; this prevents alternating
            # victim/blocker permutations from starving the crossing search.
            previous_kind = str(
                (feedback_report.get("changed_state") or {}).get("kind", "")
            )
            if previous_kind == "failure_directed_net_priority":
                deferred_priority_after_joint = priority_candidate
            else:
                candidate_queue.append(priority_candidate)
        # Once exact blocker precedence has had one chance, crossing movement
        # is the next strongest causal action.  Radius candidates preserve the
        # crossing rotation and port mapping and compute only the missing PDK
        # access distance.  Schedule their jointly prechecked move and fair
        # per-crossing fallbacks before broad cluster/rotation searches so a
        # bounded attempt count cannot starve dense-bank terminal repairs.
        if not lone_missing_route:
            candidate_queue.extend(radius_candidates)
        # Failure-component centre ripples are the smallest physical action
        # that can open a pinched crossing corridor.  Schedule one before the
        # broader four-arm access portfolio: accepted dense-case traces show
        # that a causal ripple can resolve the residual cluster directly,
        # whereas a joint access rewrite may move several independent banks
        # and create a much larger search state.  A rejected ripple leaves the
        # joint-access candidates queued as the next fallback.
        if len(first_cluster_states) >= 2 and _geometry_legal_after_centers(
            list(by_id.values()),
            joint_cluster_state,
            feedback_fixed_boxes,
            feedback_die,
            manifest,
        ):
            candidate_queue.append(
                (
                    "cluster_center",
                    "|".join(
                        event_id for event_id, _center in joint_cluster_state
                    ),
                    joint_cluster_state,
                )
            )
        elif first_cluster_states:
            event_id, state = first_cluster_states[0]
            candidate_queue.append(("cluster_center", event_id, state))

        for option_index, candidate in enumerate(joint_access_candidates):
            candidate_queue.append(candidate)
            if option_index == 0 and deferred_priority_after_joint is not None:
                candidate_queue.append(deferred_priority_after_joint)
                deferred_priority_after_joint = None
        if deferred_priority_after_joint is not None:
            candidate_queue.append(deferred_priority_after_joint)
        # Physical-bank channelization is a broad rescue action.  Keep it in
        # the bounded portfolio, but schedule it after exact blocker-order
        # repair and crossing-local geometry candidates.  This matches the
        # causal-strength ordering documented above and prevents an unrelated
        # bank reservation from consuming the attempt immediately before a
        # directly implicated crossing ripple.  If no crossing candidate is
        # available, it naturally becomes the next action.
        if physical_bank_candidate is not None:
            candidate_queue.append(physical_bank_candidate)
        if lone_missing_route:
            for event_id, state in direct_terminal_options:
                candidate_queue.append(("terminal_lateral", event_id, state))

        if not lone_missing_route:
            for event_id, state in direct_terminal_options:
                candidate_queue.append(("terminal_lateral", event_id, state))
        # A post-GDS PDK access failure has already passed native routing and
        # continuity. Its measured lateral correction is more specific than
        # any orientation or generic centre search, so schedule it first.
        if pdk_angle_failure:
            for event_id, state in remaining_terminal_options:
                candidate_queue.append(("terminal_lateral", event_id, state))
        if int(feedback_report.get("missing_routes", 10**9)) == 1:
            for event_id, state in remaining_terminal_options:
                candidate_queue.append(("terminal_lateral", event_id, state))
        elif len(terminal_cluster) >= 2 and _geometry_legal_after_centers(
            list(by_id.values()),
            terminal_cluster,
            feedback_fixed_boxes,
            feedback_die,
            manifest,
        ):
            candidate_queue.append(
                (
                    "terminal_lateral_cluster",
                    "|".join(event_id for event_id, _center in terminal_cluster),
                    terminal_cluster,
                )
            )
        if (
            int(feedback_report.get("missing_routes", 10**9)) != 1
            and not pdk_angle_failure
        ):
            for event_id, state in remaining_terminal_options:
                candidate_queue.append(("terminal_lateral", event_id, state))
        for option_index in range(maximum_center_options):
            for event_id in implicated:
                values = center_options[event_id]
                if option_index < len(values):
                    candidate_queue.append(
                        ("center", event_id, values[option_index])
                    )
        # A missing terminal route is a clearance/forward-access failure.
        # Test direction-feasible center moves before changing a crossing's
        # port assignment: rotation does not create any additional physical
        # escape length and can disturb an already routable opposite arm.
        for option_index in range(
            max((len(values) for values in orientation_options.values()), default=0)
        ):
            for event_id in implicated:
                values = orientation_options[event_id]
                if option_index < len(values):
                    candidate_queue.append(
                        ("orientation", event_id, values[option_index])
                    )
        queued_cluster_states = {
            state for _event_id, state in first_cluster_states
        }
        for event_id, state in cluster_options:
            if state not in queued_cluster_states:
                candidate_queue.append(("cluster_center", event_id, state))

    assert best_report is not None and best_assignment is not None
    summary = {
        "schema_version": 1,
        "status": best_report["status"],
        "accepted": bool(best_report.get("accepted")),
        "best_attempt": best_attempt,
        "attempt_count": len(attempt_reports),
        "frontend_rejected_feedback_state_count": len(rejected_feedback_states),
        "frontend_rejected_feedback_states": rejected_feedback_states,
        "attempts": [
            {
                "attempt": value["attempt"],
                "status": value["status"],
                "accepted": value["accepted"],
                "changed_state": value["changed_state"],
                "missing_routes": value.get("missing_routes"),
                "db_drc_violations": value.get("db_drc_violations"),
                "abnormal_segments": value.get("abnormal_segments", []),
                "physical_bank_channels": value.get(
                    "physical_bank_channels", False
                ),
                "feedback_decision": value.get("feedback_decision"),
            }
            for value in attempt_reports
        ],
        "frozen_tool_hashes": {
            key: sha256_file(Path(config["paths"][key]))
            for key in (
                "frozen_router",
                "frozen_converter",
                "frozen_renderer",
                "frozen_continuity_audit",
                "pdk_angle_audit",
            )
        },
        "backend_wall_time_s": time.perf_counter() - backend_started,
    }
    if bool(best_report.get("accepted")):
        summary["accepted_metrics"] = {
            key: best_report.get(key)
            for key in (
                "waveguide_length_um",
                "db_drc_violations",
                "missing_routes",
                "converter_elapsed_s",
                "router_elapsed_s",
                "native_runtime_init_s",
                "native_route_core_s",
                "renderer_elapsed_s",
                "pdk_angle_elapsed_s",
                "pdk_angle_clean",
                "pdk_angle_violation_counts",
                "continuity_elapsed_s",
                "attempt_elapsed_s",
                "gds_bytes",
                "gds_sha256",
            )
        }
    write_json(backend_root / "backend_summary.json", summary)
    if summary["accepted"]:
        if results_case_dir.exists():
            raise FileExistsError(f"Result directory already exists: {results_case_dir}")
        results_case_dir.mkdir(parents=True)
        source = backend_root / f"attempt_{best_attempt:02d}"
        shutil.copy2(source / "frontend" / "routing_case.yml", results_case_dir / "routing_case.yml")
        shutil.copy2(source / "frontend" / "crossing_assignment.json", results_case_dir / "crossing_assignment.json")
        shutil.copy2(source / "frozen_backend" / "final.gds", results_case_dir / "final.gds")
        shutil.copy2(source / "frozen_backend" / "continuity.json", results_case_dir / "continuity.json")
        shutil.copy2(source / "frozen_backend" / "pdk_angle_audit.json", results_case_dir / "pdk_angle_audit.json")
        shutil.copy2(backend_root / "backend_summary.json", results_case_dir / "backend_summary.json")
        summary["accepted_result_directory"] = str(results_case_dir.resolve())
        write_json(backend_root / "backend_summary.json", summary)
    return summary
