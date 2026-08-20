from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .access_contract import effective_hard_access, port_connection_feasibility
from .case_io import InstanceGeometry, NetGeometry
from .direction_solver import solve_net_directions
from .model import PlacedCrossing
from .pcell_geometry import absolute_port, crossing_views


@dataclass(frozen=True)
class _State:
    center: tuple[float, float]
    entry: str
    exit: str
    bbox: tuple[float, float, float, float]
    halo_bbox: tuple[float, float, float, float]
    move_cost: float


def _overlaps(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return (
        min(first[2], second[2]) > max(first[0], second[0]) + 1e-9
        and min(first[3], second[3]) > max(first[1], second[1]) + 1e-9
    )


def _inflate(
    box: tuple[float, float, float, float], amount: float
) -> tuple[float, float, float, float]:
    return box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount


def _rank(item: PlacedCrossing, net_name: str) -> int:
    return int(
        item.event.order_on_net_a
        if item.event.net_a == net_name
        else item.event.order_on_net_b
    )


def _pair(item: PlacedCrossing, net_name: str) -> tuple[str, str]:
    return (
        item.event.net_a_ports
        if item.event.net_a == net_name
        else item.event.net_b_ports
    )


def _box(
    center: tuple[float, float], view: dict[str, Any]
) -> tuple[float, float, float, float]:
    local = [float(value) for value in view["bbox_centered_um"]]
    return (
        center[0] + local[0],
        center[1] + local[1],
        center[0] + local[2],
        center[1] + local[3],
    )


def _leg(
    first_point: tuple[float, float],
    first_angle: float,
    second_point: tuple[float, float],
    second_angle: float,
    minimum_access: float,
    direct_threshold: float,
    minimum_radius: float,
) -> dict[str, Any]:
    return port_connection_feasibility(
        first_point,
        first_angle,
        second_point,
        second_angle,
        minimum_access,
        direct_threshold,
        minimum_radius,
    )


def _requires_straight_short_crossing_leg(audit: dict[str, Any]) -> bool:
    """Return whether a compact crossing-to-crossing leg must be collinear.

    Opposite-facing crossing ports that are close but laterally offset are the
    exact topology that legacy renderers used to hide with a tiny free-angle
    S-bend.  The production PDK has no such primitive.  Keeping these local
    links straight is both stronger and more predictable than asking global
    A* to leave the crossing cluster, make a multi-bend detour, and return.
    """

    return bool(
        audit.get("short_connection", False)
        and abs(float(audit.get("orientation_delta_deg", 0.0)) - 180.0)
        <= 1e-6
        and not audit.get("direct_straight_feasible", False)
    )


def _current_chain_feasible(
    placed: list[PlacedCrossing],
    net_name: str,
    nets: dict[str, NetGeometry],
    manifest: dict[str, Any],
    minimum_access: float,
    minimum_radius: float,
    direct_threshold: float,
) -> bool:
    try:
        _directions, audits = solve_net_directions(
            placed,
            nets,
            manifest,
            minimum_access,
            minimum_radius,
            direct_threshold,
            selected_nets={net_name},
        )
    except RuntimeError:
        return False
    selected = audits[net_name]["selected_segment_access"]
    # The first and last legs touch original benchmark components.  Only the
    # interior entries are crossing-to-crossing links whose placement is under
    # this refiner's control.
    return not any(
        _requires_straight_short_crossing_leg(item) for item in selected[1:-1]
    )


def _repair_chain(
    placed: list[PlacedCrossing],
    net_name: str,
    nets: dict[str, NetGeometry],
    obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[PlacedCrossing] | None, dict[str, Any]]:
    chain = sorted(
        [
            item
            for item in placed
            if net_name in (item.event.net_a, item.event.net_b)
        ],
        key=lambda item: (_rank(item, net_name), item.event.event_id),
    )
    if len(chain) < 2:
        return None, {"net": net_name, "status": "not_a_chain"}

    views = crossing_views(manifest)
    grid = float(config.get("grid_um", 2.0))
    radius = max(
        grid,
        float(config.get("direction_chain_refinement_radius_um", 16.0)),
    )
    center_limit = max(
        16,
        int(config.get("direction_chain_candidate_limit", 240)),
    )
    minimum_access = float(config.get("minimum_access_um", 10.0))
    minimum_radius = float(config.get("bend_radius_um", 5.0))
    direct_threshold = float(
        config.get(
            "short_direct_access_threshold_um",
            2.0 * (minimum_access + minimum_radius),
        )
    )
    halo = float(manifest.get("halo_um", 0.0))
    fixed_clearance = max(halo, minimum_access)
    fixed_boxes = [item.bbox for item in obstacles]
    chain_ids = {item.event.event_id for item in chain}
    nonchain = [item for item in placed if item.event.event_id not in chain_ids]

    nonchain_geometry = []
    for item in nonchain:
        box = _box(item.center_um, views[float(item.rotation_deg)])
        nonchain_geometry.append((item, box, _inflate(box, halo)))

    offset_limit = int(math.floor(radius / grid + 1e-9))
    offsets = sorted(
        (
            (dx * dx + dy * dy, dx * grid, dy * grid)
            for dx in range(-offset_limit, offset_limit + 1)
            for dy in range(-offset_limit, offset_limit + 1)
        ),
        key=lambda value: (value[0], value[1], value[2]),
    )
    domains: list[list[_State]] = []
    for item in chain:
        view = views[float(item.rotation_deg)]
        pair = _pair(item, net_name)
        states = []
        accepted_centers = 0
        for squared_steps, dx, dy in offsets:
            center = (item.center_um[0] + dx, item.center_um[1] + dy)
            box = _box(center, view)
            halo_box = _inflate(box, halo)
            if (
                box[0] < die[0]
                or box[1] < die[1]
                or box[2] > die[2]
                or box[3] > die[3]
                or any(
                    _overlaps(_inflate(box, fixed_clearance), fixed)
                    for fixed in fixed_boxes
                )
            ):
                continue
            collision = False
            for other, other_box, other_halo in nonchain_geometry:
                shared = bool(
                    {item.event.net_a, item.event.net_b}
                    & {other.event.net_a, other.event.net_b}
                )
                if _overlaps(
                    box if shared else halo_box,
                    other_box if shared else other_halo,
                ):
                    collision = True
                    break
            if collision:
                continue
            accepted_centers += 1
            move_cost = float(squared_steps) * grid * grid
            for entry, exit_name in (pair, (pair[1], pair[0])):
                states.append(
                    _State(
                        center,
                        entry,
                        exit_name,
                        box,
                        halo_box,
                        move_cost,
                    )
                )
            if accepted_centers >= center_limit:
                break
        if not states:
            return None, {
                "net": net_name,
                "status": "empty_candidate_domain",
                "event": item.event.event_id,
            }
        domains.append(states)

    net = nets[net_name]
    costs: list[list[float]] = []
    parents: list[list[int | None]] = []
    for index, (item, states) in enumerate(zip(chain, domains)):
        view = views[float(item.rotation_deg)]
        row = [math.inf] * len(states)
        parent_row: list[int | None] = [None] * len(states)
        for state_index, state in enumerate(states):
            entry_point, entry_angle = absolute_port(state.center, view, state.entry)
            if index == 0:
                audit = _leg(
                    (net.source.x, net.source.y),
                    net.source.orientation,
                    entry_point,
                    entry_angle,
                    minimum_access,
                    direct_threshold,
                    minimum_radius,
                )
                if not effective_hard_access(audit):
                    row[state_index] = state.move_cost + 50.0 * float(
                        audit["penalty"]
                    )
                continue
            previous_item = chain[index - 1]
            previous_view = views[float(previous_item.rotation_deg)]
            for previous_index, previous_state in enumerate(domains[index - 1]):
                if costs[index - 1][previous_index] == math.inf:
                    continue
                if _overlaps(previous_state.bbox, state.bbox):
                    continue
                exit_point, exit_angle = absolute_port(
                    previous_state.center,
                    previous_view,
                    previous_state.exit,
                )
                audit = _leg(
                    exit_point,
                    exit_angle,
                    entry_point,
                    entry_angle,
                    minimum_access,
                    direct_threshold,
                    minimum_radius,
                )
                if effective_hard_access(audit) or (
                    _requires_straight_short_crossing_leg(audit)
                ):
                    continue
                candidate_cost = (
                    costs[index - 1][previous_index]
                    + state.move_cost
                    + 50.0 * float(audit["penalty"])
                )
                if candidate_cost < row[state_index]:
                    row[state_index] = candidate_cost
                    parent_row[state_index] = previous_index
        costs.append(row)
        parents.append(parent_row)

    final_cost = math.inf
    final_state: int | None = None
    last_item = chain[-1]
    last_view = views[float(last_item.rotation_deg)]
    for state_index, state in enumerate(domains[-1]):
        if costs[-1][state_index] == math.inf:
            continue
        exit_point, exit_angle = absolute_port(state.center, last_view, state.exit)
        audit = _leg(
            exit_point,
            exit_angle,
            (net.target.x, net.target.y),
            net.target.orientation,
            minimum_access,
            direct_threshold,
            minimum_radius,
        )
        if effective_hard_access(audit):
            continue
        candidate_cost = costs[-1][state_index] + 50.0 * float(audit["penalty"])
        if candidate_cost < final_cost:
            final_cost = candidate_cost
            final_state = state_index
    if final_state is None:
        return None, {
            "net": net_name,
            "status": "no_direction_consistent_chain",
            "domain_sizes": [len(values) for values in domains],
        }

    selected = [0] * len(chain)
    selected[-1] = final_state
    for index in range(len(chain) - 1, 0, -1):
        previous = parents[index][selected[index]]
        if previous is None:
            return None, {"net": net_name, "status": "broken_predecessor_chain"}
        selected[index - 1] = previous
    selected_states = [
        domains[index][state_index] for index, state_index in enumerate(selected)
    ]
    for first_index, first in enumerate(selected_states):
        for second in selected_states[first_index + 1 :]:
            if _overlaps(first.bbox, second.bbox):
                return None, {
                    "net": net_name,
                    "status": "nonadjacent_chain_geometry_conflict",
                }

    centers = {
        item.event.event_id: state.center
        for item, state in zip(chain, selected_states)
    }
    repaired = []
    changes = []
    for item in placed:
        center = centers.get(item.event.event_id, item.center_um)
        repaired.append(
            PlacedCrossing(
                event=item.event,
                center_um=center,
                rotation_deg=item.rotation_deg,
                legal=item.legal,
                displacement_um=math.dist(center, item.event.ideal_center_um),
                failure_reason=item.failure_reason,
            )
        )
        if center != item.center_um:
            changes.append(
                {
                    "crossing": item.event.event_id,
                    "before_um": list(item.center_um),
                    "after_um": list(center),
                }
            )
    if not _current_chain_feasible(
        repaired,
        net_name,
        nets,
        manifest,
        minimum_access,
        minimum_radius,
        direct_threshold,
    ):
        return None, {"net": net_name, "status": "post_repair_direction_reject"}
    return repaired, {
        "net": net_name,
        "status": "repaired",
        "objective": final_cost,
        "domain_sizes": [len(values) for values in domains],
        "changes": changes,
    }


def refine_direction_chains(
    placed: list[PlacedCrossing],
    nets: dict[str, NetGeometry],
    obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[PlacedCrossing], dict[str, Any]]:
    """Repair crossing chains that lack one globally consistent port direction."""

    enabled = bool(config.get("direction_chain_refinement", True))
    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "pdk_octilinear_direction_chain_dynamic_programming_v2",
        "enabled": enabled,
        "repairs": [],
    }
    if not enabled:
        report.update(status="disabled", repaired_net_count=0)
        return placed, report

    current = list(placed)
    events_per_net: dict[str, int] = {}
    for item in current:
        events_per_net[item.event.net_a] = events_per_net.get(item.event.net_a, 0) + 1
        events_per_net[item.event.net_b] = events_per_net.get(item.event.net_b, 0) + 1
    candidates = sorted(
        (name for name, count in events_per_net.items() if count >= 2),
        key=lambda name: (-events_per_net[name], name),
    )
    maximum_passes = max(1, int(config.get("direction_chain_refinement_passes", 3)))
    unresolved: list[str] = []
    for _pass in range(maximum_passes):
        changed = False
        unresolved = []
        for net_name in candidates:
            if _current_chain_feasible(
                current,
                net_name,
                nets,
                manifest,
                float(config.get("minimum_access_um", 10.0)),
                float(config.get("bend_radius_um", 5.0)),
                float(
                    config.get(
                        "short_direct_access_threshold_um",
                        2.0
                        * (
                            float(config.get("minimum_access_um", 10.0))
                            + float(config.get("bend_radius_um", 5.0))
                        ),
                    )
                ),
            ):
                continue
            repaired, repair_report = _repair_chain(
                current,
                net_name,
                nets,
                obstacles,
                die,
                manifest,
                config,
            )
            report["repairs"].append(repair_report)
            if repaired is None:
                unresolved.append(net_name)
                continue
            current = repaired
            changed = True
        if not changed:
            break
    report["repaired_net_count"] = sum(
        item.get("status") == "repaired" for item in report["repairs"]
    )
    report["unresolved_nets"] = unresolved
    report["status"] = "pass" if not unresolved else "partial"
    return current, report
