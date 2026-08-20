from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

from .case_io import die_area, instance_geometry, placement_tuple, two_pin_nets


_EPSILON = 1e-8


class _UnionFind:
    def __init__(self, names: list[str]) -> None:
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        parent = self.parent[name]
        if parent != name:
            self.parent[name] = self.find(parent)
        return self.parent[name]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            self.parent[second_root] = first_root
        else:
            self.parent[first_root] = second_root


def _source_status(node: dict[str, Any]) -> str:
    placement = (node.get("settings") or {}).get("placement") or []
    return str(placement[0] if placement else "").upper()


def _alignment_groups(case: dict[str, Any], axis: str) -> dict[str, str]:
    instances = case.get("instances") or {}
    union = _UnionFind([str(name) for name in instances])
    axis_anchors = (
        {"left", "right", "center_x", "x", "horizontal"}
        if axis == "x"
        else {"lower", "upper", "center_y", "y", "vertical"}
    )
    for constraint in (case.get("constraints") or {}).values():
        if str(constraint.get("type", "")).lower() != "alignment":
            continue
        anchor = str((constraint.get("settings") or {}).get("anchor", "")).lower()
        if anchor not in axis_anchors:
            continue
        names = [str(name) for name in constraint.get("objects") or []]
        names = [name for name in names if name in instances]
        for name in names[1:]:
            union.union(names[0], name)
    return {name: union.find(name) for name in instances}


def _cardinal(angle: float) -> int | None:
    snapped = int(round(float(angle) / 90.0)) % 4
    if abs((float(angle) - 90.0 * snapped + 180.0) % 360.0 - 180.0) > 1e-6:
        return None
    return snapped


def _access_constraints(
    case: dict[str, Any],
    axis: str,
    groups: dict[str, str],
    required_um: float,
    grid_um: float,
    collinear_required_um: float,
    collinear_lateral_threshold_um: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instances = case.get("instances") or {}
    constraints: list[dict[str, Any]] = []
    immutable_short: list[dict[str, Any]] = []
    for net_name, net in sorted(two_pin_nets(case).items()):
        first_cardinal = _cardinal(net.source.orientation)
        second_cardinal = _cardinal(net.target.orientation)
        if first_cardinal is None or second_cardinal is None:
            continue
        if (first_cardinal - second_cardinal) % 4 != 2:
            continue
        horizontal = first_cardinal in {0, 2}
        if horizontal != (axis == "x"):
            continue
        delta = (
            net.target.x - net.source.x
            if axis == "x"
            else net.target.y - net.source.y
        )
        first_sign = 1.0 if first_cardinal in {0, 1} else -1.0
        if delta * first_sign <= _EPSILON:
            continue
        projection = abs(delta)
        perpendicular = abs(
            net.target.y - net.source.y
            if axis == "x"
            else net.target.x - net.source.x
        )
        selected_required_um = (
            collinear_required_um
            if perpendicular <= collinear_lateral_threshold_um + _EPSILON
            else required_um
        )
        if projection + _EPSILON >= selected_required_um:
            continue
        first_name = net.source.instance
        second_name = net.target.instance
        first_status = _source_status(instances[first_name])
        second_status = _source_status(instances[second_name])
        record = {
            "net": net_name,
            "axis": axis,
            "first_port": net.source.name,
            "second_port": net.target.name,
            "original_forward_corridor_um": projection,
            "perpendicular_offset_um": perpendicular,
            "required_forward_corridor_um": selected_required_um,
            "near_collinear": perpendicular
            <= collinear_lateral_threshold_um + _EPSILON,
        }
        if first_status != "UNPLACED" and second_status != "UNPLACED":
            immutable_short.append(record)
            continue
        if delta > 0.0:
            lower_name, upper_name = first_name, second_name
        else:
            lower_name, upper_name = second_name, first_name
        lower_group = groups[lower_name]
        upper_group = groups[upper_name]
        deficit = selected_required_um - projection
        snapped_deficit = math.ceil((deficit - _EPSILON) / grid_um) * grid_um
        constraints.append(
            {
                **record,
                "lower_group": lower_group,
                "upper_group": upper_group,
                "minimum_shift_difference_um": float(snapped_deficit),
            }
        )
    return constraints, immutable_short


def _weak_components(
    nodes: set[str], edges: list[dict[str, Any]]
) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        lower = str(edge["lower_group"])
        upper = str(edge["upper_group"])
        adjacency[lower].add(upper)
        adjacency[upper].add(lower)
    result = []
    remaining = set(nodes)
    while remaining:
        seed = min(remaining)
        component = {seed}
        queue = deque([seed])
        remaining.remove(seed)
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return result


def _solve_axis(
    case: dict[str, Any],
    axis: str,
    groups: dict[str, str],
    constraints: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not constraints:
        return {}, []
    instances = case.get("instances") or {}
    members: dict[str, list[str]] = defaultdict(list)
    for name, group in groups.items():
        members[group].append(name)
    nodes = {
        str(value)
        for edge in constraints
        for value in (edge["lower_group"], edge["upper_group"])
    }
    for edge in constraints:
        if edge["lower_group"] == edge["upper_group"]:
            raise ValueError(
                "Alignment constraint prevents source port legalization for "
                f"{edge['net']} on {axis}"
            )
    movable = {
        group: all(_source_status(instances[name]) == "UNPLACED" for name in members[group])
        for group in nodes
    }
    shifts = {group: 0.0 for group in nodes}
    ordered = sorted(
        constraints,
        key=lambda item: (
            str(item["lower_group"]),
            str(item["upper_group"]),
            str(item["net"]),
        ),
    )
    maximum_rounds = max(32, 16 * len(nodes) * max(1, len(ordered)))
    for _round in range(maximum_rounds):
        maximum_gap = 0.0
        for edge in ordered:
            lower = str(edge["lower_group"])
            upper = str(edge["upper_group"])
            required = float(edge["minimum_shift_difference_um"])
            gap = required - (shifts[upper] - shifts[lower])
            maximum_gap = max(maximum_gap, gap)
            if gap <= _EPSILON:
                continue
            lower_movable = movable[lower]
            upper_movable = movable[upper]
            if lower_movable and upper_movable:
                shifts[lower] -= gap / 2.0
                shifts[upper] += gap / 2.0
            elif lower_movable:
                shifts[lower] -= gap
            elif upper_movable:
                shifts[upper] += gap
            else:
                raise ValueError(
                    f"Source port corridor {edge['net']} is short but both "
                    "alignment groups contain FIXED instances"
                )
        if maximum_gap <= _EPSILON:
            break
    else:
        raise ValueError(f"Source port legalization did not converge on {axis}")

    die = die_area(case)
    low_die, high_die = (die[0], die[2]) if axis == "x" else (die[1], die[3])
    component_reports = []
    for component in _weak_components(nodes, ordered):
        has_fixed = any(not movable[group] for group in component)
        if not has_fixed:
            values = [shifts[group] for group in component]
            center = (min(values) + max(values)) / 2.0
            for group in component:
                shifts[group] -= center
        lower_translation = -math.inf
        upper_translation = math.inf
        for group in component:
            for name in members[group]:
                geometry = instance_geometry(case, name, instances[name])
                box = geometry.bbox
                low = box[0] if axis == "x" else box[1]
                high = box[2] if axis == "x" else box[3]
                lower_translation = max(
                    lower_translation, low_die - low - shifts[group]
                )
                upper_translation = min(
                    upper_translation, high_die - high - shifts[group]
                )
        if lower_translation > upper_translation + _EPSILON:
            raise ValueError(
                f"No die-legal translation exists for source legalization on {axis}"
            )
        boundary_corridors = []
        for net_name, net in sorted(two_pin_nets(case).items()):
            first_group = groups[net.source.instance]
            second_group = groups[net.target.instance]
            first_inside = first_group in component
            second_inside = second_group in component
            if first_inside == second_inside:
                continue
            first_cardinal = _cardinal(net.source.orientation)
            second_cardinal = _cardinal(net.target.orientation)
            if first_cardinal is None or second_cardinal is None:
                continue
            if (first_cardinal - second_cardinal) % 4 != 2:
                continue
            horizontal = first_cardinal in {0, 2}
            if horizontal != (axis == "x"):
                continue
            delta = (
                net.target.x - net.source.x
                if axis == "x"
                else net.target.y - net.source.y
            )
            first_sign = 1.0 if first_cardinal in {0, 1} else -1.0
            if delta * first_sign <= _EPSILON:
                continue
            moving_group = first_group if first_inside else second_group
            delta_sign = 1.0 if delta > 0.0 else -1.0
            coefficient = delta_sign * (1.0 if second_inside else -1.0)
            original_corridor = abs(delta)
            base_corridor = original_corridor + coefficient * shifts[moving_group]
            boundary_corridors.append(
                {
                    "net": net_name,
                    "moving_group": moving_group,
                    "original_forward_corridor_um": original_corridor,
                    "base_forward_corridor_um": base_corridor,
                    "translation_coefficient": coefficient,
                }
            )

        translation_candidates = {
            min(upper_translation, max(lower_translation, 0.0)),
            lower_translation,
            upper_translation,
        }
        for item in boundary_corridors:
            preserve_boundary = -shifts[str(item["moving_group"])]
            if lower_translation - _EPSILON <= preserve_boundary <= upper_translation + _EPSILON:
                translation_candidates.add(preserve_boundary)

        def translation_score(candidate: float) -> tuple[float, float, float, float, float]:
            # A component translation does not improve its internal corridor
            # constraints. Prefer preserving the source coordinates of groups
            # that connect to the rest of the circuit; duplicate boundary nets
            # intentionally weight a high-fanout terminal more strongly.
            boundary_displacement = sum(
                abs(shifts[str(item["moving_group"])] + candidate)
                for item in boundary_corridors
            )
            minimum_boundary = min(
                (
                    float(item["base_forward_corridor_um"])
                    + float(item["translation_coefficient"]) * candidate
                    for item in boundary_corridors
                ),
                default=math.inf,
            )
            maximum_displacement = max(
                abs(shifts[group] + candidate) for group in component
            )
            return (
                boundary_displacement,
                -minimum_boundary,
                maximum_displacement,
                abs(candidate),
                candidate,
            )

        translation = (
            0.0
            if has_fixed
            else min(translation_candidates, key=translation_score)
        )
        if not (lower_translation - _EPSILON <= translation <= upper_translation + _EPSILON):
            raise ValueError(
                f"Source legalization on {axis} would move a FIXED alignment group"
            )
        for group in component:
            shifts[group] += translation
        for item in boundary_corridors:
            item["legalized_forward_corridor_um"] = (
                float(item["base_forward_corridor_um"])
                + float(item["translation_coefficient"]) * translation
            )
        component_reports.append(
            {
                "groups": sorted(component),
                "contains_fixed_group": has_fixed,
                "common_die_translation_um": translation,
                "boundary_corridors": boundary_corridors,
                "minimum_boundary_corridor_um": min(
                    (
                        float(item["legalized_forward_corridor_um"])
                        for item in boundary_corridors
                    ),
                    default=None,
                ),
            }
        )

    # The pairwise projection solver converges geometrically.  Canonicalize
    # its sub-nanometer residue before coordinates enter an exact audit: a
    # mathematically stationary alignment group must remain byte-for-byte at
    # its source coordinate rather than acquire a 1e-9 um pseudo-movement.
    for group, value in list(shifts.items()):
        canonical = round(value, 6)
        shifts[group] = 0.0 if abs(canonical) < 1e-6 else canonical

    for edge in ordered:
        lower = str(edge["lower_group"])
        upper = str(edge["upper_group"])
        required = float(edge["minimum_shift_difference_um"])
        if shifts[upper] - shifts[lower] + _EPSILON < required:
            raise ValueError(f"Unresolved source port corridor constraint: {edge['net']}")
    return shifts, component_reports


def _boxes_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return (
        min(first[2], second[2]) - max(first[0], second[0]) > _EPSILON
        and min(first[3], second[3]) - max(first[1], second[1]) > _EPSILON
    )


def legalize_source_unplaced_instances(
    case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Create PDK-sized port corridors by moving source ``UNPLACED`` groups.

    The move is deterministic and status-driven.  Source ``FIXED`` instances
    never move.  Alignment constraints are represented as rigid one-axis
    groups; all moved instances are frozen only after legality is verified.
    """

    enabled = bool(config.get("enabled", False))
    required_um = float(config.get("minimum_forward_corridor_um", 30.0))
    grid_um = float(config.get("displacement_grid_um", 2.0))
    collinear_required_um = float(
        config.get("collinear_minimum_forward_corridor_um", required_um)
    )
    collinear_lateral_threshold_um = float(
        config.get("collinear_lateral_threshold_um", 0.0)
    )
    instances = case.get("instances") or {}
    original = {
        str(name): {
            "status": _source_status(node),
            "lower_left_um": placement_tuple(node)[0],
            "orientation": placement_tuple(node)[1],
        }
        for name, node in instances.items()
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "pdk_port_access_difference_constraints_v1",
        "enabled": enabled,
        "minimum_forward_corridor_um": required_um,
        "collinear_minimum_forward_corridor_um": collinear_required_um,
        "collinear_lateral_threshold_um": collinear_lateral_threshold_um,
        "displacement_grid_um": grid_um,
        "axis_constraints": {},
        "immutable_short_corridors": [],
        "movements": [],
        "moved_instance_count": 0,
    }
    if not enabled:
        return report
    if (
        required_um <= 0.0
        or collinear_required_um < required_um
        or collinear_lateral_threshold_um < 0.0
        or grid_um <= 0.0
    ):
        raise ValueError("Source legalization distances must be positive")

    axis_shifts: dict[str, dict[str, float]] = {}
    axis_groups: dict[str, dict[str, str]] = {}
    for axis in ("x", "y"):
        groups = _alignment_groups(case, axis)
        constraints, immutable_short = _access_constraints(
            case,
            axis,
            groups,
            required_um,
            grid_um,
            collinear_required_um,
            collinear_lateral_threshold_um,
        )
        shifts, components = _solve_axis(case, axis, groups, constraints)
        axis_groups[axis] = groups
        axis_shifts[axis] = shifts
        report["axis_constraints"][axis] = {
            "constraints": constraints,
            "components": components,
            "group_shifts_um": dict(sorted(shifts.items())),
        }
        report["immutable_short_corridors"].extend(immutable_short)

    for name, node in instances.items():
        name = str(name)
        dx = axis_shifts["x"].get(axis_groups["x"][name], 0.0)
        dy = axis_shifts["y"].get(axis_groups["y"][name], 0.0)
        if (abs(dx) > _EPSILON or abs(dy) > _EPSILON) and original[name]["status"] != "UNPLACED":
            raise ValueError(f"Source legalization attempted to move FIXED instance {name}")
        placement = node["settings"]["placement"]
        placement[1][0] = float(placement[1][0]) + dx
        placement[1][1] = float(placement[1][1]) + dy
        if abs(dx) > _EPSILON or abs(dy) > _EPSILON:
            report["movements"].append(
                {
                    "instance": name,
                    "source_status": original[name]["status"],
                    "source_lower_left_um": original[name]["lower_left_um"],
                    "legalized_lower_left_um": [
                        float(placement[1][0]),
                        float(placement[1][1]),
                    ],
                    "displacement_um": [dx, dy],
                }
            )

    geometries = {
        str(name): instance_geometry(case, str(name), node)
        for name, node in instances.items()
    }
    collisions = []
    names = sorted(geometries)
    moved_names = {item["instance"] for item in report["movements"]}
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            if first_name not in moved_names and second_name not in moved_names:
                continue
            if _boxes_overlap(geometries[first_name].bbox, geometries[second_name].bbox):
                collisions.append([first_name, second_name])
    if collisions:
        preview = ", ".join("/".join(pair) for pair in collisions[:8])
        raise ValueError(f"Source legalization created component overlaps: {preview}")

    for node in instances.values():
        node["settings"]["placement"][0] = "FIXED"
    report["movements"].sort(key=lambda item: item["instance"])
    report["moved_instance_count"] = len(report["movements"])
    report["component_overlap_count"] = len(collisions)
    report["passed"] = True
    return report
