from __future__ import annotations

import math
from typing import Any


def directed_angle_distance(first: float, second: float) -> float:
    delta = abs((float(first) - float(second)) % 360.0)
    return min(delta, 360.0 - delta)


def effective_hard_access(feasibility: dict[str, Any]) -> bool:
    """Return whether the native router has no credible local escape.

    Long connections may use a grid detour after leaving each port, so a
    behind-tangent or same-direction analytic warning is hard only for a short
    connection.  Compact S-bend radius failures remain hard.
    """

    if not bool(feasibility.get("hard_infeasible", False)):
        return False
    reason = str(feasibility.get("hard_failure_reason") or "")
    if reason == "short_nonopposite_ports_insufficient_corner_access":
        return False
    if not bool(feasibility.get("short_connection", False)) and reason in {
        "target_behind_port_tangent",
        "parallel_same_direction_requires_local_uturn",
    }:
        return False
    return True


def _access_leg_penalty(
    first_point: tuple[float, float],
    first_outward_deg: float,
    second_point: tuple[float, float],
    second_outward_deg: float,
    minimum_access_length_um: float,
) -> float:
    dx = second_point[0] - first_point[0]
    dy = second_point[1] - first_point[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        return 1_000.0
    forward = math.degrees(math.atan2(dy, dx)) % 360.0
    backward = (forward + 180.0) % 360.0
    first_error = directed_angle_distance(first_outward_deg, forward) / 45.0
    second_error = directed_angle_distance(second_outward_deg, backward) / 45.0
    shortfall = max(0.0, minimum_access_length_um - distance)
    normalized = shortfall / max(minimum_access_length_um, 1e-9)
    return first_error * first_error + second_error * second_error + 25.0 * normalized**2


def sbend_min_radius(
    first_point: tuple[float, float],
    first_outward_deg: float,
    second_point: tuple[float, float],
    *,
    samples: int = 128,
) -> float:
    """Minimum sampled radius of the cubic S-bend used by the renderer."""

    radians = math.radians(first_outward_deg)
    direction = (math.cos(radians), math.sin(radians))
    normal = (-direction[1], direction[0])
    delta = (second_point[0] - first_point[0], second_point[1] - first_point[1])
    longitudinal = abs(delta[0] * direction[0] + delta[1] * direction[1])
    lateral = abs(delta[0] * normal[0] + delta[1] * normal[1])
    if longitudinal <= 1e-12:
        return math.inf if lateral <= 1e-12 else 0.0
    if lateral <= 1e-12:
        return math.inf

    minimum = math.inf
    for sample in range(max(16, int(samples)) + 1):
        t = sample / float(max(16, int(samples)))
        u = 1.0 - t
        dx = 1.5 * longitudinal * (u * u + t * t)
        dy = 6.0 * lateral * t * u
        ddx = 3.0 * longitudinal * (2.0 * t - 1.0)
        ddy = 6.0 * lateral * (1.0 - 2.0 * t)
        numerator = abs(dx * ddy - dy * ddx)
        if numerator > 1e-15:
            minimum = min(minimum, (dx * dx + dy * dy) ** 1.5 / numerator)
    return minimum


def port_connection_feasibility(
    first_point: tuple[float, float],
    first_outward_deg: float,
    second_point: tuple[float, float],
    second_outward_deg: float,
    minimum_access_length_um: float,
    direct_threshold_um: float,
    minimum_radius_um: float,
    *,
    samples: int = 128,
) -> dict[str, Any]:
    """Audit two real ports using the native/renderer local geometry contract."""

    dx = float(second_point[0]) - float(first_point[0])
    dy = float(second_point[1]) - float(first_point[1])
    distance = math.hypot(dx, dy)
    first_radians = math.radians(float(first_outward_deg))
    second_radians = math.radians(float(second_outward_deg))
    first_forward = dx * math.cos(first_radians) + dy * math.sin(first_radians)
    second_forward = -dx * math.cos(second_radians) - dy * math.sin(second_radians)
    orientation_delta = directed_angle_distance(first_outward_deg, second_outward_deg)
    opposite = abs(orientation_delta - 180.0) <= 1e-6
    first_direction = (math.cos(first_radians), math.sin(first_radians))
    first_normal = (-first_direction[1], first_direction[0])
    direct_longitudinal = dx * first_direction[0] + dy * first_direction[1]
    direct_lateral = dx * first_normal[0] + dy * first_normal[1]
    direct_straight_feasible = bool(
        opposite
        and direct_longitudinal > 1e-9
        and abs(direct_lateral) <= 1e-3
    )
    minimum_radius = (
        sbend_min_radius(
            first_point,
            first_outward_deg,
            second_point,
            samples=samples,
        )
        if opposite
        else 0.0
    )
    short_connection = direct_threshold_um > 0.0 and distance < direct_threshold_um
    # Under the PDK contract a direct connector is legal only when it is a
    # straight line. The legacy name remains in reports for compatibility,
    # but no free-angle S-bend is considered feasible.
    direct_sbend_feasible = direct_straight_feasible
    coincident_abutment = bool(opposite and distance <= 5e-3)
    native_shortcut_connection = bool(
        opposite
        and distance < min(direct_threshold_um, 2.0 * minimum_radius_um) - 1e-9
    )
    # A non-collinear connection between opposite-facing ports needs a
    # PDK-realizable octilinear dogleg.  The smallest direction-preserving
    # escape uses four 45-degree radius turns; below 4R there is no room for
    # that construction.  Treating this region as a generic grid detour lets
    # the direction solver select the tiny free-angle S-bends that the PDK
    # explicitly forbids, especially between adjacent crossing cells.
    minimum_pdk_detour_um = 4.0 * minimum_radius_um
    insufficient_pdk_detour = bool(
        opposite
        and not direct_straight_feasible
        and distance
        < min(direct_threshold_um, minimum_pdk_detour_um) - 1e-9
    )

    second_direction = (math.cos(second_radians), math.sin(second_radians))
    ray_cross = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    ray_first_um = math.inf
    ray_second_um = math.inf
    corner_required_um = minimum_access_length_um
    corner_access_feasible = True
    if not opposite and abs(ray_cross) > 1e-9:
        ray_first_um = (
            dx * second_direction[1] - dy * second_direction[0]
        ) / ray_cross
        ray_second_um = (
            dx * first_direction[1] - dy * first_direction[0]
        ) / ray_cross
        turn_degrees = directed_angle_distance(
            first_outward_deg, (second_outward_deg + 180.0) % 360.0
        )
        if 1e-9 < turn_degrees < 180.0 - 1e-9:
            turn_radians = math.radians(turn_degrees)
            bend_trim = minimum_radius_um / math.tan(
                (math.pi - turn_radians) / 2.0
            )
            corner_required_um = max(minimum_access_length_um, bend_trim)
        corner_access_feasible = bool(
            ray_first_um + 1e-9 >= corner_required_um
            and ray_second_um + 1e-9 >= corner_required_um
        )

    if coincident_abutment:
        return {
            "distance_um": distance,
            "first_forward_um": first_forward,
            "second_forward_um": second_forward,
            "orientation_delta_deg": orientation_delta,
            "minimum_sbend_radius_um": math.inf,
            "short_connection": short_connection,
            "direct_sbend_feasible": True,
            "direct_straight_feasible": True,
            "direct_lateral_offset_um": 0.0,
            "native_shortcut_connection": native_shortcut_connection,
            "minimum_pdk_detour_um": minimum_pdk_detour_um,
            "insufficient_pdk_detour": False,
            "coincident_abutment": True,
            "endpoint_escape_feasible": True,
            "parallel_same_direction": False,
            "ray_intersection_first_um": ray_first_um,
            "ray_intersection_second_um": ray_second_um,
            "corner_required_um": corner_required_um,
            "corner_access_feasible": True,
            "escape_feasible": True,
            "hard_infeasible": False,
            "hard_failure_reason": None,
            "penalty": 0.0,
        }

    access_penalty = _access_leg_penalty(
        first_point,
        first_outward_deg,
        second_point,
        second_outward_deg,
        minimum_access_length_um,
    )
    forward_scale = max(minimum_access_length_um, minimum_radius_um, 1e-9)
    escape_penalty = (
        max(0.0, -first_forward) / forward_scale
    ) ** 2 + (max(0.0, -second_forward) / forward_scale) ** 2
    closeness = (
        max(0.0, direct_threshold_um - distance) / direct_threshold_um
        if direct_threshold_um > 0.0
        else 0.0
    )
    if not short_connection:
        direct_penalty = 0.0
    elif not opposite:
        direct_penalty = 25.0 * closeness * closeness
    elif direct_straight_feasible:
        direct_penalty = 0.0
    elif not native_shortcut_connection:
        direct_penalty = 0.0
    else:
        direct_penalty = 25.0 * closeness * closeness

    corner_penalty = 0.0
    if short_connection and not opposite and not corner_access_feasible:
        first_shortfall = max(0.0, corner_required_um - ray_first_um)
        second_shortfall = max(0.0, corner_required_um - ray_second_um)
        corner_penalty = 25.0 * closeness * closeness * (
            (first_shortfall / max(corner_required_um, 1e-9)) ** 2
            + (second_shortfall / max(corner_required_um, 1e-9)) ** 2
        )

    endpoint_escape_feasible = bool(
        first_forward >= minimum_radius_um and second_forward >= minimum_radius_um
    )
    parallel_same_direction = bool(
        orientation_delta <= 1e-6
        and (first_forward < minimum_radius_um or second_forward < minimum_radius_um)
    )
    if native_shortcut_connection or insufficient_pdk_detour:
        escape_feasible = direct_straight_feasible
    elif short_connection:
        escape_feasible = bool(
            endpoint_escape_feasible
            and corner_access_feasible
            and not parallel_same_direction
        )
    else:
        escape_feasible = bool(
            endpoint_escape_feasible and not parallel_same_direction
        )

    hard_failure_reason = None
    if native_shortcut_connection and not direct_straight_feasible:
        hard_failure_reason = "short_opposite_ports_require_forbidden_free_angle_sbend"
    elif insufficient_pdk_detour:
        hard_failure_reason = "short_opposite_ports_insufficient_pdk_octilinear_detour"
    elif parallel_same_direction:
        hard_failure_reason = "parallel_same_direction_requires_local_uturn"
    elif first_forward < -1e-9 or second_forward < -1e-9:
        hard_failure_reason = "target_behind_port_tangent"
    elif short_connection and not opposite and not corner_access_feasible:
        hard_failure_reason = "short_nonopposite_ports_insufficient_corner_access"

    return {
        "distance_um": distance,
        "first_forward_um": first_forward,
        "second_forward_um": second_forward,
        "orientation_delta_deg": orientation_delta,
        "minimum_sbend_radius_um": minimum_radius,
        "short_connection": short_connection,
        "direct_sbend_feasible": direct_sbend_feasible,
        "direct_straight_feasible": direct_straight_feasible,
        "direct_lateral_offset_um": direct_lateral,
        "native_shortcut_connection": native_shortcut_connection,
        "minimum_pdk_detour_um": minimum_pdk_detour_um,
        "insufficient_pdk_detour": insufficient_pdk_detour,
        "coincident_abutment": False,
        "endpoint_escape_feasible": endpoint_escape_feasible,
        "parallel_same_direction": parallel_same_direction,
        "ray_intersection_first_um": ray_first_um,
        "ray_intersection_second_um": ray_second_um,
        "corner_required_um": corner_required_um,
        "corner_access_feasible": corner_access_feasible,
        "escape_feasible": escape_feasible,
        "hard_infeasible": hard_failure_reason is not None,
        "hard_failure_reason": hard_failure_reason,
        "penalty": access_penalty
        + 100.0 * escape_penalty
        + direct_penalty
        + corner_penalty,
    }
