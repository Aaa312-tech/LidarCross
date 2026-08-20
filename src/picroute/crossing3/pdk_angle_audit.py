from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import klayout.db as kdb
import yaml

from .pdk_angle_contract import (
    ACCESS_LATERAL_TOLERANCE_UM,
    ANGLE_TOLERANCE_DEG,
)

def _heading(first: list[float], second: list[float]) -> float | None:
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    if math.hypot(dx, dy) <= 1e-12:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _angle_error(actual: float, expected: float) -> float:
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def _grid_error(angle: float) -> tuple[float, float]:
    nearest = (round(float(angle) / 45.0) * 45.0) % 360.0
    return nearest, _angle_error(angle, nearest)


def _turn_angle(first: float, second: float) -> float:
    return abs((float(second) - float(first) + 180.0) % 360.0 - 180.0)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _rendered_paths(net: dict[str, Any]) -> list[dict[str, Any]]:
    paths = list(net.get("paths") or [])
    if paths:
        return paths
    return [
        {
            "points": net.get("routed_path_um") or [],
            "start_port": net.get("route_start_port") or net.get("source_port"),
            "end_port": net.get("route_end_port") or net.get("target_port"),
        }
    ]


def _port_lateral_offset(
    port: dict[str, Any], point: list[float]
) -> tuple[float, float]:
    center = port.get("center") or []
    if len(center) != 2:
        raise ValueError(f"Port lacks a two-dimensional center: {port}")
    theta = math.radians(float(port.get("orientation", 0.0)))
    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    longitudinal = dx * math.cos(theta) + dy * math.sin(theta)
    lateral = -dx * math.sin(theta) + dy * math.cos(theta)
    return longitudinal, lateral


def audit_route_geometry(route_result: Path) -> dict[str, Any]:
    """Reject every centerline feature outside the discrete PDK contract.

    The renderer may use a radius-correct PDK bend for a 45 or 90 degree
    centerline turn.  It may not use a free-angle S-bend to absorb endpoint
    quantization.  Therefore the physical port-to-polyline lateral offset is
    part of the sign-off contract, rather than merely the endpoint tangent.
    """

    data = _load_yaml(route_result)
    short_sbends: list[dict[str, Any]] = []
    prohibited_short_sbends: list[dict[str, Any]] = []
    access_lateral_offsets: list[dict[str, Any]] = []
    off_grid_segments: list[dict[str, Any]] = []
    unsupported_turns: list[dict[str, Any]] = []
    zero_length_segments: list[dict[str, Any]] = []
    routed_path_count = 0

    for net_name, net in (data.get("nets") or {}).items():
        if not bool(net.get("routed", False)):
            continue
        if bool(net.get("short_sbend", False)):
            start_port = net.get("route_start_port") or net.get("source_port")
            end_port = net.get("route_end_port") or net.get("target_port")
            record = {
                "net": str(net_name),
                "length_um": float(net.get("short_sbend_length") or 0.0),
                "source": net.get("source"),
                "target": net.get("target"),
            }
            if isinstance(start_port, dict) and isinstance(end_port, dict):
                _longitudinal, lateral = _port_lateral_offset(
                    start_port, [float(value) for value in end_port["center"]]
                )
                record["lateral_offset_um"] = lateral
                record["renders_as_straight"] = (
                    abs(lateral) <= ACCESS_LATERAL_TOLERANCE_UM
                )
            else:
                record["renders_as_straight"] = False
            short_sbends.append(record)
            if not record["renders_as_straight"]:
                prohibited_short_sbends.append(record)
            continue
        for path_index, rendered_path in enumerate(_rendered_paths(net)):
            points = [
                [float(point[0]), float(point[1])]
                for point in (rendered_path.get("points") or [])
            ]
            if len(points) < 2:
                continue
            routed_path_count += 1
            route_cell = f"{net_name}_{path_index}_route"
            headings: list[float | None] = []
            for segment_index in range(len(points) - 1):
                heading = _heading(points[segment_index], points[segment_index + 1])
                headings.append(heading)
                if heading is None:
                    zero_length_segments.append(
                        {
                            "net": str(net_name),
                            "route_cell": route_cell,
                            "segment_index": segment_index,
                            "point_um": points[segment_index],
                        }
                    )
                    continue
                nearest, error = _grid_error(heading)
                if error > ANGLE_TOLERANCE_DEG:
                    off_grid_segments.append(
                        {
                            "net": str(net_name),
                            "route_cell": route_cell,
                            "segment_index": segment_index,
                            "start_um": points[segment_index],
                            "end_um": points[segment_index + 1],
                            "heading_deg": heading,
                            "nearest_supported_heading_deg": nearest,
                            "error_deg": error,
                        }
                    )
            valid_headings = [
                (index, heading)
                for index, heading in enumerate(headings)
                if heading is not None
            ]
            for pair_index in range(len(valid_headings) - 1):
                first_index, first_heading = valid_headings[pair_index]
                second_index, second_heading = valid_headings[pair_index + 1]
                if second_index != first_index + 1:
                    continue
                turn = _turn_angle(first_heading, second_heading)
                if not any(
                    abs(turn - supported) <= ANGLE_TOLERANCE_DEG
                    for supported in (0.0, 45.0, 90.0)
                ):
                    unsupported_turns.append(
                        {
                            "net": str(net_name),
                            "route_cell": route_cell,
                            "vertex_index": second_index,
                            "vertex_um": points[second_index],
                            "incoming_heading_deg": first_heading,
                            "outgoing_heading_deg": second_heading,
                            "turn_deg": turn,
                        }
                    )

            start_port = (
                rendered_path.get("start_port")
                or net.get("route_start_port")
                or net.get("source_port")
            )
            end_port = (
                rendered_path.get("end_port")
                or net.get("route_end_port")
                or net.get("target_port")
            )
            for side, port, point in (
                ("start", start_port, points[0]),
                ("end", end_port, points[-1]),
            ):
                if not isinstance(port, dict):
                    continue
                longitudinal, lateral = _port_lateral_offset(port, point)
                if abs(lateral) > ACCESS_LATERAL_TOLERANCE_UM:
                    access_lateral_offsets.append(
                        {
                            "net": str(net_name),
                            "route_cell": route_cell,
                            "side": side,
                            "port": port.get("name"),
                            "port_center_um": [float(v) for v in port["center"]],
                            "route_join_um": point,
                            "port_orientation_deg": float(
                                port.get("orientation", 0.0)
                            ),
                            "longitudinal_offset_um": longitudinal,
                            "lateral_offset_um": lateral,
                        }
                    )

    violations = {
        "short_sbend_count": len(short_sbends),
        "short_sbends": short_sbends,
        "prohibited_short_sbend_count": len(prohibited_short_sbends),
        "prohibited_short_sbends": prohibited_short_sbends,
        "access_lateral_offset_count": len(access_lateral_offsets),
        "access_lateral_offsets": access_lateral_offsets,
        "off_grid_segment_count": len(off_grid_segments),
        "off_grid_segments": off_grid_segments,
        "unsupported_turn_count": len(unsupported_turns),
        "unsupported_turns": unsupported_turns,
        "zero_length_segment_count": len(zero_length_segments),
        "zero_length_segments": zero_length_segments,
    }
    return {
        "route_result": str(route_result.resolve()),
        "routed_path_count": routed_path_count,
        **violations,
        "clean": not any(
            violations[key]
            for key in (
                "prohibited_short_sbend_count",
                "access_lateral_offset_count",
                "off_grid_segment_count",
                "unsupported_turn_count",
                "zero_length_segment_count",
            )
        ),
    }


def audit_renderer_geometry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    recoveries = list(value.get("recoveries") or [])
    access_junctions = list(value.get("access_junctions") or [])
    unsupported_recoveries = []
    for recovery in recoveries:
        method = str(recovery.get("method") or "")
        points = recovery.get("recovery_points_um") or []
        reasons = []
        if method == "short_dubins_euler":
            reasons.append("free_angle_dubins_recovery")
        headings = []
        for index in range(max(0, len(points) - 1)):
            heading = _heading(points[index], points[index + 1])
            if heading is None:
                reasons.append("zero_length_recovery_segment")
                continue
            headings.append(heading)
            if _grid_error(heading)[1] > ANGLE_TOLERANCE_DEG:
                reasons.append("off_grid_recovery_segment")
        for index in range(len(headings) - 1):
            turn = _turn_angle(headings[index], headings[index + 1])
            if not any(
                abs(turn - supported) <= ANGLE_TOLERANCE_DEG
                for supported in (0.0, 45.0, 90.0)
            ):
                reasons.append("unsupported_recovery_turn")
        if reasons:
            unsupported_recoveries.append(
                {
                    "route_cell": recovery.get("route_cell"),
                    "net": recovery.get("net"),
                    "method": method,
                    "reasons": sorted(set(reasons)),
                    "recovery_points_um": points,
                }
            )
    prohibited_access_sbends = []
    for junction in access_junctions:
        method = str(junction.get("method") or "")
        if "sbend" not in method.lower():
            continue
        label = str(junction.get("label") or "")
        prohibited_access_sbends.append(
            {
                "label": label,
                "net": label.split("/", 1)[0] if label else None,
                "method": method,
                "endpoint_error_um": float(
                    junction.get("endpoint_error_um") or 0.0
                ),
                "orientation_error_deg": float(
                    junction.get("orientation_error_deg") or 0.0
                ),
            }
        )
    return {
        "geometry_report": str(path.resolve()),
        "recovery_count": len(recoveries),
        "unsupported_recovery_count": len(unsupported_recoveries),
        "unsupported_recoveries": unsupported_recoveries,
        "access_junction_count": len(access_junctions),
        "prohibited_access_sbend_count": len(prohibited_access_sbends),
        "prohibited_access_sbends": prohibited_access_sbends,
        "clean": not unsupported_recoveries and not prohibited_access_sbends,
    }


def audit_gds(path: Path, expected_route_cells: int) -> dict[str, Any]:
    layout = kdb.Layout()
    layout.read(str(path))
    top = layout.top_cell()
    if top is None:
        raise RuntimeError(f"GDS has no top cell: {path}")
    route_cells = sorted(
        instance.cell.name
        for instance in top.each_inst()
        if instance.cell.name.lower().endswith(("_route", "_short_sbend"))
    )
    short_sbend_cells = [
        name for name in route_cells if name.lower().endswith("_short_sbend")
    ]
    return {
        "gds": str(path.resolve()),
        "top_cell": top.name,
        "gds_bytes": path.stat().st_size,
        "route_cell_count": len(route_cells),
        "expected_route_cell_count": int(expected_route_cells),
        "route_cell_count_matches": len(route_cells) == int(expected_route_cells),
        "short_sbend_cell_count": len(short_sbend_cells),
        "short_sbend_cells": short_sbend_cells,
        "clean": bool(path.stat().st_size)
        and len(route_cells) == int(expected_route_cells),
    }


def audit_pdk_angles(
    route_result: Path,
    geometry_report: Path,
    gds: Path,
) -> dict[str, Any]:
    route_audit = audit_route_geometry(route_result)
    renderer_audit = audit_renderer_geometry(geometry_report)
    gds_audit = audit_gds(
        gds,
        route_audit["routed_path_count"] + route_audit["short_sbend_count"],
    )
    clean = bool(
        route_audit["clean"] and renderer_audit["clean"] and gds_audit["clean"]
    )
    return {
        "schema_version": 1,
        "contract": {
            "allowed_centerline_headings_deg": [
                0.0,
                45.0,
                90.0,
                135.0,
                180.0,
                225.0,
                270.0,
                315.0,
            ],
            "allowed_turns_deg": [0.0, 45.0, 90.0],
            "free_angle_sbends_allowed": False,
            "access_lateral_tolerance_um": ACCESS_LATERAL_TOLERANCE_UM,
            "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
        },
        "route_result_audit": route_audit,
        "renderer_audit": renderer_audit,
        "gds_audit": gds_audit,
        "clean": clean,
    }
