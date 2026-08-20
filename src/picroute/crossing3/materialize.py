from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .access_contract import effective_hard_access, port_connection_feasibility
from .benchmark_source import sha256_file
from .case_io import absolute_ports, load_case, placement_tuple
from .model import PlacedCrossing, Prediction
from .pcell_geometry import crossing_views
from .source_legalizer import legalize_source_unplaced_instances


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def normalize_case(
    source: Path,
    output_dir: Path,
    source_placement_config: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Legalize allowed source placements, then freeze a private run copy.

    The authoritative YAML is never edited.  The original placement tuple is
    recorded before source ``UNPLACED`` instances may be moved by the generic
    PDK access legalizer. Source ``FIXED`` instances remain immutable.
    """

    source = source.resolve(strict=True)
    case = load_case(source)
    normalized = copy.deepcopy(case)
    originals: dict[str, Any] = {}
    status_counts: dict[str, int] = defaultdict(int)
    for name, node in (normalized.get("instances") or {}).items():
        lower_left, orientation = placement_tuple(node)
        placement = node["settings"]["placement"]
        status = str(placement[0]).upper()
        status_counts[status] += 1
        originals[str(name)] = {
            "status": status,
            "lower_left_um": lower_left,
            "orientation": orientation,
            "component": str(node.get("component", "")),
            "macro_type": str((node.get("settings") or {}).get("macro_type", "")),
        }
    legalization = legalize_source_unplaced_instances(
        normalized, dict(source_placement_config or {})
    )
    for node in (normalized.get("instances") or {}).values():
        node["settings"]["placement"][0] = "FIXED"
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "normalized_case.yml"
    write_yaml(output, normalized)
    manifest = {
        "schema_version": 1,
        "source_case": str(source),
        "source_sha256": sha256_file(source),
        "normalized_case": str(output.resolve()),
        "instance_count": len(originals),
        "source_status_counts": dict(sorted(status_counts.items())),
        "normalized_fixed_count": len(originals),
        "original_instances": originals,
        "source_placement_legalization": legalization,
    }
    write_json(output_dir / "source_case_manifest.json", manifest)
    return output, manifest


def _crossing_macro(
    result_library: dict[str, Any],
    manifest: dict[str, Any],
    rotation: float,
) -> tuple[str, dict[str, Any]]:
    views = crossing_views(manifest)
    view = views[float(rotation)]
    macro_name = "picroute_crossing_0" if float(rotation) == 0.0 else "picroute_crossing_m45"
    if macro_name not in result_library:
        pins = {}
        for name, port in view["ports"].items():
            pins[str(name)] = {
                "pin_offset_x": float(port["local_center_um"][0]),
                "pin_offset_y": float(port["local_center_um"][1]),
                "pin_width": float(port["width_um"]),
                "pin_orient": float(port["orientation_deg"]),
                "pin_layer": port.get("layer", "1"),
            }
        result_library[macro_name] = {
            "property": None,
            "iloss": 0.0,
            "type": "CORE",
            "origin": [0.0, 0.0],
            "size": [float(value) for value in view["size_um"]],
            "site": "core",
            "pins": pins,
            "crossing_production_rotation_deg": float(rotation),
            "crossing_production_real_pcell": manifest["generator"],
            "crossing_production_gds_sha256": manifest["gds_sha256"],
        }
    return macro_name, view


def materialize_case(
    normalized_case_path: Path,
    placed: list[PlacedCrossing],
    prediction: Prediction,
    directions: dict[str, dict[str, list[str]]],
    manifest: dict[str, Any],
    output_dir: Path,
    minimum_access_um: float = 10.0,
    minimum_radius_um: float = 5.0,
    direct_threshold_um: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Insert fixed real crossing PCells and split their two parent nets."""

    case = load_case(normalized_case_path)
    result = copy.deepcopy(case)
    result_instances = result.setdefault("instances", {})
    result_library = result.setdefault("library", {})
    original_nets = result.get("nets") or {}
    by_id = {item.event.event_id: item for item in placed}
    if set(by_id) != {event.event_id for event in prediction.events}:
        raise ValueError("Placed crossing set differs from predicted topology")

    assignments: list[dict[str, Any]] = []
    for item in sorted(placed, key=lambda value: value.event.event_id):
        event = item.event
        if event.event_id in result_instances:
            raise ValueError(f"Crossing id collides with original instance: {event.event_id}")
        macro_name, view = _crossing_macro(result_library, manifest, item.rotation_deg)
        min_x, min_y, _max_x, _max_y = [float(v) for v in view["bbox_centered_um"]]
        center = [float(item.center_um[0]), float(item.center_um[1])]
        result_instances[event.event_id] = {
            "component": macro_name,
            "settings": {
                "macro_type": macro_name,
                "placement": [
                    "FIXED",
                    [center[0] + min_x, center[1] + min_y],
                    "N",
                    [0, 0, 0, 0],
                ],
            },
        }
        assignments.append(
            {
                "id": event.event_id,
                "center_um": center,
                "ideal_center_um": list(event.ideal_center_um),
                "rotation_deg": float(item.rotation_deg),
                "net_a": event.net_a,
                "net_b": event.net_b,
                "net_a_ports": list(event.net_a_ports),
                "net_b_ports": list(event.net_b_ports),
                "order_on_net_a": int(event.order_on_net_a),
                "order_on_net_b": int(event.order_on_net_b),
                "topology_component": event.topology_component,
                "topology_stage": event.topology_stage,
                "evidence": list(event.evidence),
                "legal": bool(item.legal),
                "displacement_um": float(item.displacement_um),
            }
        )

    rewritten: dict[str, list[str]] = {}
    segment_parent: dict[str, str] = {}
    order_audit: dict[str, Any] = {}
    for raw_name, endpoints in original_nets.items():
        net_name = str(raw_name)
        source, target = str(endpoints[0]), str(endpoints[1])
        order = list(prediction.net_orders.get(net_name, []))
        if not order:
            rewritten[net_name] = [source, target]
            segment_parent[net_name] = net_name
            continue
        if set(order) != {
            event.event_id
            for event in prediction.events
            if net_name in (event.net_a, event.net_b)
        }:
            raise ValueError(f"Incomplete topology order for {net_name}")
        current = source
        selected = []
        for index, event_id in enumerate(order):
            ports = directions.get(net_name, {}).get(event_id)
            if not ports or len(ports) != 2:
                raise ValueError(f"Missing direction state for {net_name}/{event_id}")
            entry, exit_name = str(ports[0]), str(ports[1])
            segment = f"{net_name}__s{index:03d}"
            rewritten[segment] = [current, f"{event_id},{entry}"]
            segment_parent[segment] = net_name
            current = f"{event_id},{exit_name}"
            selected.append([entry, exit_name])
        tail = f"{net_name}__s{len(order):03d}"
        rewritten[tail] = [current, target]
        segment_parent[tail] = net_name
        order_audit[net_name] = {
            "method": "hctp_explicit_topology_order",
            "crossing_order": order,
            "selected_entry_exit_ports": selected,
        }
    result["nets"] = rewritten
    direct_threshold = float(
        direct_threshold_um
        if direct_threshold_um is not None
        else 2.0 * (minimum_access_um + minimum_radius_um)
    )
    materialized_ports = absolute_ports(result)
    segment_access_audit: dict[str, dict[str, Any]] = {}
    for segment_name, endpoints in rewritten.items():
        first_name, second_name = str(endpoints[0]), str(endpoints[1])
        if first_name not in materialized_ports or second_name not in materialized_ports:
            raise ValueError(
                f"Materialized segment {segment_name} references missing ports: "
                f"{first_name}, {second_name}"
            )
        first = materialized_ports[first_name]
        second = materialized_ports[second_name]
        feasibility = port_connection_feasibility(
            (first.x, first.y),
            first.orientation,
            (second.x, second.y),
            second.orientation,
            minimum_access_um,
            direct_threshold,
            minimum_radius_um,
        )
        segment_access_audit[segment_name] = {
            "endpoints": [first_name, second_name],
            **feasibility,
            "effective_hard_infeasible": effective_hard_access(feasibility),
        }
    hard_invalid_segments = sorted(
        segment_name
        for segment_name, audit in segment_access_audit.items()
        if bool(audit["effective_hard_infeasible"])
    )
    result["crossing_production"] = {
        "schema_version": 3,
        "mode": "crossing3_hctp",
        "strict_preplaced_crossings": True,
        "crossing_count": len(assignments),
        "crossing_manifest_sha256": manifest["gds_sha256"],
        "segment_parent": segment_parent,
        "net_crossing_order_audit": order_audit,
        "parity_contract": prediction.parity_contract,
        "assignments": assignments,
        "segment_access_gate": {
            "hard_invalid_count": len(hard_invalid_segments),
            "hard_invalid_segments": hard_invalid_segments,
        },
    }
    assignment_document = {
        "schema_version": 3,
        "mode": "crossing3_hctp",
        "crossing_count": len(assignments),
        "route_net_count": len(rewritten),
        "segment_parent": segment_parent,
        "segment_endpoints": rewritten,
        "net_crossing_order_audit": order_audit,
        "parity_contract": prediction.parity_contract,
        "crossings": assignments,
        "segment_access_audit": segment_access_audit,
        "hard_invalid_segments": hard_invalid_segments,
        "pre_route_access_gate_passed": not hard_invalid_segments,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    routing_case = output_dir / "routing_case.yml"
    write_yaml(routing_case, result)
    write_json(output_dir / "crossing_assignment.json", assignment_document)
    return routing_case, assignment_document
