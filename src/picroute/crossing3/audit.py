from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .benchmark_source import sha256_file
from .case_io import fixed_obstacles, load_case, placement_tuple
from .model import PlacedCrossing, Prediction
from .pcell_geometry import crossing_views
from .routing_lattice import center_is_track_compatible


def _overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return (
        min(first[2], second[2]) - max(first[0], second[0]) > 1e-9
        and min(first[3], second[3]) - max(first[1], second[1]) > 1e-9
    )


def _inflate(box: tuple[float, float, float, float], amount: float) -> tuple[float, float, float, float]:
    return box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount


def audit_frontend(
    source_path: Path,
    expected_source_sha256: str,
    normalized_path: Path,
    routing_case_path: Path,
    prediction: Prediction,
    placed: list[PlacedCrossing],
    manifest: dict[str, Any],
    die: tuple[float, float, float, float],
) -> dict[str, Any]:
    source = load_case(source_path)
    normalized = load_case(normalized_path)
    routing = load_case(routing_case_path)
    source_manifest = json.loads(
        (normalized_path.parent / "source_case_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    legalization = source_manifest.get("source_placement_legalization") or {}
    legalized_positions = {
        str(item["instance"]): [
            [float(value) for value in item["legalized_lower_left_um"]],
            str(source_manifest["original_instances"][str(item["instance"])]["orientation"]),
        ]
        for item in legalization.get("movements") or []
    }
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    checks["authoritative_source_hash_unchanged"] = (
        sha256_file(source_path) == expected_source_sha256
    )
    source_instances = source.get("instances") or {}
    normalized_instances = normalized.get("instances") or {}
    routing_instances = routing.get("instances") or {}
    original_differences = []
    fixed_original_differences = []
    legalization_differences = []
    for name, source_node in source_instances.items():
        if name not in normalized_instances or name not in routing_instances:
            original_differences.append({"instance": name, "reason": "missing"})
            continue
        source_xy, source_orientation = placement_tuple(source_node)
        source_status = str(source_node["settings"]["placement"][0]).upper()
        expected_xy, expected_orientation = (
            legalized_positions.get(name, [source_xy, source_orientation])
        )
        for label, candidate in (
            ("normalized", normalized_instances[name]),
            ("routing", routing_instances[name]),
        ):
            candidate_xy, candidate_orientation = placement_tuple(candidate)
            if candidate_xy != expected_xy or candidate_orientation != expected_orientation:
                difference = (
                    {
                        "instance": name,
                        "copy": label,
                        "source_status": source_status,
                        "source": [source_xy, source_orientation],
                        "expected": [expected_xy, expected_orientation],
                        "actual": [candidate_xy, candidate_orientation],
                    }
                )
                original_differences.append(difference)
                if source_status == "FIXED":
                    fixed_original_differences.append(difference)
                else:
                    legalization_differences.append(difference)
            if str(candidate["settings"]["placement"][0]).upper() != "FIXED":
                difference = {"instance": name, "copy": label, "reason": "not_fixed"}
                original_differences.append(difference)
                legalization_differences.append(difference)
    checks["original_instances_immutable"] = not original_differences
    checks["fixed_original_instances_immutable"] = not fixed_original_differences
    checks["source_unplaced_instances_legally_frozen"] = not legalization_differences
    checks["only_crossings_added"] = (
        set(routing_instances) - set(source_instances)
        == {item.event.event_id for item in placed}
    )
    details["original_instance_differences"] = original_differences
    details["fixed_original_instance_differences"] = fixed_original_differences
    details["source_placement_legalization_differences"] = legalization_differences
    details["source_placement_legalization"] = legalization

    views = crossing_views(manifest)
    halo = float(manifest.get("halo_um", 0.0))
    fixed_boxes = {item.name: item.bbox for item in fixed_obstacles(normalized)}
    crossing_boxes: dict[str, tuple[float, float, float, float]] = {}
    geometry_violations = []
    by_id = {item.event.event_id: item for item in placed}
    for item in placed:
        local = views[float(item.rotation_deg)]["bbox_centered_um"]
        x, y = item.center_um
        box = (x + local[0], y + local[1], x + local[2], y + local[3])
        crossing_boxes[item.event.event_id] = box
        if box[0] < die[0] or box[1] < die[1] or box[2] > die[2] or box[3] > die[3]:
            geometry_violations.append(
                {"crossing": item.event.event_id, "reason": "outside_die", "bbox": list(box)}
            )
        for fixed_name, fixed_box in fixed_boxes.items():
            if _overlap(_inflate(box, halo), fixed_box):
                geometry_violations.append(
                    {
                        "crossing": item.event.event_id,
                        "fixed": fixed_name,
                        "reason": "crossing_halo_fixed_overlap",
                    }
                )
    crossing_ids = sorted(crossing_boxes)
    for first_index, first_id in enumerate(crossing_ids):
        first_nets = {by_id[first_id].event.net_a, by_id[first_id].event.net_b}
        for second_id in crossing_ids[first_index + 1 :]:
            shared = bool(
                first_nets
                & {by_id[second_id].event.net_a, by_id[second_id].event.net_b}
            )
            first_box = crossing_boxes[first_id]
            second_box = crossing_boxes[second_id]
            invalid = (
                _overlap(first_box, second_box)
                if shared
                else _overlap(_inflate(first_box, halo), _inflate(second_box, halo))
            )
            if invalid:
                geometry_violations.append(
                    {
                        "pair": [first_id, second_id],
                        "reason": "crossing_body_overlap" if shared else "crossing_halo_overlap",
                    }
                )
    checks["crossing_geometry_legal"] = not geometry_violations
    details["geometry_violations"] = geometry_violations
    grid = float((routing.get("settings") or {}).get("grid_resolution", 2.0))
    lattice_violations = sorted(
        item.event.event_id
        for item in placed
        if not center_is_track_compatible(item.center_um, die, grid)
    )
    # This is a construction invariant for the initial placement, not an
    # immutable sign-off rule: a feedback move may deliberately align one
    # crossing arm to an off-track fixed benchmark port. Final acceptance is
    # governed by the physical access/GDS angle audit.
    details["routing_lattice_initially_compatible"] = not lattice_violations
    details["routing_lattice_violations"] = lattice_violations

    event_ids = [event.event_id for event in prediction.events]
    checks["prediction_placement_bijection"] = (
        len(event_ids) == len(set(event_ids)) == len(placed)
        and set(event_ids) == set(by_id)
    )
    topology_violations = []
    for net_name, order in prediction.net_orders.items():
        ranks = []
        for event_id in order:
            event = by_id[event_id].event
            ranks.append(
                event.order_on_net_a if event.net_a == net_name else event.order_on_net_b
            )
        if ranks != list(range(len(order))):
            topology_violations.append(
                {"net": net_name, "order": order, "ranks": ranks}
            )
    checks["topology_order_total"] = not topology_violations
    details["topology_violations"] = topology_violations

    pair_counts = Counter(event.pair() for event in prediction.events)
    parity_violations = []
    for pair_key, contract in prediction.parity_contract.items():
        pair = tuple(contract["nets"])
        actual = pair_counts[pair]
        required = str(contract["required_parity"])
        okay = actual % 2 == (0 if required == "even" else 1)
        if not okay or actual != int(contract["predicted_count"]):
            parity_violations.append(
                {"pair": pair_key, "actual": actual, "contract": contract}
            )
    checks["pair_parity_contract"] = not parity_violations
    details["parity_violations"] = parity_violations

    crossing_incidence: dict[str, list[str]] = defaultdict(list)
    for endpoints in (routing.get("nets") or {}).values():
        for endpoint in endpoints:
            instance = str(endpoint).split(",", 1)[0]
            if instance in by_id:
                crossing_incidence[instance].append(str(endpoint).split(",", 1)[1])
    incidence_violations = {
        identifier: sorted(crossing_incidence.get(identifier, []))
        for identifier in by_id
        if sorted(crossing_incidence.get(identifier, [])) != ["o1", "o2", "o3", "o4"]
    }
    checks["each_crossing_uses_four_ports_once"] = not incidence_violations
    details["incidence_violations"] = incidence_violations

    checks["all_crossings_legal_flag"] = all(item.legal for item in placed)
    return {
        "schema_version": 1,
        "stage": "crossing3_frontend",
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
        "counts": {
            "original_instances": len(source_instances),
            "predicted_crossings": len(prediction.events),
            "placed_crossings": len(placed),
            "routing_segments": len(routing.get("nets") or {}),
        },
    }
