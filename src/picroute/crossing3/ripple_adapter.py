from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .benchmark_source import sha256_file
from .model import PlacedCrossing, Prediction


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def place_with_ripple(
    normalized_case_path: Path,
    prediction: Prediction,
    crossing_manifest_path: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[PlacedCrossing], dict[str, Any]]:
    """Run the generic cold-start Ripple placer and adapt its finite states."""

    paths = config["paths"]
    placement_config = config["placement"]
    script = Path(paths["ripple_placer"]).resolve(strict=True)
    ripple_config = Path(paths["ripple_placer_config"]).resolve(strict=True)
    window_radius = float(placement_config.get("ripple_window_radius_um", 180.0))
    output_dir.mkdir(parents=True, exist_ok=False)
    estimate_path = output_dir / "crossing_estimate.json"
    estimate = {
        "schema_version": 2,
        "mode": "crossing3_cold_start_adapter",
        "input_case": str(normalized_case_path.resolve()),
        "rule": "hctp_topology_with_generic_ripple_physical_placement",
        "crossing_count": len(prediction.events),
        "crossing_topology": {
            "schema_version": 1,
            "contract": "immutable_source_to_target_crossing_order",
            "net_crossing_order": prediction.net_orders,
        },
        "crossings": [
            {
                "id": event.event_id,
                "net_a": event.net_a,
                "net_b": event.net_b,
                "confidence": event.confidence,
                "evidence": list(event.evidence),
                "ideal_center_um": list(event.ideal_center_um),
                "placement_window_um": [
                    event.ideal_center_um[0] - window_radius,
                    event.ideal_center_um[1] - window_radius,
                    event.ideal_center_um[0] + window_radius,
                    event.ideal_center_um[1] + window_radius,
                ],
                "preferred_rotation_deg": event.preferred_rotation_deg,
                "calibrated_orientation_candidate_index": None,
                "order_on_net_a": event.order_on_net_a,
                "order_on_net_b": event.order_on_net_b,
            }
            for event in prediction.events
        ],
    }
    _write_json(estimate_path, estimate)
    command = [
        sys.executable,
        str(script),
        str(normalized_case_path.resolve()),
        str(estimate_path.resolve()),
        str(crossing_manifest_path.resolve(strict=True)),
        str((output_dir / "native_ripple").resolve()),
        "--config",
        str(ripple_config),
    ]
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=script.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    (output_dir / "ripple_placer.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"Generic Ripple placer failed with exit code {process.returncode}; "
            f"see {output_dir / 'ripple_placer.log'}"
        )

    placement_path = output_dir / "native_ripple" / "ripple_placement.json"
    report_path = output_dir / "native_ripple" / "ripple_report.json"
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    native_report = json.loads(report_path.read_text(encoding="utf-8"))
    by_id = {event.event_id: event for event in prediction.events}
    records = {str(item["id"]): item for item in placement["crossings"]}
    if set(records) != set(by_id):
        raise RuntimeError("Ripple crossing set differs from the HCTP prediction")

    placed = []
    for event_id in sorted(by_id):
        event = by_id[event_id]
        record = records[event_id]
        committed = next(
            (
                item
                for item in record.get("placement_orientation_candidates", [])
                if bool(item.get("committed"))
            ),
            None,
        )
        if committed is None:
            candidate_index = int(record["committed_candidate_index"])
            committed = next(
                item
                for item in record.get("placement_orientation_candidates", [])
                if int(item["candidate_index"]) == candidate_index
            )
        event.net_a_ports = tuple(str(value) for value in committed["net_a_ports"])
        event.net_b_ports = tuple(str(value) for value in committed["net_b_ports"])
        center = (float(record["center_um"][0]), float(record["center_um"][1]))
        placed.append(
            PlacedCrossing(
                event=event,
                center_um=center,
                rotation_deg=float(record["rotation"]),
                legal=True,
                displacement_um=math.dist(center, event.ideal_center_um),
            )
        )

    report = {
        "schema_version": 2,
        "algorithm": "hctp_topology_plus_generic_crossing_ripple_v2",
        "crossing_count": len(placed),
        "component_count": len(prediction.braid_components),
        "original_instances_moved": 0,
        "historical_placement_seed_used": False,
        "case_specific_dispatch_used": False,
        "command": command,
        "elapsed_s": elapsed,
        "tool": {
            "path": str(script),
            "sha256": sha256_file(script),
            "config_path": str(ripple_config),
            "config_sha256": sha256_file(ripple_config),
        },
        "estimate": str(estimate_path.resolve()),
        "native_placement": str(placement_path.resolve()),
        "native_report": native_report,
    }
    _write_json(output_dir / "adapter_report.json", report)
    return placed, report
