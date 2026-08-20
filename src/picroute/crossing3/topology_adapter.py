from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .benchmark_source import sha256_file
from .model import CrossingEvent, Prediction


def predict_with_production_estimator(
    normalized_case_path: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[Prediction, dict[str, Any], dict[str, Any]]:
    """Adapt the generic cold-start topology estimator into crossing3."""

    script = Path(config["paths"]["crossing_estimator"]).resolve(strict=True)
    estimator_config = Path(
        config["paths"]["crossing_estimator_config"]
    ).resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(script),
        str(normalized_case_path.resolve()),
        str(output_dir.resolve()),
        "--config",
        str(estimator_config),
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
    (output_dir / "estimator.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"Generic crossing estimator failed with exit code {process.returncode}; "
            f"see {output_dir / 'estimator.log'}"
        )
    estimate_path = output_dir / "crossing_estimate.json"
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    topology = estimate.get("crossing_topology") or {}
    net_orders = {
        str(name): [str(value) for value in values]
        for name, values in (topology.get("net_crossing_order") or {}).items()
    }
    component_by_event: dict[str, tuple[str, int]] = {}
    braid_components = list(topology.get("braid_components") or [])
    for component_index, component in enumerate(braid_components):
        ordered = [
            str(value)
            for value in (
                component.get("event_order")
                or component.get("events")
                or component.get("crossings")
                or []
            )
        ]
        for stage, event_id in enumerate(ordered):
            component_by_event[event_id] = (f"braid_{component_index:03d}", stage)

    events = []
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in estimate.get("crossings") or []:
        event_id = str(record["id"])
        net_a = str(record["net_a"])
        net_b = str(record["net_b"])
        pair_counts[tuple(sorted((net_a, net_b)))] += 1
        component, stage = component_by_event.get(event_id, (None, None))
        events.append(
            CrossingEvent(
                event_id=event_id,
                net_a=net_a,
                net_b=net_b,
                ideal_center_um=(
                    float(record["ideal_center_um"][0]),
                    float(record["ideal_center_um"][1]),
                ),
                evidence=[str(value) for value in record.get("evidence") or []],
                confidence=str(record.get("confidence", "high")),
                topology_component=component,
                topology_stage=stage,
                order_on_net_a=int(record["order_on_net_a"]),
                order_on_net_b=int(record["order_on_net_b"]),
                preferred_rotation_deg=float(
                    record.get("preferred_rotation_deg") or 0.0
                ),
            )
        )
    parity_contract = {
        f"{pair[0]}|{pair[1]}": {
            "nets": list(pair),
            "predicted_count": count,
            "required_parity": "odd" if count % 2 else "even",
            "parity_satisfied": True,
        }
        for pair, count in sorted(pair_counts.items())
    }
    prediction = Prediction(
        events=events,
        net_orders=net_orders,
        braid_components=braid_components,
        parity_contract=parity_contract,
        diagnostics=[],
    )
    motif_records = [
        event.event_id
        for event in events
        if "corridor_pressure_recrossing" in event.evidence
    ]
    motif_report = {
        "algorithm": "production_corridor_pressure_recrossing",
        "motif_count": len(motif_records),
        "event_ids": motif_records,
        "corridor_diagnostics": estimate.get("corridor_diagnostics") or [],
    }
    report = {
        "algorithm": "generic_production_crossing_estimator_v2",
        "command": command,
        "elapsed_s": elapsed,
        "estimate": str(estimate_path.resolve()),
        "tool": {
            "path": str(script),
            "sha256": sha256_file(script),
            "config_path": str(estimator_config),
            "config_sha256": sha256_file(estimator_config),
        },
    }
    return prediction, motif_report, report
