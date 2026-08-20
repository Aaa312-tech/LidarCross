from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from .audit import audit_frontend
from .backend import run_strict_backend
from .benchmark_source import (
    AUTHORITATIVE_BENCHMARK_ROOT,
    BenchmarkRef,
    discover_benchmarks,
    sha256_file,
)
from .capacity_motifs import (
    add_capacity_recrossing_motifs,
    suppress_overpacked_capacity_components,
)
from .case_io import die_area, fixed_obstacles, load_case, two_pin_nets
from .channel_guides import build_channel_guides
from .constraint_placer import place_crossings
from .direction_solver import solve_net_directions
from .dense_ladder_refiner import refine_dense_source_ladders
from .direction_chain_refiner import refine_direction_chains
from .finite_clearance_refiner import add_finite_clearance_crossings
from .materialize import materialize_case, normalize_case, write_json
from .pcell_geometry import extract_crossing_manifest
from .predictor import predict_crossings
from .ripple_adapter import place_with_ripple
from .routing_lattice import legalize_crossing_centers_to_routing_tracks
from .small_braid_refiner import refine_closed_three_net_braids
from .topology_adapter import predict_with_production_estimator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "crossing3.yml"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = path.resolve(strict=True)
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    configured = Path(value["paths"]["benchmark_root"]).resolve(strict=True)
    authoritative = AUTHORITATIVE_BENCHMARK_ROOT.resolve(strict=True)
    if configured != authoritative:
        raise ValueError(
            f"benchmark_root must remain authoritative: {authoritative}; got {configured}"
        )
    if not bool(value.get("immutable", {}).get("original_instances")):
        raise ValueError("crossing3 requires immutable.original_instances=true")
    if bool(value.get("source_placement", {}).get("enabled", False)) and not bool(
        value.get("immutable", {}).get("allow_source_unplaced_legalization", False)
    ):
        raise ValueError(
            "source_placement legalization requires "
            "immutable.allow_source_unplaced_legalization=true"
        )
    if not bool(value.get("immutable", {}).get("detailed_router")):
        raise ValueError("crossing3 requires immutable.detailed_router=true")
    return value


def _below(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Generated path escaped work root: {resolved}") from error
    return resolved


def _serialize(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _serialize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _source_inventory() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted(package.glob("*.py"))
    }


def _tool_inventory(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    result: dict[str, Any] = {}
    for key in (
        "frozen_router",
        "frozen_converter",
        "frozen_renderer",
        "frozen_continuity_audit",
        "pdk_angle_audit",
        "ripple_placer",
        "ripple_placer_config",
        "crossing_estimator",
        "crossing_estimator_config",
    ):
        path = Path(paths[key]).resolve(strict=True)
        result[key] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _make_run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"hctp-{timestamp}-{os.getpid()}"


def _prepare_run(
    config: dict[str, Any], config_path: Path, run_id: str | None, names: list[str]
) -> tuple[str, Path, dict[str, Any]]:
    identifier = run_id or _make_run_id()
    if not SAFE_ID.fullmatch(identifier):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and dash")
    root = Path(config["paths"]["work_root"]).resolve()
    run_dir = _below(root / identifier, root)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "run_id": identifier,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "RUNNING",
        "requested_cases": names,
        "project_root": str(PROJECT_ROOT),
        "work_directory": str(run_dir),
        "authoritative_benchmark_root": str(AUTHORITATIVE_BENCHMARK_ROOT.resolve()),
        "config": config,
        "config_path": str(config_path.resolve(strict=True)),
        "config_sha256": sha256_file(config_path.resolve(strict=True)),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "crossing3_source_sha256": _source_inventory(),
        "external_tools": _tool_inventory(config),
        "cases": {},
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return identifier, run_dir, manifest


def _plan_one(
    benchmark: BenchmarkRef,
    config: dict[str, Any],
    run_dir: Path,
    results_run_dir: Path,
    run_backend: bool,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    if not SAFE_ID.fullmatch(benchmark.name):
        raise ValueError(f"Unsafe benchmark name: {benchmark.name!r}")
    case_dir = _below(run_dir / benchmark.name, run_dir)
    case_dir.mkdir(parents=False, exist_ok=False)

    normalized_path, source_manifest = normalize_case(
        benchmark.path,
        case_dir / "01_input",
        {
            **dict(config.get("technology") or {}),
            **dict(config.get("source_placement") or {}),
        },
    )
    case = load_case(normalized_path)
    nets = two_pin_nets(case)
    obstacles = fixed_obstacles(case)
    die = die_area(case)
    if len(nets) != len(case.get("nets") or {}):
        raise ValueError(
            f"{benchmark.name}: expected only two-pin optical nets; "
            f"parsed {len(nets)} of {len(case.get('nets') or {})}"
        )

    technology = dict(config.get("technology") or {})
    guide = build_channel_guides(nets, obstacles, die, technology)
    write_json(case_dir / "02_guides" / "channel_guides.json", guide.to_dict())

    prediction_config = dict(config.get("prediction") or {})
    topology_adapter_report = None
    if str(prediction_config.get("algorithm", "")).startswith(
        "production_estimator"
    ):
        prediction, motif_report, topology_adapter_report = (
            predict_with_production_estimator(
                normalized_path,
                config,
                case_dir / "03_topology" / "production_estimator",
            )
        )
    else:
        prediction = predict_crossings(
            nets,
            endpoint_epsilon=float(
                prediction_config.get("endpoint_epsilon_um", 1e-6)
            ),
        )
        motif_report = add_capacity_recrossing_motifs(
            prediction,
            nets,
            prediction_config,
            bend_radius_um=float(technology.get("bend_radius_um", 5.0)),
            crossing_body_um=float(technology.get("crossing_body_um", 8.0)),
            minimum_access_um=float(technology.get("minimum_access_um", 10.0)),
            grid_um=float(technology.get("grid_um", 2.0)),
        )
    capacity_suppression_report = suppress_overpacked_capacity_components(
        prediction,
        nets,
        {**technology, **prediction_config},
    )
    motif_report["short_corridor_suppression"] = capacity_suppression_report
    manifest = extract_crossing_manifest(
        case_dir / "04_pcell", float(technology.get("crossing_halo_um", 4.5))
    )
    placement_config = {**technology, **dict(config.get("placement") or {})}
    events_per_net: dict[str, int] = {}
    for event in prediction.events:
        events_per_net[event.net_a] = events_per_net.get(event.net_a, 0) + 1
        events_per_net[event.net_b] = events_per_net.get(event.net_b, 0) + 1
    maximum_events_per_net = max(events_per_net.values(), default=0)
    configured_placer = str(placement_config.get("algorithm", ""))
    use_ripple = configured_placer.startswith("ripple_adapter") or (
        configured_placer.startswith("hybrid_access_ripple")
        and maximum_events_per_net
        >= int(placement_config.get("ripple_minimum_events_per_net", 3))
    )
    if use_ripple:
        placed, placement_report = place_with_ripple(
            normalized_path,
            prediction,
            case_dir / "04_pcell" / "crossing_manifest.json",
            config,
            case_dir / "05_ripple_adapter",
        )
        placed, small_braid_report = refine_closed_three_net_braids(
            prediction,
            placed,
            nets,
            obstacles,
            die,
            manifest,
            placement_config,
            guide.paths,
        )
        placement_report["small_braid_refinement"] = small_braid_report
    else:
        placed, placement_report = place_crossings(
            prediction,
            nets,
            obstacles,
            die,
            manifest,
            placement_config,
            guide.paths,
        )
    placed, finite_clearance_report = add_finite_clearance_crossings(
        prediction,
        placed,
        nets,
        obstacles,
        die,
        manifest,
        placement_config,
    )
    placed, routing_lattice_report = legalize_crossing_centers_to_routing_tracks(
        placed,
        obstacles,
        die,
        manifest,
        float(technology.get("grid_um", 2.0)),
        search_rings=int(placement_config.get("routing_track_search_rings", 4)),
    )
    placed, direction_chain_report = refine_direction_chains(
        placed,
        nets,
        obstacles,
        die,
        manifest,
        {**technology, **placement_config},
    )
    # Dense ladder refinement depends on a globally feasible crossing-port
    # direction assignment.  Running it before the direction-chain repair
    # made one initially infeasible braid abort ladder optimization for every
    # otherwise independent component.  Repair directions first, then move
    # the generated first-crossing columns, and finally re-legalize/recheck
    # because a technology-pitch shift changes both track and chain geometry.
    placed, dense_ladder_report = refine_dense_source_ladders(
        prediction,
        placed,
        nets,
        obstacles,
        die,
        manifest,
        {**prediction_config, **placement_config},
    )
    placed, post_ladder_lattice_report = (
        legalize_crossing_centers_to_routing_tracks(
            placed,
            obstacles,
            die,
            manifest,
            float(technology.get("grid_um", 2.0)),
            search_rings=int(
                placement_config.get("routing_track_search_rings", 4)
            ),
        )
    )
    placed, post_ladder_direction_report = refine_direction_chains(
        placed,
        nets,
        obstacles,
        die,
        manifest,
        {**technology, **placement_config},
    )
    placement_report["finite_clearance_refinement"] = finite_clearance_report
    placement_report["dense_ladder_refinement"] = dense_ladder_report
    placement_report["routing_lattice_legalization"] = routing_lattice_report
    placement_report["direction_chain_refinement"] = direction_chain_report
    placement_report["post_ladder_routing_lattice_legalization"] = (
        post_ladder_lattice_report
    )
    placement_report["post_ladder_direction_chain_refinement"] = (
        post_ladder_direction_report
    )
    prediction_doc = {
        "schema_version": 1,
        "algorithm": "hctp_terminal_parity_and_reduced_wiring_diagram",
        "crossing_count": len(prediction.events),
        "events": [_serialize(event) for event in prediction.events],
        "net_orders": prediction.net_orders,
        "braid_components": prediction.braid_components,
        "parity_contract": prediction.parity_contract,
        "diagnostics": prediction.diagnostics,
        "capacity_motifs": motif_report,
        "finite_clearance_refinement": finite_clearance_report,
        "topology_adapter": topology_adapter_report,
    }
    write_json(case_dir / "03_topology" / "crossing_prediction.json", prediction_doc)
    placement_report["solver_selection"] = {
        "configured": configured_placer,
        "selected": "generic_ripple" if use_ripple else "access_component_milp",
        "maximum_events_per_net": maximum_events_per_net,
        "ripple_minimum_events_per_net": int(
            placement_config.get("ripple_minimum_events_per_net", 3)
        ),
    }
    placed_doc = {
        "schema_version": 1,
        "algorithm": placement_report["algorithm"],
        "crossing_count": len(placed),
        "crossings": [
            {
                "id": item.event.event_id,
                "net_a": item.event.net_a,
                "net_b": item.event.net_b,
                "center_um": list(item.center_um),
                "ideal_center_um": list(item.event.ideal_center_um),
                "rotation_deg": item.rotation_deg,
                "net_a_ports": list(item.event.net_a_ports),
                "net_b_ports": list(item.event.net_b_ports),
                "order_on_net_a": item.event.order_on_net_a,
                "order_on_net_b": item.event.order_on_net_b,
                "topology_component": item.event.topology_component,
                "topology_stage": item.event.topology_stage,
                "evidence": item.event.evidence,
                "legal": item.legal,
                "displacement_um": item.displacement_um,
            }
            for item in placed
        ],
    }
    write_json(case_dir / "05_placement" / "placed_crossings.json", placed_doc)
    write_json(case_dir / "05_placement" / "placement_report.json", placement_report)

    directions, direction_audit = solve_net_directions(
        placed,
        nets,
        manifest,
        float(technology.get("minimum_access_um", 10.0)),
        float(technology.get("bend_radius_um", 5.0)),
        float(
            technology.get(
                "short_direct_access_threshold_um",
                2.0
                * (
                    float(technology.get("minimum_access_um", 10.0))
                    + float(technology.get("bend_radius_um", 5.0))
                ),
            )
        ),
    )
    write_json(
        case_dir / "06_directions" / "direction_assignment.json",
        {"schema_version": 1, "assignments": directions, "audit": direction_audit},
    )
    routing_case, assignment = materialize_case(
        normalized_path,
        placed,
        prediction,
        directions,
        manifest,
        case_dir / "07_materialized",
        float(technology.get("minimum_access_um", 10.0)),
        float(technology.get("bend_radius_um", 5.0)),
        float(
            technology.get(
                "short_direct_access_threshold_um",
                2.0
                * (
                    float(technology.get("minimum_access_um", 10.0))
                    + float(technology.get("bend_radius_um", 5.0))
                ),
            )
        ),
    )
    if assignment["hard_invalid_segments"]:
        preview = ", ".join(assignment["hard_invalid_segments"][:8])
        raise RuntimeError(
            f"{benchmark.name}: pre-route access gate rejected "
            f"{len(assignment['hard_invalid_segments'])} segment(s): {preview}"
        )
    audit = audit_frontend(
        benchmark.path,
        benchmark.sha256,
        normalized_path,
        routing_case,
        prediction,
        placed,
        manifest,
        die,
    )
    write_json(case_dir / "08_audit" / "frontend_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"{benchmark.name}: crossing frontend audit failed")
    result = {
        "status": "FRONTEND_PASS",
        "source": {
            "path": source_manifest["source_case"],
            "sha256": source_manifest["source_sha256"],
            "instance_count": source_manifest["instance_count"],
            "manifest": str(
                (case_dir / "01_input" / "source_case_manifest.json").resolve()
            ),
        },
        "net_count": len(nets),
        "crossing_count": len(placed),
        "route_net_count": assignment["route_net_count"],
        "guide_failed_net_count": len(guide.failed_nets),
        "guide_maximum_usage": guide.maximum_usage,
        "braid_component_count": len(prediction.braid_components),
        "capacity_motif_count": max(
            0,
            int(motif_report.get("motif_count", 0))
            - int(capacity_suppression_report.get("removed_event_count", 0)),
        ),
        "capacity_motif_proposed_count": int(motif_report.get("motif_count", 0)),
        "capacity_motif_removed_count": int(
            capacity_suppression_report.get("removed_event_count", 0)
        ),
        "routing_case": str(routing_case.resolve()),
        "frontend_audit": str((case_dir / "08_audit" / "frontend_audit.json").resolve()),
    }
    if run_backend:
        backend = run_strict_backend(
            benchmark,
            normalized_path,
            prediction,
            placed,
            nets,
            manifest,
            config,
            case_dir,
            results_run_dir / benchmark.name,
        )
        result["backend"] = backend
        result["status"] = "ACCEPTED" if backend["accepted"] else "STRICT_BACKEND_FAIL"
    result["case_wall_time_s"] = time.perf_counter() - case_started
    return result


def plan_benchmarks(
    names: Iterable[str] | None = None,
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_id: str | None = None,
    run_backend: bool = False,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    config = load_config(config_path)
    catalog = discover_benchmarks()
    requested = sorted(set(names or catalog))
    unknown = sorted(set(requested) - set(catalog))
    if unknown:
        raise KeyError(
            f"Unknown benchmark(s): {', '.join(unknown)}; available: {', '.join(sorted(catalog))}"
        )
    identifier, run_dir, manifest = _prepare_run(
        config, config_path, run_id, requested
    )
    results_root = Path(config["paths"]["results_root"]).resolve()
    results_run_dir = _below(results_root / identifier, results_root)
    manifest["backend_requested"] = bool(run_backend)
    failures = []
    for name in requested:
        try:
            manifest["cases"][name] = _plan_one(
                catalog[name], config, run_dir, results_run_dir, run_backend
            )
            if run_backend and manifest["cases"][name]["status"] != "ACCEPTED":
                failures.append(name)
        except Exception as error:
            manifest["cases"][name] = {
                "status": "BACKEND_ERROR" if run_backend else "FRONTEND_FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(name)
        write_json(run_dir / "run_manifest.json", manifest)
    if failures:
        manifest["status"] = "STRICT_BACKEND_FAIL" if run_backend else "FRONTEND_FAIL"
    else:
        manifest["status"] = "ACCEPTED" if run_backend else "FRONTEND_PASS"
    manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["run_wall_time_s"] = time.perf_counter() - run_started
    manifest["failed_cases"] = failures
    write_json(run_dir / "run_manifest.json", manifest)
    return {
        "run_id": identifier,
        "run_directory": str(run_dir),
        "status": manifest["status"],
        "failed_cases": failures,
        "cases": manifest["cases"],
    }


def plan_case(
    name: str,
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_id: str | None = None,
    run_backend: bool = False,
) -> dict[str, Any]:
    return plan_benchmarks(
        [name],
        config_path=config_path,
        run_id=run_id,
        run_backend=run_backend,
    )
