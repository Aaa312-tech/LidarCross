from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from picroute.crossing3.case_io import (  # noqa: E402
    die_area,
    fixed_obstacles,
    load_case,
    two_pin_nets,
)
from picroute.crossing3.direction_solver import solve_net_directions  # noqa: E402
from picroute.crossing3.materialize import materialize_case  # noqa: E402
from picroute.crossing3.model import (  # noqa: E402
    CrossingEvent,
    PlacedCrossing,
    Prediction,
)
from picroute.crossing3.placement_legal import placement_is_legal  # noqa: E402
from picroute.crossing3.planner import load_config  # noqa: E402


def _placed(document: dict) -> list[PlacedCrossing]:
    result = []
    for item in document["crossings"]:
        event = CrossingEvent(
            event_id=str(item["id"]),
            net_a=str(item["net_a"]),
            net_b=str(item["net_b"]),
            ideal_center_um=tuple(float(value) for value in item["ideal_center_um"]),
            evidence=[str(value) for value in item.get("evidence", [])],
            topology_component=item.get("topology_component"),
            topology_stage=item.get("topology_stage"),
            order_on_net_a=int(item["order_on_net_a"]),
            order_on_net_b=int(item["order_on_net_b"]),
            preferred_rotation_deg=float(item["rotation_deg"]),
            net_a_ports=tuple(str(value) for value in item["net_a_ports"]),
            net_b_ports=tuple(str(value) for value in item["net_b_ports"]),
        )
        result.append(
            PlacedCrossing(
                event=event,
                center_um=tuple(float(value) for value in item["center_um"]),
                rotation_deg=float(item["rotation_deg"]),
                legal=bool(item["legal"]),
                displacement_um=float(item["displacement_um"]),
            )
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-materialize one crossing candidate through the real access gate."
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--crossing", required=True)
    parser.add_argument("--center", nargs=2, type=float, required=True)
    parser.add_argument("--rotation", type=float, required=True)
    parser.add_argument("--swap-pairs", action="store_true")
    parser.add_argument("--recross-centers", nargs=4, type=float)
    parser.add_argument("--recross-rotations", nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument("--recross-swap", nargs=2, type=int, default=(0, 0))
    parser.add_argument(
        "--recross-order",
        choices=("base-r0-r1", "r0-base-r1", "r0-r1-base"),
        default="base-r0-r1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve(strict=True)
    normalized = case_dir / "01_input" / "normalized_case.yml"
    manifest = json.loads(
        (case_dir / "04_pcell" / "crossing_manifest.json").read_text("utf-8")
    )
    placed = _placed(
        json.loads(
            (case_dir / "05_placement" / "placed_crossings.json").read_text("utf-8")
        )
    )
    prediction_doc = json.loads(
        (case_dir / "03_topology" / "crossing_prediction.json").read_text("utf-8")
    )
    by_id = {item.event.event_id: item for item in placed}
    target = by_id[args.crossing]
    target.center_um = tuple(args.center)
    target.rotation_deg = float(args.rotation)
    target.displacement_um = math.dist(target.center_um, target.event.ideal_center_um)
    if args.swap_pairs:
        target.event.net_a_ports, target.event.net_b_ports = (
            target.event.net_b_ports,
            target.event.net_a_ports,
        )

    added: list[PlacedCrossing] = []
    if args.recross_centers:
        labels = ("r0", "r1")
        centers = (
            tuple(args.recross_centers[:2]),
            tuple(args.recross_centers[2:]),
        )
        for label, center, rotation, swap in zip(
            labels, centers, args.recross_rotations, args.recross_swap
        ):
            pair_a = target.event.net_a_ports
            pair_b = target.event.net_b_ports
            if bool(swap):
                pair_a, pair_b = pair_b, pair_a
            event = CrossingEvent(
                event_id=f"{args.crossing}__{label}",
                net_a=target.event.net_a,
                net_b=target.event.net_b,
                ideal_center_um=center,
                evidence=[
                    "pair_preserving_access_recrossing",
                    "odd_pair_parity",
                    "diagnostic_anchor",
                ],
                confidence="medium",
                preferred_rotation_deg=rotation,
                net_a_ports=pair_a,
                net_b_ports=pair_b,
            )
            added.append(
                PlacedCrossing(
                    event=event,
                    center_um=center,
                    rotation_deg=rotation,
                    legal=True,
                    displacement_um=0.0,
                )
            )
        placed.extend(added)

    case = load_case(normalized)
    nets = two_pin_nets(case)
    obstacles = fixed_obstacles(case)
    if not placement_is_legal(
        target,
        [item for item in placed if item is not target],
        obstacles,
        die_area(case),
        manifest,
    ):
        raise RuntimeError("Candidate violates the physical crossing placement contract")
    legal_existing = [item for item in placed if item not in added]
    for item in added:
        if not placement_is_legal(
            item,
            legal_existing,
            obstacles,
            die_area(case),
            manifest,
        ):
            raise RuntimeError(
                f"Added recrossing {item.event.event_id} violates the placement contract"
            )
        legal_existing.append(item)

    prediction = Prediction(
        events=[item.event for item in placed],
        net_orders={
            str(name): [str(value) for value in values]
            for name, values in prediction_doc["net_orders"].items()
        },
        braid_components=list(prediction_doc.get("braid_components", [])),
        parity_contract=dict(prediction_doc.get("parity_contract", {})),
        diagnostics=list(prediction_doc.get("diagnostics", [])),
    )
    if added:
        base_id = target.event.event_id
        recross_ids = [item.event.event_id for item in added]
        sequences = {
            "base-r0-r1": [base_id, *recross_ids],
            "r0-base-r1": [recross_ids[0], base_id, recross_ids[1]],
            "r0-r1-base": [*recross_ids, base_id],
        }
        order = sequences[args.recross_order]
        for name in (target.event.net_a, target.event.net_b):
            inherited = prediction.net_orders[name]
            replacement = []
            for event_id in inherited:
                replacement.extend(order if event_id == base_id else [event_id])
            prediction.net_orders[name] = replacement
        for name in (target.event.net_a, target.event.net_b):
            for rank, event_id in enumerate(prediction.net_orders[name]):
                event = next(item for item in prediction.events if item.event_id == event_id)
                if event.net_a == name:
                    event.order_on_net_a = rank
                else:
                    event.order_on_net_b = rank
        key = "|".join(sorted((target.event.net_a, target.event.net_b)))
        prediction.parity_contract[key] = {
            "nets": sorted((target.event.net_a, target.event.net_b)),
            "required_parity": "odd",
            "predicted_count": 3,
            "parity_satisfied": True,
        }
    config = load_config()
    technology = config["technology"]
    minimum_access = float(technology["minimum_access_um"])
    minimum_radius = float(technology["bend_radius_um"])
    direct_threshold = float(technology["short_direct_access_threshold_um"])
    directions, audit = solve_net_directions(
        placed,
        nets,
        manifest,
        minimum_access,
        minimum_radius,
        direct_threshold,
    )
    routing_case, assignment = materialize_case(
        normalized,
        placed,
        prediction,
        directions,
        manifest,
        args.output.resolve(),
        minimum_access,
        minimum_radius,
        direct_threshold,
    )
    result = {
        "routing_case": str(routing_case),
        "crossing": args.crossing,
        "center_um": list(target.center_um),
        "rotation_deg": target.rotation_deg,
        "swapped_pairs": bool(args.swap_pairs),
        "recrossings": [
            {
                "id": item.event.event_id,
                "center_um": list(item.center_um),
                "rotation_deg": item.rotation_deg,
            }
            for item in added
        ],
        "recross_order": args.recross_order,
        "selected_ports": {
            name: value[args.crossing]
            for name, value in directions.items()
            if args.crossing in value
        },
        "direction_audit": {
            name: value
            for name, value in audit.items()
            if args.crossing in value.get("crossing_order", [])
        },
        "hard_invalid_segments": assignment["hard_invalid_segments"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
