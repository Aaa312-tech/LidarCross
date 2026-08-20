from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from .case_io import NetGeometry
from .model import CrossingEvent, Prediction
from .predictor import _assign_net_orders, _orientation_assignment


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    if length <= 1e-12:
        return 1.0, 0.0
    return vector[0] / length, vector[1] / length


def _dot(point: tuple[float, float], axis: tuple[float, float]) -> float:
    return point[0] * axis[0] + point[1] * axis[1]


def suppress_overpacked_capacity_components(
    prediction: Prediction,
    nets: dict[str, NetGeometry],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Remove optional even recrossings that cannot fit in a short corridor.

    Capacity motifs are optional, parity-preserving routing aids.  A connected
    motif component is counterproductive when one of its parent nets is
    shorter than the technology pitch required by all of that parent's motif
    crossings.  Removing the whole connected component avoids leaving a
    half-motif behind and preserves every pair's even parity.
    """

    if not bool(config.get("suppress_overpacked_capacity_components", True)):
        return {
            "enabled": False,
            "algorithm": "technology_pitch_capacity_component_suppression_v1",
            "removed_event_ids": [],
            "components": [],
        }

    def is_capacity(event: CrossingEvent) -> bool:
        evidence = set(event.evidence)
        return bool(
            evidence
            & {
                "capacity_recrossing",
                "corridor_pressure_recrossing",
            }
        )

    capacity_events = [event for event in prediction.events if is_capacity(event)]
    if not capacity_events:
        return {
            "enabled": True,
            "algorithm": "technology_pitch_capacity_component_suppression_v1",
            "technology_pitch_um": (
                math.sqrt(2.0) * float(config.get("crossing_body_um", 8.0))
                + 2.0 * float(config.get("minimum_access_um", 10.0))
                + float(config.get("waveguide_spacing_um", 1.0))
            ),
            "removed_event_ids": [],
            "components": [],
        }

    adjacency: dict[str, set[str]] = defaultdict(set)
    events_by_pair: dict[tuple[str, str], list[CrossingEvent]] = defaultdict(list)
    all_events_by_pair: dict[tuple[str, str], list[CrossingEvent]] = defaultdict(list)
    for event in prediction.events:
        all_events_by_pair[event.pair()].append(event)
    for event in capacity_events:
        adjacency[event.net_a].add(event.net_b)
        adjacency[event.net_b].add(event.net_a)
        events_by_pair[event.pair()].append(event)

    components: list[set[str]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        members = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in sorted(adjacency[current]):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    members.add(neighbour)
                    stack.append(neighbour)
        components.append(members)

    pitch = (
        math.sqrt(2.0) * float(config.get("crossing_body_um", 8.0))
        + 2.0 * float(config.get("minimum_access_um", 10.0))
        + float(config.get("waveguide_spacing_um", 1.0))
    )
    removed_ids: set[str] = set()
    audits = []
    for members in sorted(components, key=lambda values: tuple(sorted(values))):
        component_events = [
            event
            for event in capacity_events
            if event.net_a in members and event.net_b in members
        ]
        component_pairs = {event.pair() for event in component_events}
        pair_is_optional_even = all(
            len(events_by_pair[pair]) % 2 == 0
            and len(events_by_pair[pair]) == len(all_events_by_pair[pair])
            for pair in component_pairs
        )
        counts = {
            name: sum(
                name in (event.net_a, event.net_b) for event in component_events
            )
            for name in sorted(members)
        }
        required = {name: counts[name] * pitch for name in sorted(members)}
        overpacked = [
            name
            for name in sorted(members)
            if name in nets and required[name] > nets[name].length + 1e-9
        ]
        endpoint_clearance = (
            float(config.get("minimum_access_um", 10.0))
            + 0.5 * float(config.get("crossing_body_um", 8.0))
        )
        pinched_endpoints = []
        for event in component_events:
            for name in (event.net_a, event.net_b):
                if name not in nets:
                    continue
                net = nets[name]
                for endpoint_name, point in (
                    ("source", (net.source.x, net.source.y)),
                    ("target", (net.target.x, net.target.y)),
                ):
                    distance = math.dist(point, event.ideal_center_um)
                    if distance + 1e-9 < endpoint_clearance:
                        pinched_endpoints.append(
                            {
                                "event_id": event.event_id,
                                "net": name,
                                "endpoint": endpoint_name,
                                "distance_um": distance,
                            }
                        )
        suppressed = bool(
            pair_is_optional_even and (overpacked or pinched_endpoints)
        )
        if suppressed:
            removed_ids.update(event.event_id for event in component_events)
        audits.append(
            {
                "nets": sorted(members),
                "event_ids": sorted(event.event_id for event in component_events),
                "event_count_by_net": counts,
                "required_corridor_um_by_net": required,
                "available_chord_um_by_net": {
                    name: nets[name].length
                    for name in sorted(members)
                    if name in nets
                },
                "optional_even_pairs": pair_is_optional_even,
                "overpacked_nets": overpacked,
                "minimum_endpoint_clearance_um": endpoint_clearance,
                "pinched_endpoints": pinched_endpoints,
                "suppressed": suppressed,
            }
        )

    if removed_ids:
        prediction.events = [
            event for event in prediction.events if event.event_id not in removed_ids
        ]
        prediction.net_orders = {
            name: [event_id for event_id in order if event_id not in removed_ids]
            for name, order in prediction.net_orders.items()
        }
        ranks = {
            name: {event_id: rank for rank, event_id in enumerate(order)}
            for name, order in prediction.net_orders.items()
        }
        for event in prediction.events:
            event.order_on_net_a = ranks[event.net_a][event.event_id]
            event.order_on_net_b = ranks[event.net_b][event.event_id]

        retained_components = []
        for component in prediction.braid_components:
            identifiers = {
                str(value)
                for key in ("crossing_order", "event_order", "events", "crossings")
                for value in (component.get(key) or [])
            }
            if identifiers & removed_ids:
                continue
            retained_components.append(component)
        prediction.braid_components = retained_components

        counts: dict[tuple[str, str], int] = defaultdict(int)
        for event in prediction.events:
            counts[event.pair()] += 1
        for pair in sorted({event.pair() for event in capacity_events}):
            key = "|".join(pair)
            contract = prediction.parity_contract.setdefault(
                key,
                {
                    "nets": list(pair),
                    "required_parity": "even",
                },
            )
            count = counts.get(pair, 0)
            required_parity = str(contract.get("required_parity", "even"))
            contract.update(
                predicted_count=count,
                parity_satisfied=(
                    count % 2 == 0
                    if required_parity == "even"
                    else count % 2 == 1
                ),
            )
        prediction.diagnostics.append(
            "suppressed_overpacked_capacity_events=" + str(len(removed_ids))
        )

    return {
        "enabled": True,
        "algorithm": "technology_pitch_capacity_component_suppression_v1",
        "technology_pitch_um": pitch,
        "removed_event_ids": sorted(removed_ids),
        "removed_event_count": len(removed_ids),
        "components": audits,
    }


def add_capacity_recrossing_motifs(
    prediction: Prediction,
    nets: dict[str, NetGeometry],
    config: dict[str, Any],
    *,
    bend_radius_um: float,
    crossing_body_um: float,
    minimum_access_um: float,
    grid_um: float,
) -> dict[str, Any]:
    """Add parity-preserving two-crossing bypass motifs at compressed stages.

    The detector uses only terminal geometry, source fanout, and technology
    dimensions.  It never inspects a case/net name or a learned coordinate.
    """

    if not bool(config.get("enable_capacity_recrossing", True)):
        return {"enabled": False, "motifs": [], "stage_audits": []}
    minimum_groups = int(config.get("minimum_stage_source_pairs", 8))
    pressure_limit = float(config.get("corridor_pressure_ratio", 3.0))
    maximum_per_pair = int(config.get("maximum_recrossings_per_pair", 1))
    if maximum_per_pair < 1:
        return {"enabled": True, "motifs": [], "stage_audits": []}

    outgoing: dict[str, list[NetGeometry]] = defaultdict(list)
    for net in nets.values():
        outgoing[net.source.instance].append(net)
    pairs = [values for values in outgoing.values() if len(values) == 2]
    stages: dict[tuple[int, int, int], list[list[NetGeometry]]] = defaultdict(list)
    tolerance = max(grid_um, 1.0)
    for values in pairs:
        # A stage is defined in the source ports' launch frame.  Deriving a
        # separate axis from every terminal chord over-splits a fanout because
        # the two targets deliberately diverge.  Quantizing the physical port
        # launch to the router's 45-degree direction families groups equal
        # stages without using a component type, net name, or case coordinate.
        launch_vector = (
            statistics.fmean(
                math.cos(math.radians(net.source.orientation)) for net in values
            ),
            statistics.fmean(
                math.sin(math.radians(net.source.orientation)) for net in values
            ),
        )
        if math.hypot(*launch_vector) <= 1e-9:
            launch_vector = (
                statistics.fmean(net.vector[0] for net in values),
                statistics.fmean(net.vector[1] for net in values),
            )
        raw_angle = math.degrees(math.atan2(launch_vector[1], launch_vector[0]))
        direction_index = int(round(raw_angle / 45.0)) % 8
        angle = math.radians(45.0 * direction_index)
        axis = (math.cos(angle), math.sin(angle))
        normal = (-axis[1], axis[0])
        source = (
            statistics.fmean(net.source.x for net in values),
            statistics.fmean(net.source.y for net in values),
        )
        target = (
            statistics.fmean(net.target.x for net in values),
            statistics.fmean(net.target.y for net in values),
        )
        key = (
            int(round(_dot(source, axis) / tolerance)),
            int(round(_dot(target, axis) / tolerance)),
            direction_index,
        )
        values.sort(key=lambda net: _dot((net.source.x, net.source.y), normal))
        stages[key].append(values)

    existing_counts: dict[tuple[str, str], int] = defaultdict(int)
    for event in prediction.events:
        existing_counts[event.pair()] += 1
    inherited_net_orders = {
        name: list(order) for name, order in prediction.net_orders.items()
    }
    inherited_event_ranks = {
        event.event_id: (event.order_on_net_a, event.order_on_net_b)
        for event in prediction.events
    }
    inherited_required_parity = {
        tuple(sorted(str(net) for net in contract.get("nets", []))): str(
            contract.get("required_parity", "odd")
        )
        for contract in prediction.parity_contract.values()
        if len(contract.get("nets", [])) == 2
    }
    motif_reports = []
    stage_audits = []
    for stage_index, (key, groups) in enumerate(sorted(stages.items())):
        if len(groups) < minimum_groups:
            continue
        all_nets = [net for group in groups for net in group]
        axis = _unit(
            (
                statistics.fmean(net.vector[0] for net in all_nets),
                statistics.fmean(net.vector[1] for net in all_nets),
            )
        )
        normal = (-axis[1], axis[0])
        groups.sort(
            key=lambda group: statistics.fmean(
                _dot((net.source.x, net.source.y), normal) for net in group
            )
        )
        source_lanes = [
            _dot((net.source.x, net.source.y), normal) for net in all_nets
        ]
        longitudinal = [
            abs(
                _dot((net.target.x, net.target.y), axis)
                - _dot((net.source.x, net.source.y), axis)
            )
            for net in all_nets
        ]
        source_span = max(source_lanes) - min(source_lanes)
        corridor_length = statistics.median(longitudinal) if longitudinal else 0.0
        pressure = source_span / corridor_length if corridor_length > 1e-9 else math.inf
        triggered = pressure > pressure_limit
        audit = {
            "stage_key": list(key),
            "source_pair_count": len(groups),
            "source_span_um": source_span,
            "corridor_length_um": corridor_length,
            "pressure": pressure,
            "threshold": pressure_limit,
            "triggered": triggered,
            "motifs": [],
        }
        if not triggered:
            stage_audits.append(audit)
            continue

        pressured = groups[1] if len(groups) > 1 else groups[0]
        candidates: list[tuple[NetGeometry, NetGeometry, str]] = [
            (pressured[0], pressured[1], "same_source_capacity_bypass")
        ]
        if len(groups) >= 2 * minimum_groups and len(groups) >= 3:
            candidates.append(
                (pressured[0], groups[2][0], "adjacent_source_capacity_bypass")
            )

        lane_centers = [
            statistics.fmean(
                _dot((net.source.x, net.source.y), normal) for net in group
            )
            for group in groups
        ]
        lane_pitch = (
            statistics.median(
                abs(second - first)
                for first, second in zip(lane_centers, lane_centers[1:])
                if abs(second - first) > 1e-9
            )
            if len(set(lane_centers)) > 1
            else 2.0 * (bend_radius_um + crossing_body_um)
        )
        launch = 2.0 * bend_radius_um + crossing_body_um + minimum_access_um
        lateral = bend_radius_um + crossing_body_um + grid_um
        bypass_lateral = max(2.0 * lateral, 0.6 * lane_pitch)
        target_inset = bend_radius_um + crossing_body_um / 2.0

        for first, second, kind in candidates:
            pair = tuple(sorted((first.name, second.name)))
            if existing_counts[pair] != 0:
                continue
            source_center = (
                (first.source.x + second.source.x) / 2.0,
                (first.source.y + second.source.y) / 2.0,
            )
            target_center = (
                (first.target.x + second.target.x) / 2.0,
                (first.target.y + second.target.y) / 2.0,
            )
            if kind == "same_source_capacity_bypass":
                points = [
                    (
                        source_center[0] + launch * axis[0] - lateral * normal[0],
                        source_center[1] + launch * axis[1] - lateral * normal[1],
                    ),
                    (
                        target_center[0] - target_inset * axis[0],
                        target_center[1] - target_inset * axis[1],
                    ),
                ]
            else:
                points = [
                    (
                        target_center[0]
                        - launch * axis[0]
                        + bypass_lateral * normal[0],
                        target_center[1]
                        - launch * axis[1]
                        + bypass_lateral * normal[1],
                    ),
                    (
                        target_center[0] - target_inset * axis[0],
                        target_center[1] - target_inset * axis[1],
                    ),
                ]
            component = f"recross_{stage_index:03d}_{pair[0]}_{pair[1]}"
            created = []
            for motif_index, point in enumerate(points):
                rotation, pair_a, pair_b = _orientation_assignment(
                    nets[pair[0]], nets[pair[1]]
                )
                event = CrossingEvent(
                    event_id=f"xc__{pair[0]}__{pair[1]}__r{motif_index}",
                    net_a=pair[0],
                    net_b=pair[1],
                    ideal_center_um=(float(point[0]), float(point[1])),
                    evidence=[
                        "capacity_recrossing",
                        kind,
                        "even_pair_parity",
                        "technology_derived_anchor",
                    ],
                    confidence="medium",
                    topology_component=component,
                    topology_stage=motif_index,
                    preferred_rotation_deg=-45.0 if motif_index == 0 else rotation,
                    net_a_ports=pair_a,
                    net_b_ports=pair_b,
                )
                prediction.events.append(event)
                created.append(event.event_id)
            existing_counts[pair] += 2
            motif = {
                "pair": list(pair),
                "kind": kind,
                "events": created,
                "parity": "even",
                "technology_anchors_um": [list(point) for point in points],
            }
            motif_reports.append(motif)
            audit["motifs"].append(motif)
        stage_audits.append(audit)

    # Preserve the base estimator's event order and append deterministic
    # completion motifs.  This keeps manifests incrementally comparable; the
    # planner separately isolates disconnected completion components during
    # physical placement because the external Ripple tool may reorder inputs
    # internally.  Per-net topology order is rebuilt below and is independent
    # of this storage order.
    prediction.net_orders = _assign_net_orders(prediction.events, nets)
    affected_nets = {
        str(net)
        for motif in motif_reports
        for net in motif.get("pair", [])
    }
    for net_name, order in inherited_net_orders.items():
        if net_name not in affected_nets:
            prediction.net_orders[net_name] = order
    for event in prediction.events:
        inherited = inherited_event_ranks.get(event.event_id)
        if inherited is None:
            continue
        if event.net_a not in affected_nets:
            event.order_on_net_a = inherited[0]
        if event.net_b not in affected_nets:
            event.order_on_net_b = inherited[1]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for event in prediction.events:
        counts[event.pair()] += 1
    required_parity = {
        pair: (
            "even"
            if inherited_required_parity.get(pair) == "even"
            or any(
                "even_pair_parity" in event.evidence
                for event in prediction.events
                if event.pair() == pair
            )
            else "odd"
        )
        for pair in counts
    }
    prediction.parity_contract = {
        "|".join(pair): {
            "nets": list(pair),
            "required_parity": required_parity[pair],
            "predicted_count": count,
            "parity_satisfied": (
                count % 2 == 0
                if required_parity[pair] == "even"
                else count % 2 == 1
            ),
        }
        for pair, count in sorted(counts.items())
    }
    return {
        "enabled": True,
        "motif_count": len(motif_reports),
        "motifs": motif_reports,
        "stage_audits": stage_audits,
    }
