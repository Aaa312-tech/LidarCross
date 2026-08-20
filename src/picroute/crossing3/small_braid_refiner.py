from __future__ import annotations

import copy
import itertools
import math
from typing import Any

from .case_io import InstanceGeometry, NetGeometry
from .constraint_placer import place_crossings
from .model import PlacedCrossing, Prediction
from .pcell_geometry import crossing_views


def closed_three_net_braids(prediction: Prediction) -> list[tuple[str, ...]]:
    """Return isolated complete three-parent braid event sets.

    The predicate is purely topological: every parent pair crosses exactly
    once and none of the three parents owns a crossing outside the component.
    It therefore targets the smallest nontrivial reduced wiring diagram
    without dispatching on a benchmark, net name, or coordinate.
    """

    event_by_id = {event.event_id: event for event in prediction.events}
    result = []
    for component in prediction.braid_components:
        parents = tuple(sorted(str(value) for value in component.get("parents", [])))
        identifiers = tuple(
            str(value)
            for value in (
                component.get("crossing_order")
                or component.get("event_order")
                or component.get("events")
                or []
            )
        )
        if len(parents) != 3 or len(identifiers) != 3:
            continue
        if any(identifier not in event_by_id for identifier in identifiers):
            continue
        required_pairs = set(itertools.combinations(parents, 2))
        actual_pairs = {event_by_id[identifier].pair() for identifier in identifiers}
        if actual_pairs != required_pairs:
            continue
        identifier_set = set(identifiers)
        if any(
            set(prediction.net_orders.get(parent, [])) != {
                identifier
                for identifier in identifiers
                if parent
                in (
                    event_by_id[identifier].net_a,
                    event_by_id[identifier].net_b,
                )
            }
            for parent in parents
        ):
            continue
        if any(
            not set(prediction.net_orders.get(parent, [])) <= identifier_set
            for parent in parents
        ):
            continue
        result.append(tuple(sorted(identifiers)))
    return sorted(result)


def _crossing_obstacle(
    item: PlacedCrossing, manifest: dict[str, Any]
) -> InstanceGeometry:
    view = crossing_views(manifest)[float(item.rotation_deg)]
    local = [float(value) for value in view["bbox_centered_um"]]
    box = (
        item.center_um[0] + local[0],
        item.center_um[1] + local[1],
        item.center_um[0] + local[2],
        item.center_um[1] + local[3],
    )
    return InstanceGeometry(
        name=item.event.event_id,
        lower_left=(box[0], box[1]),
        width=box[2] - box[0],
        height=box[3] - box[1],
        orientation="N",
    )


def refine_closed_three_net_braids(
    prediction: Prediction,
    placed: list[PlacedCrossing],
    nets: dict[str, NetGeometry],
    fixed_obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
    placement_config: dict[str, Any],
    guide_paths: dict[str, list[tuple[float, float]]] | None,
) -> tuple[list[PlacedCrossing], dict[str, Any]]:
    enabled = bool(placement_config.get("small_braid_access_refinement", True))
    components = closed_three_net_braids(prediction) if enabled else []
    if not components:
        return placed, {
            "enabled": enabled,
            "algorithm": "closed_three_net_braid_access_milp_v1",
            "component_count": 0,
            "components": [],
        }

    body = max(
        float(value)
        for value in manifest.get(
            "rotation_union_envelope_um",
            manifest.get("bbox_size_um", [8.0, 8.0]),
        )
    )
    derived_pitch = (
        body
        + 2.0 * float(placement_config.get("minimum_access_um", 10.0))
        + float(placement_config.get("waveguide_spacing_um", 1.0))
    )
    order_spacing = float(
        placement_config.get("small_braid_order_spacing_um", derived_pitch)
    )
    local_config = {
        **placement_config,
        "topology_order_spacing_um": order_spacing,
        "candidate_limit": int(
            placement_config.get("small_braid_candidate_limit", 256)
        ),
    }
    current_by_id = {item.event.event_id: item for item in placed}
    original_order = [item.event.event_id for item in placed]
    reports = []
    event_by_id = {event.event_id: event for event in prediction.events}
    for identifiers in components:
        identifier_set = set(identifiers)
        other_placements = [
            item
            for event_id, item in current_by_id.items()
            if event_id not in identifier_set
        ]
        obstacles = list(fixed_obstacles) + [
            _crossing_obstacle(item, manifest) for item in other_placements
        ]
        # Ripple has already found a geometry-legal open corridor, often far
        # from coincident terminal-chord intersections.  Use that placement as
        # the local optimization anchor so the MILP repairs sequence/access
        # without snapping the braid back into fixed devices.  Restore the
        # immutable topology ideals on the accepted result below.
        component_events = []
        for identifier in identifiers:
            event = copy.deepcopy(event_by_id[identifier])
            event.ideal_center_um = current_by_id[identifier].center_um
            component_events.append(event)
        parent_names = {
            name
            for event in component_events
            for name in (event.net_a, event.net_b)
        }
        component_prediction = Prediction(
            events=component_events,
            net_orders={
                name: [
                    identifier
                    for identifier in prediction.net_orders.get(name, [])
                    if identifier in identifier_set
                ]
                for name in sorted(parent_names)
            },
            braid_components=[],
            parity_contract={
                name: dict(contract)
                for name, contract in prediction.parity_contract.items()
                if set(str(value) for value in contract.get("nets", []))
                <= parent_names
            },
            diagnostics=[],
        )
        refined, report = place_crossings(
            component_prediction,
            nets,
            obstacles,
            die,
            manifest,
            local_config,
            guide_paths,
        )
        before = {
            identifier: list(current_by_id[identifier].center_um)
            for identifier in identifiers
        }
        local_anchor_centers = {
            event.event_id: list(event.ideal_center_um)
            for event in component_events
        }
        for item in refined:
            original_event = event_by_id[item.event.event_id]
            original_event.net_a_ports = item.event.net_a_ports
            original_event.net_b_ports = item.event.net_b_ports
            item.event = original_event
            item.displacement_um = math.dist(
                item.center_um, original_event.ideal_center_um
            )
            current_by_id[item.event.event_id] = item
        reports.append(
            {
                "events": list(identifiers),
                "before_centers_um": before,
                "local_anchor_centers_um": local_anchor_centers,
                "after_centers_um": {
                    item.event.event_id: list(item.center_um) for item in refined
                },
                "milp": report,
            }
        )
    return [current_by_id[event_id] for event_id in original_order], {
        "enabled": True,
        "algorithm": "closed_three_net_braid_access_milp_v1",
        "component_count": len(reports),
        "technology_derived_order_spacing_um": order_spacing,
        "components": reports,
    }
