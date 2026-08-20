from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .case_io import NetGeometry
from .model import CrossingEvent, Prediction


Point = tuple[float, float]


def _cross(first: Point, second: Point) -> float:
    return first[0] * second[1] - first[1] * second[0]


def proper_segment_intersection(
    first: NetGeometry, second: NetGeometry, epsilon: float
) -> Point | None:
    p = (first.source.x, first.source.y)
    q = (second.source.x, second.source.y)
    r = first.vector
    s = second.vector
    denominator = _cross(r, s)
    if abs(denominator) <= epsilon:
        return None
    q_minus_p = (q[0] - p[0], q[1] - p[1])
    t = _cross(q_minus_p, s) / denominator
    u = _cross(q_minus_p, r) / denominator
    if not epsilon < t < 1.0 - epsilon or not epsilon < u < 1.0 - epsilon:
        return None
    return p[0] + t * r[0], p[1] + t * r[1]


def _connected_components(events: Iterable[CrossingEvent]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for event in events:
        adjacency[event.net_a].add(event.net_b)
        adjacency[event.net_b].add(event.net_a)
    remaining = set(adjacency)
    result: list[list[str]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        result.append(sorted(component))
    return result


def _projection(point: Point, axis: Point) -> float:
    return point[0] * axis[0] + point[1] * axis[1]


def _normalized(vector: Point) -> Point:
    length = math.hypot(*vector)
    if length <= 1e-12:
        return 1.0, 0.0
    return vector[0] / length, vector[1] / length


def _orientation_assignment(
    net_a: NetGeometry, net_b: NetGeometry
) -> tuple[float, tuple[str, str], tuple[str, str]]:
    angle_a = math.degrees(math.atan2(net_a.vector[1], net_a.vector[0])) % 180.0
    angle_b = math.degrees(math.atan2(net_b.vector[1], net_b.vector[0])) % 180.0

    def axis_distance(first: float, second: float) -> float:
        delta = abs((first - second) % 180.0)
        return min(delta, 180.0 - delta)

    candidates = []
    for rotation in (0.0, -45.0):
        axes = (rotation % 180.0, (rotation + 90.0) % 180.0)
        for a_index in (0, 1):
            b_index = 1 - a_index
            error = axis_distance(angle_a, axes[a_index]) ** 2 + axis_distance(
                angle_b, axes[b_index]
            ) ** 2
            pair_a = ("o1", "o3") if a_index == 0 else ("o4", "o2")
            pair_b = ("o1", "o3") if b_index == 0 else ("o4", "o2")
            candidates.append((error, abs(rotation), rotation, pair_a, pair_b))
    _error, _abs_rotation, rotation, pair_a, pair_b = min(candidates)
    return rotation, pair_a, pair_b


def _try_braid_schedule(
    component_index: int,
    members: list[str],
    nets: dict[str, NetGeometry],
    events_by_pair: dict[tuple[str, str], list[CrossingEvent]],
) -> dict | None:
    if len(members) < 3:
        return None
    relevant = {
        pair: records
        for pair, records in events_by_pair.items()
        if pair[0] in members and pair[1] in members
    }
    if not relevant or any(len(records) != 1 for records in relevant.values()):
        return None

    mean_vector = (
        sum(nets[name].vector[0] for name in members) / len(members),
        sum(nets[name].vector[1] for name in members) / len(members),
    )
    axis = _normalized(mean_vector)
    normal = (-axis[1], axis[0])
    source_order = sorted(
        members,
        key=lambda name: (
            _projection((nets[name].source.x, nets[name].source.y), normal), name
        ),
    )
    target_order = sorted(
        members,
        key=lambda name: (
            _projection((nets[name].target.x, nets[name].target.y), normal), name
        ),
    )
    target_rank = {name: index for index, name in enumerate(target_order)}
    inversions = {
        tuple(sorted((first, second)))
        for index, first in enumerate(source_order)
        for second in source_order[index + 1 :]
        if target_rank[first] > target_rank[second]
    }
    if inversions != set(relevant):
        return None

    current = list(source_order)
    layers: list[list[tuple[str, str]]] = []
    guard = len(members) * len(members) + 1
    while current != target_order and len(layers) < guard:
        changed = False
        for parity in (0, 1):
            swaps: list[tuple[int, str, str]] = []
            for index in range(parity, len(current) - 1, 2):
                first, second = current[index], current[index + 1]
                if target_rank[first] > target_rank[second]:
                    swaps.append((index, first, second))
            if not swaps:
                continue
            for index, first, second in swaps:
                current[index], current[index + 1] = second, first
            layers.append([(first, second) for _index, first, second in swaps])
            changed = True
        if not changed:
            break
    if current != target_order:
        return None

    source_u = sum(
        _projection((nets[name].source.x, nets[name].source.y), axis)
        for name in members
    ) / len(members)
    target_u = sum(
        _projection((nets[name].target.x, nets[name].target.y), axis)
        for name in members
    ) / len(members)
    source_v = {
        name: _projection((nets[name].source.x, nets[name].source.y), normal)
        for name in members
    }
    target_v = {
        name: _projection((nets[name].target.x, nets[name].target.y), normal)
        for name in members
    }
    component_name = f"braid_{component_index:03d}"
    for stage, swaps in enumerate(layers):
        fraction = (stage + 1.0) / (len(layers) + 1.0)
        u = source_u + fraction * (target_u - source_u)
        for first, second in swaps:
            pair = tuple(sorted((first, second)))
            event = relevant[pair][0]
            first_v = source_v[first] + fraction * (target_v[first] - source_v[first])
            second_v = source_v[second] + fraction * (
                target_v[second] - source_v[second]
            )
            v = (first_v + second_v) / 2.0
            event.ideal_center_um = (
                u * axis[0] + v * normal[0],
                u * axis[1] + v * normal[1],
            )
            event.topology_component = component_name
            event.topology_stage = stage
            event.evidence.append("reduced_wiring_diagram")
    return {
        "component": component_name,
        "members": members,
        "source_order": source_order,
        "target_order": target_order,
        "layer_count": len(layers),
        "layers": [
            [list(tuple(sorted(pair))) for pair in layer] for layer in layers
        ],
        "method": "parallel_adjacent_swap_layers",
    }


def _assign_net_orders(
    events: list[CrossingEvent], nets: dict[str, NetGeometry]
) -> dict[str, list[str]]:
    by_net: dict[str, list[CrossingEvent]] = defaultdict(list)
    for event in events:
        by_net[event.net_a].append(event)
        by_net[event.net_b].append(event)

    orders: dict[str, list[str]] = {}
    for net_name, records in sorted(by_net.items()):
        net = nets[net_name]
        dx, dy = net.vector
        length_squared = max(dx * dx + dy * dy, 1e-12)

        def key(event: CrossingEvent) -> tuple[float, int, str]:
            x, y = event.ideal_center_um
            parameter = (
                (x - net.source.x) * dx + (y - net.source.y) * dy
            ) / length_squared
            stage = event.topology_stage if event.topology_stage is not None else 10**9
            return parameter, stage, event.event_id

        ordered = sorted(records, key=key)
        orders[net_name] = [event.event_id for event in ordered]
        for rank, event in enumerate(ordered):
            if event.net_a == net_name:
                event.order_on_net_a = rank
            else:
                event.order_on_net_b = rank
    return orders


def predict_crossings(
    nets: dict[str, NetGeometry], endpoint_epsilon: float = 1e-6
) -> Prediction:
    records = sorted(nets.values(), key=lambda item: item.name)
    events: list[CrossingEvent] = []
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            if {
                first.source_name,
                first.target_name,
            } & {second.source_name, second.target_name}:
                continue
            center = proper_segment_intersection(first, second, endpoint_epsilon)
            if center is None:
                continue
            net_a, net_b = sorted((first.name, second.name))
            rotation, pair_a, pair_b = _orientation_assignment(nets[net_a], nets[net_b])
            events.append(
                CrossingEvent(
                    event_id=f"xc__{net_a}__{net_b}",
                    net_a=net_a,
                    net_b=net_b,
                    ideal_center_um=center,
                    evidence=["proper_terminal_chord_intersection", "odd_pair_parity"],
                    preferred_rotation_deg=rotation,
                    net_a_ports=pair_a,
                    net_b_ports=pair_b,
                )
            )

    events_by_pair: dict[tuple[str, str], list[CrossingEvent]] = defaultdict(list)
    for event in events:
        events_by_pair[event.pair()].append(event)

    braid_components: list[dict] = []
    for component_index, members in enumerate(_connected_components(events)):
        audit = _try_braid_schedule(
            component_index, members, nets, events_by_pair
        )
        if audit is not None:
            braid_components.append(audit)

    orders = _assign_net_orders(events, nets)
    parity_contract = {
        "|".join(pair): {
            "nets": list(pair),
            "required_parity": "odd",
            "predicted_count": len(pair_events),
            "parity_satisfied": len(pair_events) % 2 == 1,
        }
        for pair, pair_events in sorted(events_by_pair.items())
    }
    diagnostics: list[str] = []
    if any(not value["parity_satisfied"] for value in parity_contract.values()):
        diagnostics.append("one_or_more_pair_parity_contracts_failed")
    return Prediction(
        events=events,
        net_orders=orders,
        braid_components=braid_components,
        parity_contract=parity_contract,
        diagnostics=diagnostics,
    )

