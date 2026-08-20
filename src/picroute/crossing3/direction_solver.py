from __future__ import annotations

from collections import defaultdict
from typing import Any

from .access_contract import effective_hard_access, port_connection_feasibility
from .case_io import NetGeometry
from .model import PlacedCrossing
from .pcell_geometry import absolute_port, crossing_views


def _leg_feasibility(
    first_point: tuple[float, float],
    first_outward: float,
    second_point: tuple[float, float],
    second_outward: float,
    minimum_access_um: float,
    minimum_radius_um: float,
    direct_threshold_um: float,
) -> dict[str, Any]:
    return port_connection_feasibility(
        first_point,
        first_outward,
        second_point,
        second_outward,
        minimum_access_um,
        direct_threshold_um,
        minimum_radius_um,
    )


def solve_net_directions(
    placed: list[PlacedCrossing],
    nets: dict[str, NetGeometry],
    manifest: dict[str, Any],
    minimum_access_um: float,
    minimum_radius_um: float = 5.0,
    direct_threshold_um: float | None = None,
    *,
    selected_nets: set[str] | None = None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Choose a physically feasible entry/exit state for every crossing chain."""

    direct_threshold = float(
        direct_threshold_um
        if direct_threshold_um is not None
        else 2.0 * (minimum_access_um + minimum_radius_um)
    )
    events_by_net: dict[str, list[PlacedCrossing]] = defaultdict(list)
    for item in placed:
        events_by_net[item.event.net_a].append(item)
        events_by_net[item.event.net_b].append(item)
    views = crossing_views(manifest)
    result: dict[str, dict[str, list[str]]] = {}
    audits: dict[str, Any] = {}
    for net_name, items in sorted(events_by_net.items()):
        if selected_nets is not None and net_name not in selected_nets:
            continue
        net = nets[net_name]

        def rank(item: PlacedCrossing) -> tuple[int, str]:
            value = (
                item.event.order_on_net_a
                if item.event.net_a == net_name
                else item.event.order_on_net_b
            )
            return int(value), item.event.event_id

        items.sort(key=rank)
        states: list[list[tuple[str, str]]] = []
        for item in items:
            pair = (
                item.event.net_a_ports
                if item.event.net_a == net_name
                else item.event.net_b_ports
            )
            states.append([tuple(pair), (pair[1], pair[0])])

        dynamic: list[list[tuple[float, int | None]]] = []
        for index, item in enumerate(items):
            view = views[float(item.rotation_deg)]
            row: list[tuple[float, int | None]] = []
            for entry, _exit in states[index]:
                entry_point, entry_angle = absolute_port(item.center_um, view, entry)
                if index == 0:
                    feasibility = _leg_feasibility(
                        (net.source.x, net.source.y),
                        net.source.orientation,
                        entry_point,
                        entry_angle,
                        minimum_access_um,
                        minimum_radius_um,
                        direct_threshold,
                    )
                    row.append(
                        (
                            float("inf")
                            if effective_hard_access(feasibility)
                            else float(feasibility["penalty"]),
                            None,
                        )
                    )
                    continue
                previous = items[index - 1]
                previous_view = views[float(previous.rotation_deg)]
                choices = []
                for previous_state, (_previous_entry, previous_exit) in enumerate(
                    states[index - 1]
                ):
                    exit_point, exit_angle = absolute_port(
                        previous.center_um, previous_view, previous_exit
                    )
                    feasibility = _leg_feasibility(
                        exit_point,
                        exit_angle,
                        entry_point,
                        entry_angle,
                        minimum_access_um,
                        minimum_radius_um,
                        direct_threshold,
                    )
                    previous_cost = dynamic[index - 1][previous_state][0]
                    if previous_cost == float("inf") or effective_hard_access(
                        feasibility
                    ):
                        continue
                    choices.append(
                        (previous_cost + float(feasibility["penalty"]), previous_state)
                    )
                row.append(min(choices) if choices else (float("inf"), None))
            dynamic.append(row)

        final_choices = []
        last = items[-1]
        last_view = views[float(last.rotation_deg)]
        for state_index, (_entry, exit_name) in enumerate(states[-1]):
            exit_point, exit_angle = absolute_port(last.center_um, last_view, exit_name)
            feasibility = _leg_feasibility(
                exit_point,
                exit_angle,
                (net.target.x, net.target.y),
                net.target.orientation,
                minimum_access_um,
                minimum_radius_um,
                direct_threshold,
            )
            if dynamic[-1][state_index][0] == float("inf") or effective_hard_access(
                feasibility
            ):
                continue
            final_choices.append(
                (
                    dynamic[-1][state_index][0]
                    + float(feasibility["penalty"]),
                    state_index,
                )
            )
        if not final_choices:
            raise RuntimeError(
                f"No physically feasible crossing-port direction chain for {net_name}"
            )
        final_cost, final_state = min(final_choices)
        selected = [0] * len(items)
        selected[-1] = final_state
        for index in range(len(items) - 1, 0, -1):
            previous_state = dynamic[index][selected[index]][1]
            assert previous_state is not None
            selected[index - 1] = previous_state
        result[net_name] = {
            item.event.event_id: list(states[index][selected[index]])
            for index, item in enumerate(items)
        }
        selected_access = []
        previous_point = (net.source.x, net.source.y)
        previous_angle = net.source.orientation
        previous_name = net.source_name
        for index, item in enumerate(items):
            entry, exit_name = states[index][selected[index]]
            view = views[float(item.rotation_deg)]
            entry_point, entry_angle = absolute_port(item.center_um, view, entry)
            feasibility = _leg_feasibility(
                previous_point,
                previous_angle,
                entry_point,
                entry_angle,
                minimum_access_um,
                minimum_radius_um,
                direct_threshold,
            )
            selected_access.append(
                {
                    "from": previous_name,
                    "to": f"{item.event.event_id},{entry}",
                    **feasibility,
                    "effective_hard_infeasible": effective_hard_access(feasibility),
                }
            )
            previous_point, previous_angle = absolute_port(
                item.center_um, view, exit_name
            )
            previous_name = f"{item.event.event_id},{exit_name}"
        tail = _leg_feasibility(
            previous_point,
            previous_angle,
            (net.target.x, net.target.y),
            net.target.orientation,
            minimum_access_um,
            minimum_radius_um,
            direct_threshold,
        )
        selected_access.append(
            {
                "from": previous_name,
                "to": net.target_name,
                **tail,
                "effective_hard_infeasible": effective_hard_access(tail),
            }
        )
        audits[net_name] = {
            "crossing_order": [item.event.event_id for item in items],
            "selected_entry_exit_ports": [
                list(states[index][selected[index]]) for index in range(len(items))
            ],
            "connection_proxy_cost": float(final_cost),
            "method": "chain_dynamic_programming_exact_access_contract",
            "selected_segment_access": selected_access,
            "hard_invalid_segment_count": sum(
                bool(item["effective_hard_infeasible"])
                for item in selected_access
            ),
        }
    return result, audits
