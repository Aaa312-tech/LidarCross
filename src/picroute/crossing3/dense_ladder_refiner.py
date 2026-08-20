from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from typing import Any

from .case_io import InstanceGeometry, NetGeometry
from .direction_solver import solve_net_directions
from .model import PlacedCrossing, Prediction
from .placement_legal import placement_is_legal


def refine_dense_source_ladders(
    prediction: Prediction,
    placed: list[PlacedCrossing],
    nets: dict[str, NetGeometry],
    fixed_obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[PlacedCrossing], dict[str, Any]]:
    """Move a congested first crossing column into its launch corridor.

    The topology predicate is a two-spine ladder whose branch nets all launch
    from the same physical source instance and meet the same spine first.  The
    amount of movement is selected from the exact port-access penalty, then
    quantized to a technology-derived crossing pitch.
    """

    enabled = bool(config.get("dense_ladder_refinement", True))
    if not enabled:
        return placed, {
            "enabled": False,
            "algorithm": "dense_same_source_ladder_access_shift_v2",
            "ladders": [],
        }
    # Ladder access is a local multi-branch routing problem, not a capacity
    # completion motif.  Reusing the latter's eight-pair threshold silently
    # skipped physically dense six- and seven-branch source banks.  Four
    # common branches are sufficient to establish the two-spine topology and
    # justify a joint, technology-pitch entrance shift.
    minimum_branches = int(config.get("dense_ladder_minimum_branches", 4))
    minimum_access = float(config.get("minimum_access_um", 10.0))
    minimum_radius = float(config.get("bend_radius_um", 5.0))
    direct_threshold = float(
        config.get(
            "short_direct_access_threshold_um",
            2.0 * (minimum_access + minimum_radius),
        )
    )
    try:
        _directions, direction_audit = solve_net_directions(
            placed,
            nets,
            manifest,
            minimum_access,
            minimum_radius,
            direct_threshold,
        )
    except RuntimeError as error:
        return placed, {
            "enabled": True,
            "algorithm": "dense_same_source_ladder_access_shift_v1",
            "ladders": [],
            "skipped_reason": str(error),
        }

    events_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for event in prediction.events:
        pair = event.pair()
        events_by_pair[pair].append(event.event_id)
        adjacency[pair[0]].add(pair[1])
        adjacency[pair[1]].add(pair[0])

    body = max(
        float(value)
        for value in manifest.get(
            "rotation_union_envelope_um",
            manifest.get("bbox_size_um", [8.0, 8.0]),
        )
    )
    pitch = (
        body
        + 2.0 * minimum_access
        + float(config.get("waveguide_spacing_um", 1.0))
    )
    one_pitch_threshold = float(
        config.get("dense_ladder_one_pitch_penalty", 1.0)
    )
    two_pitch_threshold = float(
        config.get("dense_ladder_two_pitch_penalty", 3.0)
    )
    grid = float(config.get("grid_um", 2.0))
    current = copy.deepcopy(placed)
    current_by_id = {item.event.event_id: item for item in current}
    reports = []
    shifted_events: set[str] = set()

    for spine_pair, direct_events in sorted(events_by_pair.items()):
        if len(direct_events) != 1:
            continue
        first_spine, second_spine = spine_pair
        common = (adjacency[first_spine] & adjacency[second_spine]) - set(spine_pair)
        if len(common) < minimum_branches:
            continue
        first_by_branch: dict[str, tuple[str, str]] = {}
        for branch in sorted(common):
            first_pair = tuple(sorted((first_spine, branch)))
            second_pair = tuple(sorted((second_spine, branch)))
            if (
                len(events_by_pair[first_pair]) != 1
                or len(events_by_pair[second_pair]) != 1
            ):
                continue
            order = prediction.net_orders.get(branch, [])
            if not order:
                continue
            if order[0] == events_by_pair[first_pair][0]:
                first_by_branch[branch] = (first_spine, order[0])
            elif order[0] == events_by_pair[second_pair][0]:
                first_by_branch[branch] = (second_spine, order[0])
        if len(first_by_branch) != len(common):
            continue
        side_counts = Counter(spine for spine, _event_id in first_by_branch.values())
        selected_spine, branch_count = side_counts.most_common(1)[0]
        if branch_count != len(common) or branch_count < minimum_branches:
            continue
        branches = sorted(first_by_branch)
        source_instances = {nets[branch].source.instance for branch in branches}
        if source_instances != {nets[selected_spine].source.instance}:
            continue

        source_penalties = {
            branch: float(
                direction_audit[branch]["selected_segment_access"][0]["penalty"]
            )
            for branch in branches
        }
        maximum_penalty = max(source_penalties.values(), default=0.0)
        pitch_count = (
            2
            if maximum_penalty > two_pitch_threshold
            else 1
            if maximum_penalty > one_pitch_threshold
            else 0
        )
        event_ids = [first_by_branch[branch][1] for branch in branches]
        audit: dict[str, Any] = {
            "spines": list(spine_pair),
            "selected_source_spine": selected_spine,
            "source_instance": nets[selected_spine].source.instance,
            "branches": branches,
            "event_ids": event_ids,
            "source_access_penalty_by_branch": source_penalties,
            "maximum_source_access_penalty": maximum_penalty,
            "pitch_count": pitch_count,
            "technology_pitch_um": pitch,
            "shifted": False,
        }
        if pitch_count == 0 or any(event_id in shifted_events for event_id in event_ids):
            reports.append(audit)
            continue

        launch = (
            sum(
                math.cos(math.radians(nets[branch].source.orientation))
                for branch in branches
            ),
            sum(
                math.sin(math.radians(nets[branch].source.orientation))
                for branch in branches
            ),
        )
        launch_length = math.hypot(*launch)
        if launch_length <= 1e-9:
            launch = nets[selected_spine].vector
            launch_length = math.hypot(*launch)
        axis = (launch[0] / launch_length, launch[1] / launch_length)
        moving = set(event_ids)
        legal_existing = [
            item for item in current if item.event.event_id not in moving
        ]
        accepted_candidates: list[PlacedCrossing] = []
        attempted_shifts = []
        accepted_shift: tuple[float, float] | None = None
        accepted_pitch_count = 0
        for candidate_pitch_count in range(pitch_count, 0, -1):
            distance = candidate_pitch_count * pitch
            shift = (
                round(distance * axis[0] / grid) * grid,
                round(distance * axis[1] / grid) * grid,
            )
            candidates = []
            for event_id in event_ids:
                original = current_by_id[event_id]
                center = (
                    original.center_um[0] + shift[0],
                    original.center_um[1] + shift[1],
                )
                candidate = copy.deepcopy(original)
                candidate.center_um = center
                candidate.displacement_um = math.dist(
                    center, candidate.event.ideal_center_um
                )
                candidates.append(candidate)

            trial: list[PlacedCrossing] = []
            legal = True
            for candidate in candidates:
                if not placement_is_legal(
                    candidate,
                    [*legal_existing, *trial],
                    fixed_obstacles,
                    die,
                    manifest,
                ):
                    legal = False
                    break
                trial.append(candidate)
            attempted_shifts.append(
                {
                    "pitch_count": candidate_pitch_count,
                    "shift_vector_um": list(shift),
                    "legal": legal,
                }
            )
            if legal:
                accepted_candidates = trial
                accepted_shift = shift
                accepted_pitch_count = candidate_pitch_count
                break

        audit["attempted_shifts"] = attempted_shifts
        audit["legal"] = accepted_shift is not None
        if accepted_shift is not None:
            before = {
                event_id: list(current_by_id[event_id].center_um)
                for event_id in event_ids
            }
            for candidate in accepted_candidates:
                current_by_id[candidate.event.event_id] = candidate
            current = [current_by_id[item.event.event_id] for item in current]
            shifted_events.update(event_ids)
            audit["shifted"] = True
            audit["accepted_pitch_count"] = accepted_pitch_count
            audit["shift_vector_um"] = list(accepted_shift)
            audit["before_centers_um"] = before
            audit["after_centers_um"] = {
                item.event.event_id: list(item.center_um)
                for item in accepted_candidates
            }
        reports.append(audit)

    return current, {
        "enabled": True,
        "algorithm": "dense_same_source_ladder_access_shift_v2",
        "technology_pitch_um": pitch,
        "one_pitch_penalty_threshold": one_pitch_threshold,
        "two_pitch_penalty_threshold": two_pitch_threshold,
        "shifted_ladder_count": sum(bool(item["shifted"]) for item in reports),
        "ladders": reports,
    }
