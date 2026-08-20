from __future__ import annotations

import copy
import math
from typing import Any

from .case_io import InstanceGeometry
from .model import PlacedCrossing
from .placement_legal import placement_is_legal


TRACK_TOLERANCE_UM = 1e-6


def snap_to_routing_track(value: float, origin: float, grid: float) -> float:
    """Snap to native A* cell centres: origin + (index + 1/2) * grid."""

    if grid <= 0.0:
        raise ValueError("Routing grid must be positive")
    normalized = (float(value) - float(origin) - grid / 2.0) / grid
    # Avoid Python's tie-to-even rule. A point exactly between tracks moves
    # deterministically toward the positive track.
    index = math.floor(normalized + 0.5 + 1e-12)
    return float(origin) + (index + 0.5) * grid


def is_on_routing_track(
    value: float, origin: float, grid: float, tolerance: float = TRACK_TOLERANCE_UM
) -> bool:
    return abs(float(value) - snap_to_routing_track(value, origin, grid)) <= tolerance


def center_is_track_compatible(
    center: tuple[float, float] | list[float],
    die: tuple[float, float, float, float],
    grid: float,
) -> bool:
    return is_on_routing_track(center[0], die[0], grid) and is_on_routing_track(
        center[1], die[1], grid
    )


def _candidate_track_centers(
    center: tuple[float, float],
    die: tuple[float, float, float, float],
    grid: float,
    rings: int,
) -> list[tuple[float, float]]:
    base = (
        snap_to_routing_track(center[0], die[0], grid),
        snap_to_routing_track(center[1], die[1], grid),
    )
    result = []
    for x_index in range(-rings, rings + 1):
        for y_index in range(-rings, rings + 1):
            result.append(
                (base[0] + x_index * grid, base[1] + y_index * grid)
            )
    return sorted(
        set(result),
        key=lambda candidate: (
            round(math.dist(candidate, center), 12),
            candidate[0],
            candidate[1],
        ),
    )


def legalize_crossing_centers_to_routing_tracks(
    placed: list[PlacedCrossing],
    fixed_obstacles: list[InstanceGeometry],
    die: tuple[float, float, float, float],
    manifest: dict[str, Any],
    grid: float,
    *,
    search_rings: int = 4,
) -> tuple[list[PlacedCrossing], dict[str, Any]]:
    """Move only crossings onto port-compatible native routing tracks.

    A centre on both cell-centre tracks makes cardinal port axes and diagonal
    x+y/x-y invariants intersect the native 8-neighbour A* lattice. This
    removes renderer-side free-angle endpoint S-bends while leaving every
    original benchmark instance untouched.
    """

    current = copy.deepcopy(placed)
    by_id = {item.event.event_id: item for item in current}
    changes = []
    for event_id in sorted(by_id):
        item = by_id[event_id]
        before = item.center_um
        if center_is_track_compatible(before, die, grid):
            continue
        existing = [other for other in current if other.event.event_id != event_id]
        selected = None
        for center in _candidate_track_centers(before, die, grid, search_rings):
            candidate = copy.deepcopy(item)
            candidate.center_um = center
            candidate.displacement_um = math.dist(
                center, candidate.event.ideal_center_um
            )
            if placement_is_legal(
                candidate, existing, fixed_obstacles, die, manifest
            ):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                f"No legal routing-track centre for {event_id} within "
                f"{search_rings} grid rings of {before}"
            )
        item.center_um = selected.center_um
        item.displacement_um = selected.displacement_um
        changes.append(
            {
                "crossing": event_id,
                "before_um": list(before),
                "after_um": list(item.center_um),
                "movement_um": math.dist(before, item.center_um),
            }
        )

    incompatible = sorted(
        item.event.event_id
        for item in current
        if not center_is_track_compatible(item.center_um, die, grid)
    )
    return current, {
        "schema_version": 1,
        "algorithm": "native_cell_center_lattice_legalizer_v1",
        "grid_um": float(grid),
        "grid_origin_um": [float(die[0]), float(die[1])],
        "track_offset_um": float(grid) / 2.0,
        "input_crossing_count": len(placed),
        "moved_crossing_count": len(changes),
        "changes": changes,
        "incompatible_crossing_count": len(incompatible),
        "incompatible_crossings": incompatible,
        "clean": not incompatible,
    }
