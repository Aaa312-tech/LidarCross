from __future__ import annotations

from typing import Iterable

from .case_io import InstanceGeometry
from .model import PlacedCrossing
from .pcell_geometry import crossing_views


Box = tuple[float, float, float, float]


def _overlaps(first: Box, second: Box) -> bool:
    return (
        min(first[2], second[2]) - max(first[0], second[0]) > 1e-9
        and min(first[3], second[3]) - max(first[1], second[1]) > 1e-9
    )


def _inflate(box: Box, amount: float) -> Box:
    return (
        box[0] - amount,
        box[1] - amount,
        box[2] + amount,
        box[3] + amount,
    )


def crossing_box(item: PlacedCrossing, manifest: dict) -> Box:
    local = crossing_views(manifest)[float(item.rotation_deg)]["bbox_centered_um"]
    return (
        item.center_um[0] + float(local[0]),
        item.center_um[1] + float(local[1]),
        item.center_um[0] + float(local[2]),
        item.center_um[1] + float(local[3]),
    )


def placement_is_legal(
    candidate: PlacedCrossing,
    existing: Iterable[PlacedCrossing],
    fixed_obstacles: Iterable[InstanceGeometry],
    die: Box,
    manifest: dict,
) -> bool:
    """Check one candidate against the physical crossing placement contract."""

    box = crossing_box(candidate, manifest)
    if (
        box[0] < die[0]
        or box[1] < die[1]
        or box[2] > die[2]
        or box[3] > die[3]
    ):
        return False
    halo = float(manifest.get("halo_um", 0.0))
    if any(_overlaps(_inflate(box, halo), obstacle.bbox) for obstacle in fixed_obstacles):
        return False
    candidate_nets = {candidate.event.net_a, candidate.event.net_b}
    for item in existing:
        other = crossing_box(item, manifest)
        shared_parent = bool(candidate_nets & {item.event.net_a, item.event.net_b})
        if _overlaps(box if shared_parent else _inflate(box, halo), other):
            return False
    return True

