from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CrossingEvent:
    event_id: str
    net_a: str
    net_b: str
    ideal_center_um: tuple[float, float]
    evidence: list[str]
    confidence: str = "high"
    topology_component: str | None = None
    topology_stage: int | None = None
    order_on_net_a: int = -1
    order_on_net_b: int = -1
    preferred_rotation_deg: float = 0.0
    net_a_ports: tuple[str, str] = ("o1", "o3")
    net_b_ports: tuple[str, str] = ("o4", "o2")

    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.net_a, self.net_b)))


@dataclass
class PlacedCrossing:
    event: CrossingEvent
    center_um: tuple[float, float]
    rotation_deg: float
    legal: bool
    displacement_um: float
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event"]["ideal_center_um"] = list(self.event.ideal_center_um)
        data["center_um"] = list(self.center_um)
        return data


@dataclass
class Prediction:
    events: list[CrossingEvent]
    net_orders: dict[str, list[str]]
    braid_components: list[dict[str, Any]]
    parity_contract: dict[str, dict[str, Any]]
    diagnostics: list[str] = field(default_factory=list)

