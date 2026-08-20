from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode


class CaseLoader(yaml.SafeLoader):
    """Load LiDAR YAML without constructing tagged Python objects."""


def _construct_tuple(loader: CaseLoader, node: SequenceNode) -> list[Any]:
    return loader.construct_sequence(node, deep=True)


def _construct_tagged_data(
    loader: CaseLoader, _suffix: str, node: yaml.Node
) -> Any:
    if isinstance(node, SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, ScalarNode):
        return loader.construct_scalar(node)
    raise TypeError(f"Unsupported YAML node {type(node).__name__}")


CaseLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)
CaseLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:", _construct_tagged_data
)


ORIENTATION_TRANSFORMS = {
    "N": (0.0, False),
    "W": (90.0, False),
    "S": (180.0, False),
    "E": (270.0, False),
    "FN": (0.0, True),
    "FW": (90.0, True),
    "FS": (180.0, True),
    "FE": (270.0, True),
}


@dataclass(frozen=True)
class Port:
    instance: str
    pin: str
    x: float
    y: float
    orientation: float
    width: float

    @property
    def name(self) -> str:
        return f"{self.instance},{self.pin}"


@dataclass(frozen=True)
class InstanceGeometry:
    name: str
    lower_left: tuple[float, float]
    width: float
    height: float
    orientation: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x, y = self.lower_left
        if self.orientation in {"E", "W", "FE", "FW"}:
            return x, y, x + self.height, y + self.width
        return x, y, x + self.width, y + self.height


@dataclass(frozen=True)
class NetGeometry:
    name: str
    source_name: str
    target_name: str
    source: Port
    target: Port

    @property
    def vector(self) -> tuple[float, float]:
        return self.target.x - self.source.x, self.target.y - self.source.y

    @property
    def length(self) -> float:
        return math.hypot(*self.vector)


def load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=CaseLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def placement_tuple(node: dict[str, Any]) -> tuple[list[float], str]:
    placement = (node.get("settings") or {}).get("placement")
    if not placement or len(placement) < 3:
        raise ValueError("LiDAR instance is missing settings.placement")
    return [float(placement[1][0]), float(placement[1][1])], str(
        placement[2] or "N"
    ).upper()


def macro_for_instance(
    case: dict[str, Any], node: dict[str, Any]
) -> dict[str, Any]:
    settings = node.get("settings") or {}
    library = case.get("library") or {}
    macro_type = settings.get("macro_type")
    if macro_type in library:
        return library[macro_type]
    component = node.get("component")
    if component in library:
        return library[component]
    raise KeyError(
        f"No library macro for component={component!r}, macro_type={macro_type!r}"
    )


def instance_geometry(
    case: dict[str, Any], name: str, node: dict[str, Any]
) -> InstanceGeometry:
    lower_left, orientation = placement_tuple(node)
    macro = macro_for_instance(case, node)
    width, height = [float(value) for value in macro["size"]]
    return InstanceGeometry(
        name=name,
        lower_left=(lower_left[0], lower_left[1]),
        width=width,
        height=height,
        orientation=orientation,
    )


def _transform_local_point(
    x: float, y: float, width: float, height: float, orientation: str
) -> tuple[float, float]:
    transforms = {
        "N": (x, y),
        "S": (width - x, height - y),
        "W": (height - y, x),
        "E": (y, width - x),
        "FN": (width - x, y),
        "FS": (x, height - y),
        "FW": (height - y, width - x),
        "FE": (y, x),
    }
    try:
        return transforms[orientation]
    except KeyError as error:
        raise ValueError(f"Unsupported orientation {orientation}") from error


def _transform_orientation(angle: float, orientation: str) -> float:
    rotation, mirror = ORIENTATION_TRANSFORMS[orientation]
    return ((180.0 - angle if mirror else angle) + rotation) % 360.0


def absolute_ports(case: dict[str, Any]) -> dict[str, Port]:
    result: dict[str, Port] = {}
    for name, node in (case.get("instances") or {}).items():
        geometry = instance_geometry(case, str(name), node)
        macro = macro_for_instance(case, node)
        for pin_name, pin in (macro.get("pins") or {}).items():
            dx, dy = _transform_local_point(
                float(pin.get("pin_offset_x", 0.0)),
                float(pin.get("pin_offset_y", 0.0)),
                geometry.width,
                geometry.height,
                geometry.orientation,
            )
            port = Port(
                instance=str(name),
                pin=str(pin_name),
                x=geometry.lower_left[0] + dx,
                y=geometry.lower_left[1] + dy,
                orientation=_transform_orientation(
                    float(pin.get("pin_orient", 0.0)), geometry.orientation
                ),
                width=float(pin.get("pin_width", 0.5)),
            )
            result[port.name] = port
    return result


def two_pin_nets(case: dict[str, Any]) -> dict[str, NetGeometry]:
    ports = absolute_ports(case)
    result: dict[str, NetGeometry] = {}
    for name, endpoints in (case.get("nets") or {}).items():
        if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2:
            continue
        source_name, target_name = str(endpoints[0]), str(endpoints[1])
        if source_name not in ports or target_name not in ports:
            continue
        result[str(name)] = NetGeometry(
            name=str(name),
            source_name=source_name,
            target_name=target_name,
            source=ports[source_name],
            target=ports[target_name],
        )
    return result


def fixed_obstacles(case: dict[str, Any]) -> list[InstanceGeometry]:
    # Every source instance is immutable in crossing3, even when the historical
    # YAML says UNPLACED.  Its supplied coordinates are the fixed baseline.
    return [
        instance_geometry(case, str(name), node)
        for name, node in (case.get("instances") or {}).items()
    ]


def die_area(case: dict[str, Any]) -> tuple[float, float, float, float]:
    area = (case.get("settings") or {}).get("die_area")
    if area and len(area) == 2:
        return (
            float(area[0][0]),
            float(area[0][1]),
            float(area[1][0]),
            float(area[1][1]),
        )
    boxes = [geometry.bbox for geometry in fixed_obstacles(case)]
    if not boxes:
        raise ValueError("Case has neither settings.die_area nor instances")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )

