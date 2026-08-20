from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASE_PORT_PAIRS = (("o1", "o3"), ("o4", "o2"))


def _xy(value: Any, scale: float = 1.0) -> list[float]:
    if hasattr(value, "x") and hasattr(value, "y"):
        return [float(value.x) * scale, float(value.y) * scale]
    return [float(value[0]) * scale, float(value[1]) * scale]


def extract_crossing_manifest(output_dir: Path, halo_um: float) -> dict[str, Any]:
    """Extract the exact gdsfactory crossing used by materialization/rendering."""

    import gdsfactory as gf

    try:
        gf.get_active_pdk()
    except Exception:
        try:
            gf.gpdk.PDK.activate()
        except Exception:
            from gdsfactory.generic_tech import get_generic_pdk

            get_generic_pdk().activate()

    output_dir.mkdir(parents=True, exist_ok=True)
    component = gf.components.crossing()
    gds_path = output_dir / "crossing_reference.gds"
    component.write_gds(gds_path)
    ports: dict[str, dict[str, Any]] = {}
    for port in component.ports:
        if hasattr(port, "dcenter"):
            center = _xy(port.dcenter)
            width = float(getattr(port, "dwidth", port.width))
        else:
            dbu = float(getattr(getattr(component, "kcl", None), "dbu", 0.001))
            center = _xy(port.center, dbu)
            width = float(port.width) * dbu
        ports[str(port.name)] = {
            "center_um": center,
            "orientation_deg": float(port.orientation),
            "width_um": width,
            "layer": str(port.layer),
        }
    width = float(component.dxsize)
    height = float(component.dysize)
    diagonal = math.hypot(width, height)
    manifest = {
        "schema_version": 1,
        "generator": "gdsfactory.components.crossing",
        "generator_settings": {},
        "gdsfactory_version": getattr(gf, "__version__", "unknown"),
        "cell_name": component.name,
        "gds_sha256": hashlib.sha256(gds_path.read_bytes()).hexdigest(),
        "bbox_size_um": [width, height],
        "rotation_union_envelope_um": [diagonal, diagonal],
        "halo_um": float(halo_um),
        "placement_union_envelope_um": [
            diagonal + 2.0 * float(halo_um),
            diagonal + 2.0 * float(halo_um),
        ],
        "allowed_rotations_deg": [0.0, -45.0],
        "ports": ports,
        "reference_gds": str(gds_path.resolve()),
    }
    (output_dir / "crossing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def rotate(point: tuple[float, float] | list[float], degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return (
        float(point[0]) * cosine - float(point[1]) * sine,
        float(point[0]) * sine + float(point[1]) * cosine,
    )


def crossing_view(manifest: dict[str, Any], rotation: float) -> dict[str, Any]:
    ports: dict[str, dict[str, Any]] = {}
    for name, source in manifest["ports"].items():
        offset = rotate(source["center_um"], rotation)
        ports[str(name)] = {
            "offset_centered_um": [offset[0], offset[1]],
            "orientation_deg": (float(source["orientation_deg"]) + rotation) % 360.0,
            "width_um": float(source["width_um"]),
            "layer": source.get("layer", "1"),
        }
    width, height = [float(value) for value in manifest["bbox_size_um"]]
    corners = [
        rotate((x, y), rotation)
        for x in (-width / 2.0, width / 2.0)
        for y in (-height / 2.0, height / 2.0)
    ]
    min_x = min(point[0] for point in corners)
    min_y = min(point[1] for point in corners)
    max_x = max(point[0] for point in corners)
    max_y = max(point[1] for point in corners)
    for port in ports.values():
        port["local_center_um"] = [
            port["offset_centered_um"][0] - min_x,
            port["offset_centered_um"][1] - min_y,
        ]
    return {
        "rotation_deg": float(rotation),
        "bbox_centered_um": [min_x, min_y, max_x, max_y],
        "size_um": [max_x - min_x, max_y - min_y],
        "ports": ports,
    }


def crossing_views(manifest: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {
        float(rotation): crossing_view(manifest, float(rotation))
        for rotation in manifest["allowed_rotations_deg"]
    }


def absolute_port(
    center: tuple[float, float] | list[float],
    view: dict[str, Any],
    port_name: str,
) -> tuple[tuple[float, float], float]:
    port = view["ports"][port_name]
    offset = port["offset_centered_um"]
    return (
        (float(center[0]) + float(offset[0]), float(center[1]) + float(offset[1])),
        float(port["orientation_deg"]),
    )

