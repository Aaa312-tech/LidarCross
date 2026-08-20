from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from picroute.crossing3.pdk_angle_audit import audit_pdk_angles


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce the straight/45/90 centerline contract on final GDS."
    )
    parser.add_argument("--route-result", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--gds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_pdk_angles(
        args.route_result.resolve(strict=True),
        args.geometry_report.resolve(strict=True),
        args.gds.resolve(strict=True),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "clean": report["clean"],
                "short_sbends": report["route_result_audit"][
                    "short_sbend_count"
                ],
                "prohibited_short_sbends": report["route_result_audit"][
                    "prohibited_short_sbend_count"
                ],
                "access_lateral_offsets": report["route_result_audit"][
                    "access_lateral_offset_count"
                ],
                "off_grid_segments": report["route_result_audit"][
                    "off_grid_segment_count"
                ],
                "unsupported_turns": report["route_result_audit"][
                    "unsupported_turn_count"
                ],
                "unsupported_recoveries": report["renderer_audit"][
                    "unsupported_recovery_count"
                ],
                "prohibited_access_sbends": report["renderer_audit"][
                    "prohibited_access_sbend_count"
                ],
                "short_sbend_cells": report["gds_audit"][
                    "short_sbend_cell_count"
                ],
                "report": str(args.out.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
