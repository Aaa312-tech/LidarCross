from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from picroute.crossing3.benchmark_source import discover_benchmarks
from picroute.crossing3.planner import DEFAULT_CONFIG, plan_benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the crossing3 HCTP crossing-only frontend."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", action="append", dest="cases")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--list", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--backend",
        action="store_true",
        help="Run the frozen strict-preplaced router, GDS renderer, and continuity gate.",
    )
    args = parser.parse_args()
    catalog = discover_benchmarks()
    if args.list:
        for name, benchmark in sorted(catalog.items()):
            print(f"{name}\t{benchmark.sha256}\t{benchmark.path}")
        return 0
    if not args.all and not args.cases:
        parser.error("select --all or at least one --case")
    result = plan_benchmarks(
        None if args.all else args.cases,
        config_path=args.config,
        run_id=args.run_id,
        run_backend=args.backend,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"FRONTEND_PASS", "ACCEPTED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
