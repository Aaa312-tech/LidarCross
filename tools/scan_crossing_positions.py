from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(
    r"D:\xprogram\lidar_crossing\.venv-crossing-production\Scripts\python.exe"
)
DEFAULT_CONVERTER = Path(
    r"D:\xprogram\lidar_crossing_2\code\tools\pr_lidar_native\scripts"
    r"\lidar_yml_to_picdb_yml.py"
)
DEFAULT_ROUTER = Path(
    r"D:\xprogram\PIC-DB-main\PIC-DB-main\build_native_release"
    r"\pr_lidar_native.exe"
)
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _centers(value: str) -> list[tuple[float, float]]:
    result = []
    for item in value.split(";"):
        x_text, y_text = item.split(",", 1)
        result.append((float(x_text), float(y_text)))
    return result


def _summary(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    if not path.exists():
        return result
    markers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("marker\t"):
            marker = {}
            for field in line.split("\t")[1:]:
                if "=" in field:
                    key, value = field.split("=", 1)
                    marker[key] = value
            markers.append(marker)
        elif "=" in line:
            key, value = line.split("=", 1)
            if key in {"clean", "markers", "violations", "missing_route"}:
                result[key] = int(value)
    result["missing_nets"] = sorted(
        str(marker.get("nets", ""))
        for marker in markers
        if marker.get("type") == "missing_route"
    )
    return result


def _run_candidate(
    baseline: dict,
    crossing: str,
    center: tuple[float, float],
    output_root: Path,
    python: Path,
    converter: Path,
    router: Path,
) -> dict[str, object]:
    x, y = center
    label = f"x{x:g}_y{y:g}".replace("-", "m").replace(".", "p")
    candidate_dir = output_root / label
    candidate_dir.mkdir(parents=True, exist_ok=False)
    case = yaml.safe_load(yaml.safe_dump(baseline, sort_keys=False))
    instance = case["instances"][crossing]
    placement = instance["settings"]["placement"]
    macro = str(instance["settings"]["macro_type"])
    if macro != "picroute_crossing_0":
        raise ValueError(f"Only zero-degree crossings are supported; got {macro}")
    placement[1] = [x - 4.0, y - 4.0]
    case_path = candidate_dir / "routing_case.yml"
    case_path.write_text(
        yaml.safe_dump(case, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    converted = candidate_dir / "converted"
    native = candidate_dir / "native"
    converted.mkdir()
    native.mkdir()
    started = time.perf_counter()
    convert = subprocess.run(
        [
            str(python),
            str(converter),
            str(case_path),
            str(converted),
            "--preserve-lidar-library-lef",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    route_code = None
    if convert.returncode == 0:
        route = subprocess.run(
            [
                str(router),
                str(converted / "converted_lef.yml"),
                str(converted / "converted_def.yml"),
                str(native),
                "--skip-render",
                "--deterministic-order",
                "--max-iteration=20",
                "--mfot-mode=off",
                "--strict-preplaced-crossings",
                "--max-search-expanded=250000",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        route_code = route.returncode
        (candidate_dir / "router.stdout.txt").write_text(
            route.stdout, encoding="utf-8"
        )
        (candidate_dir / "router.stderr.txt").write_text(
            route.stderr, encoding="utf-8"
        )
    result = {
        "center_um": [x, y],
        "converter_exit_code": convert.returncode,
        "router_exit_code": route_code,
        "elapsed_s": time.perf_counter() - started,
        **_summary(native / "db_drc_summary.txt"),
    }
    (candidate_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable native router over crossing-center candidates."
    )
    parser.add_argument("--routing-case", type=Path, required=True)
    parser.add_argument("--crossing", required=True)
    parser.add_argument("--centers", type=_centers, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    args = parser.parse_args()
    if not SAFE_NAME.fullmatch(args.crossing):
        raise ValueError(f"Unsafe crossing name: {args.crossing!r}")
    baseline = yaml.safe_load(args.routing_case.resolve(strict=True).read_text("utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _run_candidate,
                baseline,
                args.crossing,
                center,
                output,
                args.python.resolve(strict=True),
                args.converter.resolve(strict=True),
                args.router.resolve(strict=True),
            )
            for center in args.centers
        ]
        results = [future.result() for future in futures]
    results.sort(key=lambda item: (int(item.get("missing_route", 10**9)), item["center_um"]))
    (output / "scan_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
