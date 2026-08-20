from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


AUTHORITATIVE_BENCHMARK_ROOT = Path(
    r"D:\DATE27\LiDAR-main\LiDAR-main\src\picroute\benchmarks"
)
BENCHMARK_OVERRIDE_ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
# ``toy_example`` is intentionally outside the production crossing3 suite.
# Keep the source benchmark immutable, but do not expose it through --list or
# --all and do not create result directories for it.
EXCLUDED_BENCHMARKS = frozenset({"toy_example"})


@dataclass(frozen=True)
class BenchmarkRef:
    name: str
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_below(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Benchmark escaped the authoritative root: {path}") from error


def discover_benchmarks() -> dict[str, BenchmarkRef]:
    """Return the immutable DATE27 catalog with explicit local replacements.

    There is intentionally no root argument.  Production planning must not
    silently fall back to recursively copied benchmarks or to modified
    snapshots in older experiments.  A YAML placed directly in the local
    benchmark root is an explicit, auditable replacement for a same-named
    authoritative case; nested copies remain ignored.
    """

    root = AUTHORITATIVE_BENCHMARK_ROOT.resolve(strict=True)
    candidates = sorted(
        path.resolve(strict=True)
        for path in root.rglob("*.yml")
        if path.is_file()
    )
    result: dict[str, BenchmarkRef] = {}
    for path in candidates:
        _assert_below(path, root)
        name = path.parent.name if path.parent != root else path.stem
        if name in EXCLUDED_BENCHMARKS:
            continue
        if name in result:
            raise ValueError(f"Duplicate benchmark name {name!r}")
        result[name] = BenchmarkRef(name=name, path=path, sha256=sha256_file(path))
    if not result:
        raise FileNotFoundError(f"No benchmark YAML found below {root}")
    override_root = BENCHMARK_OVERRIDE_ROOT.resolve(strict=True)
    for path in sorted(override_root.glob("*.yml")):
        resolved = path.resolve(strict=True)
        _assert_below(resolved, override_root)
        name = resolved.stem
        if name not in result:
            raise ValueError(
                f"Local benchmark replacement {resolved} has no authoritative "
                f"case named {name!r}"
            )
        result[name] = BenchmarkRef(
            name=name, path=resolved, sha256=sha256_file(resolved)
        )
    return result


def resolve_benchmark(name: str) -> BenchmarkRef:
    catalog = discover_benchmarks()
    try:
        return catalog[name]
    except KeyError as error:
        available = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown benchmark {name!r}; choose one of: {available}") from error
