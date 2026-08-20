"""Crossing-only planning for the third LiDAR experiment.

This package is deliberately upstream of detailed routing.  It may predict and
place crossing PCells, but it must not modify LiDAR's router, DRC, post-process,
or GDS implementation.
"""

from .benchmark_source import AUTHORITATIVE_BENCHMARK_ROOT, discover_benchmarks
from .planner import plan_case

__all__ = ["AUTHORITATIVE_BENCHMARK_ROOT", "discover_benchmarks", "plan_case"]

