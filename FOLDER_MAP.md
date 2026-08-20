# lidar_crossing_3 folder contract

`lidar_crossing_3` keeps reviewed source separate from generated experiments.

| Path | Purpose | Policy |
| --- | --- | --- |
| `src/picroute/crossing3/` | HCTP prediction, topology, direction, placement, materialization, and audit | reviewed source |
| `src/picroute/routing/`, `drc/`, `database/`, `utils/` | original LiDAR implementation | frozen; never edited by crossing3 |
| `configs/crossing3.yml` | case-independent HCTP and frozen-backend policy | reviewed source |
| `tools/run_crossing3.py` | the only production launcher | reviewed source |
| `tests/crossing3/` | unit and structural regression tests | reviewed source |
| `docs/crossing3/` | architecture, invariants, and experiment notes | reviewed documentation |
| `work/crossing3/<run-id>/` | staging data, solver domains, attempts, and logs | generated; ignored |
| `results/crossing3/<run-id>/` | final independently audited results only | generated; ignored |

The authoritative benchmark root is read-only and external:

`D:\DATE27\LiDAR-main\LiDAR-main\src\picroute\benchmarks`

No benchmark snapshot, virtual environment, build directory, historical run,
or router binary is copied into this repository.  External frozen tools are
identified by absolute path and SHA-256 in each run manifest.

