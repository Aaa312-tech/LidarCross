# crossing3 runbook

## Environment

The verified runs use:

```text
D:\xprogram\lidar_crossing\.venv-crossing-production\Scripts\python.exe
```

The standalone dependency contract is `requirements-crossing3.txt`.  Do not
copy that virtual environment, the PIC-DB build, or old experiment work trees
into this repository.

## Commands

From `D:\DATE27\lidar_crossing_3`:

```powershell
& 'D:\xprogram\lidar_crossing\.venv-crossing-production\Scripts\python.exe' `
  tools\run_crossing3.py --list

& 'D:\xprogram\lidar_crossing\.venv-crossing-production\Scripts\python.exe' `
  tools\run_crossing3.py --all --run-id frontend-all-hctp-v4

& 'D:\xprogram\lidar_crossing\.venv-crossing-production\Scripts\python.exe' `
  tools\run_crossing3.py --case clements_8x8 --backend `
  --run-id strict-c8-hctp-v3
```

Run tests with:

```powershell
& 'D:\xprogram\lidar_crossing\.venv-crossing-production\Scripts\python.exe' `
  -m unittest discover -s tests\crossing3 -v
```

## Run directory contract

Each case uses numbered stages:

1. `01_input`: normalized private copy plus source hash and original placement manifest.
2. `02_guides`: coarse channel graph paths.
3. `03_topology`: crossing events, parity, braid stages, and capacity motifs.
4. `04_pcell`: exact real-PCell GDS and geometry manifest.
5. `05_placement`: selected centers/states and MILP audit.
6. `06_directions`: per-parent-net entry/exit state solution.
7. `07_materialized`: fixed crossing instances and split routing netlist.
8. `08_audit`: source immutability, geometry, topology, parity, and port incidence.
9. `09_backend/attempt_NN`: converter, frozen native router, DRC, renderer,
   continuity gate, and explicit no-good feedback.

The launcher refuses an existing run id.  Failed backend attempts remain only
under `work/`; only a fully accepted attempt creates a `results/` directory.

## Interpretation

`FRONTEND_PASS` means HCTP prediction, placement, materialization, and all
frontend invariants passed.  It does **not** mean detailed routing passed.
`ACCEPTED` additionally requires frozen strict routing, clean DB DRC, strict
GDS generation, and clean route/GDS continuity.  `STRICT_BACKEND_FAIL` is a
valid experiment result and must not be repackaged as accepted.
