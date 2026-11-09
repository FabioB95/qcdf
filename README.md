# QCDF — Quantum Correlated Decision Field

Code and results for the paper

> F. Buffoli, P. Zappa, *Constraint-Preserving Quantum Allocation on Spatial Grids: Static and Sequential Formulations of the Quantum Correlated Decision Field*.

QCDF selects exactly *k* cells of a spatial grid (sky pixels, landscape cells) from per-cell fields and pairwise couplings on the grid graph. The resulting Ising problem with a cardinality constraint is solved by a constraint-preserving quantum ansatz (Dicke state and *XY* mixers), by exact mixed-integer programming, by annealing inside the feasible subspace, and by greedy rules. The same code runs on two independent real problems:

* **Sky** — a follow-up campaign for the unidentified Fermi-LAT sources (4FGL DR2/DR3/DR4, HEALPix *N*<sub>side</sub> = 64, 8,576 cells), planned with DR3 information and evaluated against the sources identified in DR4.
* **Wildfire** — a methodological hindcast of the 2021 Dixie Fire (H3 resolution 7) with replications on the Caldor and Monument fires, where fields and couplings are measured on a spread simulator fitted on earlier fires, and plans are rebuilt from the weather forecasts issued on each decision day.

Every number, table and figure of the paper is produced from the result files in `results/` by the generators in this repository.

## Repository layout

| Folder | Content |
|---|---|
| `out/` | Core quantum code: Dicke subspace simulator and QAOA (`qcdf_scaling.py`), structured Dicke-state circuits (`dicke_scs.py`), data download helpers. |
| `runs/` | Small-instance experiments: astronomical pilot, penalty vs constraint-preserving QAOA, time-to-solution scaling, annealing in the feasible subspace, toy sequential study, and their figure generators. |
| `hardware/` | Hardware ablation script (IQM / IBM back-ends) and the 12-qubit Dixie window. |
| `real/` | Wildfire study: grids, spread models (prospective, rolling, retrospective, recalibrated), fast simulator, full-scale allocation, forward simulation, forecast-only replay, validation protocol checks, thinning-law experiment, maps. |
| `astro/` | Sky study: identification model, beam-coverage Hamiltonian, allocation, protocol checks, replication, thinning-law experiment on the sky Hamiltonian. |
| `gefs/` | Download of the archived GEFS v12 ensemble (NOAA open data on AWS). |
| `paper/` | Paper sources and the build pipeline (`build_all.sh`). |
| `results/` | Raw results of every run (JSON / CSV / NPZ). |

## Installation

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

The hardware scripts additionally need `qiskit-ibm-runtime` (IBM) or `iqm-client` (IQM) and the corresponding credentials, which are never stored in the repository.

## Data

All inputs are public and are downloaded by the scripts into `realdata/` (set `QCDF_DATA` to change the location):

| Input | Script |
|---|---|
| Dixie Fire infrared perimeters (Hatchett, Zenodo 15492982) | `real/fetch_data.sh`, `real/load_perims.py` |
| GOFER fire progression, 28 Californian fires (Zenodo 14642378) | `real/multifire_build2.py`, `real/build_fire.py` |
| gridMET daily weather 2018–2021 | `real/fetch_data.sh` |
| Copernicus GLO-90 DEM (AWS open data) | `real/build_fire.py` |
| Census 2020 blocks, Census 2010 blocks, TIGER 2021 roads and places | `real/fetch_data.sh`, `real/census10.py` |
| CAL FIRE DINS damage records | `real/make_dins.py`, `real/run_third.sh` |
| GEFS v12 ensemble forecasts | `gefs/gefs_fetch.py` |
| HCAI licensed healthcare facilities, 30 June 2021 | `real/ops.py` |
| USFS fireline engagement archive RDS-2025-0011 | `real/fireline.py` |
| Fermi-LAT 4FGL DR1–DR4 catalogues (`gll_psc_v22/v27/v31/v35.fit`) | Fermi Science Support Center |

## Reproducing the paper

Run from inside each folder (scripts use relative paths). Approximate times on a 2-core machine.

1. **Small instances** (`runs/`): `pilot.py`, `penalty.py`, `penalty_big.py`, `tts.py`, `sa_budget.py`, `anneal.py`, `seq.py`, `seq_big2.py` (minutes to a few hours each).
2. **Wildfire data and models** (`real/`): `fetch_data.sh`, `load_perims.py`, `build_grid.py`, `make_dins.py`, `multifire_build2.py`, `prospective.py`, `pert_models.py`, `build_dt.py`, `build_train.sh`, `recal.py`, `census10.py`.
3. **Wildfire allocation**: `QCDF_MODEL=prospective python fullscale.py main <dates> fsP_act.json` (about 2 min per date), the same with `rolling` and `oracle` (retrospective) models, `fwd.py` (forward simulation, about 2 h), `QCDF_FIRE=fire_dixie_dt/ python fc.py <dates> fc.json` (forecast-only replay, about 6 min per date), `run_third.sh` (Monument Fire).
4. **Protocol checks**: `dq.py`, `ops.py`, `extra12.py`, `stats_v11.py`, `calib_check2.py`, `fireline.py`, `lnsq.py`, `analyse_q.py`, `maps.py`.
5. **Sky** (`astro/`): `run_astro.py prospective`, `run_astro.py oracle`, `astro_v11.py`, `lnsq_astro.py`; replication with `ASTRO_TRAIN=gll_psc_v22.fit,gll_psc_v27.fit ASTRO_TEST=gll_psc_v27.fit,gll_psc_v31.fit ASTRO_SUFFIX=_rep python run_astro.py prospective`.
6. **Paper**: `paper/build_all.sh` runs every generator and compiles the PDF.

To rebuild the paper from the released results without rerunning the experiments, copy each folder of `results/` into the folder of the same name (`cp -r results/* .`), decompress the two `.json.gz` files of the thinning-law experiment (`gunzip real/lnsq.json.gz astro/lnsq_astro.json.gz`), and run `paper/build_all.sh`.

## What the results are, and are not

The wildfire benefits are simulated housing exposure avoided under a spread model whose probabilities are poorly calibrated, with an assumed effectiveness of a protection unit; they are not houses saved, and the plans are not validated against real deployments. The quantum solver is benchmarked, not shown to be faster than classical methods. The work is a methodological study and a reproducible benchmark.

## Citation

Please cite the paper above. A machine-readable citation will be added on publication.
