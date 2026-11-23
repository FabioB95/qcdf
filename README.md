# QCDF — Quantum Correlated Decision Field

Code and results for the paper

> F. Buffoli, P. Zappa, *The Forecast, Not the Solver: Benchmarking the Quantum Correlated Decision Field on Two Real Spatial Allocation Problems* (manuscript v19, submitted to *Quantum Machine Intelligence*).

Repository: https://github.com/FabioB95/qcdf

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
| `paper/` | Number generators and assembly pipeline (`build_all.sh`); `paper/sn/` holds the final two-column manuscript source in the Springer Nature template (`QCDF_v19_sn.tex`). |
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

From a clean clone, without re-running any experiment:

```bash
git clone https://github.com/FabioB95/qcdf.git && cd qcdf   # or Download ZIP from the repository page
pip install -r requirements.txt
./restore_results.sh
bash paper/build_all.sh
```

`restore_results.sh` copies the raw results from `results/` next to the scripts that read them. `paper/build_all.sh` then regenerates the numbers, tables and figures with the generators. The submitted manuscript is `paper/sn/QCDF_v19_sn.tex` (compile with `pdflatex`, `bibtex`, `pdflatex` twice in `paper/sn/`); its numbers come from the same generators, and the resource grid table from `real/gen_grid.py`.

[`REPRODUCE.md`](REPRODUCE.md) gives:

* the script and result file behind each table and figure;
* the order of the experiment scripts;
* their run times.

## What the results are, and are not

The wildfire benefits are simulated housing exposure avoided under a spread model whose probabilities are poorly calibrated, with an assumed effectiveness of a protection unit; they are not houses saved, and the plans are not validated against real deployments. The quantum solver is benchmarked, not shown to be faster than classical methods; the resource estimate states what a quantum sub-solver would need, not that one exists. The work is a methodological study and a reproducible benchmark.

## Citation

Please cite the paper above. A machine-readable citation will be added on publication.
