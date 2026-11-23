# Reproducing the paper

Manuscript version: **v19** (`paper/sn/QCDF_v19_sn.tex`). The repository has no version tags: the current contents of https://github.com/FabioB95/qcdf are the release.

There are two levels of reproduction:

1. **Rebuild every number, table and figure from the released raw results.** This takes a few minutes and needs no data download.
2. **Re-run the experiments.** This regenerates the raw results from the public inputs. Times are on a 2-core machine.

## 1. Rebuild the paper from the released results (clean folder)

```bash
git clone https://github.com/FabioB95/qcdf.git qcdf && cd qcdf   # or Download ZIP
pip install -r requirements.txt   # or requirements-lock.txt for the exact versions used
./restore_results.sh     # copies results/<folder>/* next to the scripts that read them
bash paper/build_all.sh  # runs every generator, assembles and assembles the manuscript
```

The last step needs a TeX distribution with `pdflatex`, `bibtex`, `pgfplots` and `quantikz`.

`paper/build_all.sh` does the following, in order:

1. `make_v13.py` (the assembler, unchanged name) assembles the manuscript body from `paper/v13/*.tex`.
2. Each `gen_*.py` replaces its `@@TOKEN@@` placeholders with numbers computed from the result files, and writes the TikZ figures.
3. The build compiles the PDF.

The build reports `0` remaining placeholders when every result file is present.

## 2. Where each table and figure comes from

Numbers refer to the v19 manuscript (main text) and its Supplementary Material (S).

| Paper item | Generator | Raw results | Experiment script |
|---|---|---|---|
| Table 1 (protocol), Table S1 (summary) | text, `real/gen_v11.py`, `real/gen_v12.py` | `v11_tokens.json`, `v12_tokens.json` | — |
| Sky campaign: Table 2, Fig. 2, Table 3 (forecast x rule), Tables S23–S25, Sec. S9.1 | `astro/gen_astro.py`, `real/gen_v11.py`, `real/gen_v12.py`, `real/gen_v14.py`, `real/gen_v15.py`, `real/maps.py` | `astro/astro_full_*.json`, `astro_v11.json`, `astro_v14.json`, `astro_v15.json` | `astro/run_astro.py`, `astro_v11.py`, `astro_v14.py`, `astro_v15.py` |
| Wildfire hindcast: Table 4, Fig. 3, Tables S9, S11, S14, S16 | `real/gen_v8.py`, `real/gen_full.py`, `real/gen_v15.py`, `real/maps.py` | `real/fsP_act.json`, `fsP_late.json`, `fsR.json`, `fs_a/b.json`, `fsDT.json` | `real/fullscale.py` (`QCDF_MODEL=prospective|rolling|oracle`; `QCDF_FIRE=fire_dixie_dt/` for the decision-time domain) |
| Resource grid: Table S20 (units × effectiveness) | `real/gen_grid.py` | `real/fs_grid_*.json` | `real/fs_grid.py` |
| Per-residence DINS anticipation (DINS columns of Table 4 and Table S13, Table S17) | `real/gen_v8.py`, `real/gen_v15.py`, `real/gen_full.py` | `real/dins_v15.json` | `python real/dins_v15.py fsP_act.json fsC.json:fire_caldor/:` |
| Forecast x rule, wildfire (Table S15) and decomposition | `real/gen_v14.py`, `real/gen_v15.py` | `real/xev_*.json`, `sgreedy.json` | `QCDF_MODEL=<model> python real/fire_v14.py xeval xev_<model>.json`, `QCDF_MODEL=prospective python real/fire_v14.py sgreedy sgreedy.json` |
| Spread-model calibration (Tables S5–S8, Fig. S10) | `real/gen_real.py`, `real/gen_full.py`, `real/gen_v8.py`, `real/gen_v15.py` | `real/calib*.json`, `mf_calib.json`, `aridity.json`, `transfer.json` | `real/calibrate.py`, `multifire_fit.py`, `aridity_test.py`, `prospective.py`, `calib_check2.py` |
| Forecast replay (Table S18) | `real/gen_v11.py` | `real/fc_a/b.json`, `fc_members.json` | `real/fc.py` |
| Forward simulation (Fig. S13, Sec. S7.1) | `real/gen_fwd.py`, `real/gen_v13.py` | `real/fwd_a/b.json`, `fwd_ol.json` | `real/fwd.py`, `real/fwd_ol.py` |
| Caldor and Monument (Table S13) | `real/gen_v8.py`, `real/gen_v12.py` | `real/fsC.json`, `fsM.json`, `calib_monument.json` | `real/fullscale.py`, `run_third.sh` |
| Protocol results (Table 5, Tables S19, S21, S22) | `real/gen_v11.py`, `real/gen_v12.py`, `real/gen_v13.py`, `real/gen_v15.py` | `real/ops.json`, `dq.json`, `dq_re.json`, `extra12_a/b.json`, `fireline.json`, `fireline_dissolve.json`, `stats_v11.json` | `real/ops.py`, `dq.py`, `dq_re.py`, `extra12.py`, `fireline.py`, `fireline_dissolve.py`, `stats_v11.py` |
| Small instances: penalty, scaling, TTS (Sec. 5.4, Sec. S5, Figs. S5–S9, Table S4) | `runs/gen_extra.py`, `runs/*fig*.py` | `runs/penalty*.json`, `pen10/12.json`, `tts.json`, `sa_budget.json` | `runs/penalty.py`, `penalty_big.py`, `tts.py`, `sa_budget.py` |
| Binomial time change (Fig. 4) and QAOA inside LNS (Table S27) | `real/gen_q.py`, `real/gen_qlns.py` | `real/lnsq.json`, `astro/lnsq_astro.json`, `lns.json`, `lnsq_qaoa_*.json` | `real/lnsq.py`, `astro/lnsq_astro.py`, `real/lns.py`, `real/lnsq_qaoa.py` |
| Quantum resource estimate (Fig. 5, Table 6) | `real/gen_qres.py` | `real/qres_fire.json`, `qres_astro.json` | `real/qres.py` |
| Mixer classes, threshold-oracle cost, noise (Table S28, Sec. S12.1) | `real/gen_qres2.py` | `real/qres2_fire.json`, `qres2_astro.json` | `real/qres2.py` (`Q2_DMAX=495 Q2_NPER=2 Q2_RESTARTS=2 Q2_PMAX=8`) |
| Sensitivity to the spread model (Table S10) | `real/gen_pert.py` | `real/pert_a/b.json`, `pert_recal.json` | `real/pert_models.py` |
| Toy sequential study (Sec. S10) | `paper/fill_v4.py` | `runs/seq_*.json`, `big*_*.json` | `runs/seq.py`, `seq_big2.py` |
| Hardware (Sec. S11, Table S26) | text | job identifiers in Table S26 | `hardware/hw_ablation.py` |

## 3. Re-running the experiments

All inputs are public. They are downloaded into `realdata/` (set `QCDF_DATA` to change the location):

- wildfire: `real/fetch_data.sh`
- GEFS: `gefs/gefs_fetch.py`
- fireline archive RDS-2025-0011: `real/fireline.py` reads it
- Fermi-LAT catalogues (`gll_psc_v22/v27/v31/v35.fit`): downloaded manually from the Fermi Science Support Center; the extracts used are archived in `results/astro/`

Every wildfire script builds its problem through `real/fullscale.py`, which stops unless `QCDF_MODEL` is set (`prospective`, `rolling`, `oracle`, or `file:<calibration.json>` for a perturbed refit; the retrospective model is `oracle`). Result files by model: `fsP_act`, `fsP_late`, `fsDT`, `fsC`, `fsM`, `fs_grid_*` are prospective; `fsR` is rolling; `fs_a`, `fs_b` are oracle. Apart from these two and the spread-model × rule cross-evaluation, every published wildfire result uses `QCDF_MODEL=prospective`; the commands below set it explicitly.

The run order and approximate times are:

| Step | Command | Time |
|---|---|---|
| Small instances | `runs/pilot.py`, `penalty.py`, `penalty_big.py`, `tts.py`, `sa_budget.py`, `anneal.py`, `seq.py`, `seq_big2.py` | minutes to hours each |
| Wildfire data | `real/fetch_data.sh`, `load_perims.py`, `build_grid.py`, `make_dins.py`, `multifire_build2.py`, `prospective.py`, `build_dt.py`, `build_train.sh`, `recal.py`, `census10.py` | about 1 h |
| Hindcast | `QCDF_MODEL=prospective python fullscale.py main <dates> fsP_act.json`; `QCDF_MODEL=rolling` for `fsR.json` (rolling column), `QCDF_MODEL=oracle` for `fs_a.json`, `fs_b.json` (retrospective) | about 2 min per date |
| Forecast replay | `QCDF_FIRE=fire_dixie_dt/ QCDF_MODEL=prospective python fc.py <dates> fc.json` | about 6 min per date |
| Forward simulation | `fwd.py 0 25 fwd_a.json`, `fwd.py 25 50 fwd_b.json`, then `fwd_ol.py` (the prospective model is fixed inside `fwd.py`) | about 2 h and 5 min |
| Protocol checks | `QCDF_MODEL=prospective python dq.py <dates> dq.json`, and with the same `QCDF_MODEL=prospective`: `dq_re.py`, `ops.py`, `extra12.py`; `stats_v11.py`, `fireline.py`, `fireline_dissolve.py` work from archived plans and need no model; `calib_check2.py` (models set by `CAL_MODELS`), `run_third.sh` (sets them itself) | about 1 h each |
| Sky | `astro/run_astro.py prospective`, `run_astro.py oracle`, `astro_v11.py`, `lnsq_astro.py`; replication with `ASTRO_TRAIN=gll_psc_v22.fit,gll_psc_v27.fit ASTRO_TEST=gll_psc_v27.fit,gll_psc_v31.fit ASTRO_SUFFIX=_rep python run_astro.py prospective` | minutes |
| Thinning and resource estimate | `QCDF_MODEL=prospective python real/lnsq.py`, `QCDF_MODEL=prospective python real/lns.py 2021-07-27,2021-07-31,2021-08-04,2021-07-24 lns.json`, `QCDF_MODEL=prospective python real/lnsq_qaoa.py fire ...`, `astro/lnsq_astro.py`, `QCDF_MODEL=prospective python real/qres.py fire qres_fire.json 2021-08-04`, `real/qres.py astro qres_astro.json ../astro/astro_ham_prospective.pkl`, then `QCDF_MODEL=prospective Q2_DMAX=495 Q2_NPER=2 Q2_RESTARTS=2 Q2_PMAX=8 python qres2.py fire qres2_fire.json` and the same without `QCDF_MODEL` for `qres2.py astro qres2_astro.json` | about 1–2 h each |
| Council checks (v15) | `QCDF_FIRE=fire_dixie_dt/ QCDF_MODEL=prospective python fullscale.py main <10 active dates> fsDT.json`, `real/dins_v15.py`, `astro/astro_v15.py`; spread-model × rule cross-evaluation `QCDF_MODEL=<model> python real/fire_v14.py xeval xev_<model>.json` for each model | about 30 min, minutes, minutes |
| Resource grid (v19) | `QCDF_FIRE=fire_dixie_dt/ QCDF_MODEL=prospective python real/fs_grid.py <10 active dates> fs_grid_a.json [setting indices 0-4]` (K ∈ {10, 20, 40, 80}; η = 0.85, 0.5, 0.3, weather, wind and slope), then `python real/gen_grid.py` (Table S20). The decision-time weather `real/fire_dixie_dt/met.npz` is rebuilt by `real/build_dt.py` (gridMET) and is included in `results/real/fire_dixie_dt/` | about 2 min per date and setting |

Hardware runs need IQM or IBM credentials. Credentials are never stored in the repository.
