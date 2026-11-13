# Crest-scale fair-weather bias of optical flood records and the timing-weighted crest correction (TWCC)

Code and minimal data for the paper

> Feng, X., Du, J., Chai, B., Lombardo, L. Predicting and correcting the crest-scale fair-weather bias of optical flood
> records from storm timing (submitted).

The repository holds the derived crest tables, daily GOES-16 clear-sky series, rating-curve fits, cross-validation
assignments, the TWCC code and the code and source data of every figure. Raw inputs (USGS NWIS records, GOES-16, MODIS,
MRMS, Caravan, Sentinel-2, HAND and population grids) are public and are listed in Section 6 with the scripts that
download them.

All paths below are relative to the repository root. Run every script from the root (`python3 code/...`); figure
scripts need `PYTHONPATH=code/v2`. `./reproduce.sh` runs the whole quick start below, rebuilding the TWCC evaluation,
Tables S2, S3 and S5, the residents totals and all 12 figures (Figs. 1-11 and Fig. S1, written to `fig/v2/`).

## 1. Quick start

```bash
pip install -r requirements.txt                 # tested with Python 3.13 and the versions listed there
python3 code/cubes_from_yearly.py              # daily GOES-16 clear-sky cubes data/terra.npy, aqua.npy, days.npy (2018-2024) from data/clear_YYYY.npz
python3 code/v3/twcc.py                        # TWCC, nested by gauge: data/v3_twcc*.parquet, v3_twcc_laws.npy, v3_exposure.parquet (about 3 min on 4 cores)
python3 code/v3/twcc_folds.py                  # writes the outer and inner fold of every crest: data/twcc_folds_daily.csv, twcc_folds_storm.csv
python3 code/v3/twcc_supp.py                   # Table S2 and the interval coverage and widths by area class
python3 code/v3/twcc_overpass_direct.py        # Table S3 and Section S2 depth fractions: 15-minute stage at the overpass
python3 code/osse_edge.py                      # Table S5: duration-revisit curve with and without lower-bound durations
python3 code/global_exposure_nolook.py         # residents with no-look crests counted as unseen
for f in f01_data f02_workflow f03_case f04_us f05_iv f06_robust f07_mech f08_global f09_twcc f10_exposure f11_osse fS1_modis; do
  PYTHONPATH=code/v2 python3 code/v2/$f.py; done   # Figs. 1-11 and Fig. S1 into fig/v2/
```

The scripts that download raw inputs (`cloudcube.py`, `case_goes.py`, `mrms_5day.py`, `mrms_feat.py`, `s2_scl*.py`,
`s2_water.py`, `hand_*.py`, `iv_extent.py`, `recon_map_data.py`) additionally need `netCDF4`, `rasterio`, `pyproj`,
`eccodes` and `mgrs`, and network access to the public archives listed in Section 6. Their outputs are included, so
they do not need to be rerun to reproduce the results.

## 2. TWCC and its cross-validation

`code/v3/twcc.py` implements Eqs. (1)-(4) of Section 3.3.

- Image branch: `ln d_I = ln d_look + f(tau, A, month, lat, lon)`; prior branch: `ln d_P = g(A, month, lat, lon)`
  (gradient-boosted regressors, `random_state=0`).
- Fusion: `ln d = w ln d_I + (1 - w) ln d_P`, with the weight that minimises the mean squared log error, computed from
  uncentred moments of the inner held-out errors in area-class x wait cells and summarised by
  `w = 1 / [1 + ((tau + 1) / T_w)^k]`, `T_w = alpha A^beta` (fitted as a logistic law in log10 A and ln(tau + 1)).
- Intervals: cross-validated residual-quantile intervals. Inside each outer training set, inner folds give held-out
  log errors of the fused estimate; their 10/90% and 5/95% quantiles by area class set the 80% and 90% bounds of the
  outer held-out crests. Coverage is measured on the outer held-out gauges, each crest counted once.
- Folds: `GroupKFold(5)` grouped by gauge, outer and inner, deterministic for the row order of `data/rev_recon.parquet`
  (daily evaluation, 13,711 crests) and `data/rev_recon_storm.parquet` (radar storm-day timing, 6,401 crests).
  `data/twcc_folds_daily.csv` and `data/twcc_folds_storm.csv` list `outer` (0-4) and `inner_in_outer0..4` (the inner
  fold of the crest when it is in the training set of that outer fold, -1 when it is the outer test fold). No gauge
  appears in more than one outer fold.

Output columns of `data/v3_twcc.parquet` (depths in feet above the pre-event base):
`dpk` gauged crest depth (target), `d1` first-look depth, `lag` wait tau (days), `e_look` first look, `e_img` image branch,
`e_pri` prior branch, `e_tw` fusion with cell weights, `e_law` TWCC (law weights, the headline estimator), `w`, `w_law`
weights, `lo80 hi80 lo90 hi90` interval bounds, `c80 c90` coverage flags, `wd80` ratio of the upper to the lower 80% bound.

## 3. Overpass-stage test (Table S3) and first-look depth fractions (Section S2)

`code/v3/twcc_overpass_direct.py`, output `data/v3_twcc_overpass_direct.parquet` and `fig/rev/twcc_overpass_direct.csv`:
315 evaluated crests among the 392 fifteen-minute crests (at least 50 stage values and a crest rise of 0.3 ft or more,
as in `iv_ana.py`) with a clear Terra or Aqua overpass on or after the crest day.

- `hb`: pre-event base, the lowest 15-minute stage in the 72 h before the crest.
- `d_pk = h(t_crest) - hb` (target) and `d_ov = h(t_overpass) - hb` (input) come from the same 15-minute record, so
  input and target share one datum.
- `d_day`: mean 15-minute stage over the local day of that overpass minus `hb` (daily input on the same datum).
- `f`, the law weight and the prior are the stored held-out values of the main evaluation (gauges never seen in
  training); `look_d_ov`, `img_d_ov`, `tw_d_ov` and `look_d_day`, `img_d_day`, `tw_d_day` are the first look, image branch
  and TWCC for the two inputs. `after` flags first looks after the crest; `g3` is the three-class area group.
- Inputs below 0.05 ft are floored; the script reports 0.01, 0.1 and 0.2 ft floors, gauge-bootstrap intervals and the numbers of
  crests within 25% for each input.
- First-look depth fractions of Section S2: `f_rated` (rated daily flow, `d1 / dpk` of the main evaluation), `f_day` (mean
  15-minute stage of the day) and `f_ov` (stage at the overpass), with medians by area class in
  `fig/rev/overpass_depth_fractions.csv`. `f_ov_rb` and `f_day_rb` measure the same 15-minute stages from the rated base
  (annual-peak gauge height in `data/usgs/peaks.parquet` minus the rated crest depth), summarised in
  `fig/rev/overpass_depth_fractions_rated_base.csv`. The script also prints the share of 15-minute crest stages equal to the
  annual-peak gauge height, the offset between the two bases and the ratio of the two crest depths quoted in Section S4.

## 4. Duration sensitivity (Table S5)

`code/osse_edge.py` flags crests whose half-extent run touches the start (`edge_start`) or end (`edge_end`) of the
15-minute window (`data/osse_edge.parquet`, 390 crests, 41 flagged) and refits the duration-revisit curve, the
held-out R2, the D50-area exponent and the revisit required for 80% capture on the 349 unflagged crests
(`fig/rev/osse_edge_sensitivity.csv`). Fig. 11a draws the flagged durations as open symbols.

## 5. Map from the paper to code and data

| Paper | Script(s) | Main outputs |
|---|---|---|
| 2.1 Gauged crests, Table S1 | `rating.py`, `rating_fit.py`, `ana1.py`, `rev_quick.py` | `data/main_sample.parquet` (17,033 crests), `data/stage_first.parquet`, `fig/rev/counts_*.csv` |
| 2.2 Clear-sky records, S1 | `cloudcube.py`, `cubes_from_yearly.py`, `match.py`, `coarsen_test.py`, `s2_scl*.py` | `data/clear_YYYY.npz`, `data/clim_month.npy`, `data/matched.parquet`, `data/s2/*.parquet` |
| 2.3 Rainfall, terrain, population, S3 | `mrms_5day.py`, `mrms_feat.py`, `hand_pop*.py`, `hand_hyp.py`, `hand_delta.py`, `hand_radius.py`, `s2_water.py` | `data/mrms_*.parquet`, `data/hand_pop*.parquet`, `data/global/hand_pop_caravan.parquet` |
| 3.1, 4.1 Crest visibility, 15-minute capture | `iv_ana.py`, `iv_extent.py`, `iv_null.py` | `data/iv_capture*.parquet`, `data/iv_null.parquet` |
| 3.2, 4.2 Storm timing and clearing, worldwide | `mech.py`, `mech_tests.py`, `global_ana.py`, `global_dedup.py`, `global_mech2.py` | `data/mech.parquet`, `data/global/caravan_ana.parquet`, `fig/rev/mech_*.csv` |
| 3.3, 4.1, 4.3 TWCC, S2, S4, Tables S2-S3 | `recon_tests.py`, `recon_tests2.py`, `v3/twcc.py`, `v3/twcc_supp.py`, `v3/twcc_folds.py`, `v3/twcc_overpass_direct.py` | `data/rev_recon*.parquet`, `data/v3_twcc*.parquet`, `data/twcc_folds_*.csv` |
| 4.3 Residents | `exposure.py`, `global_exposure.py`, `global_exposure_nolook.py` | `data/exposure.parquet`, `data/v3_exposure.parquet`, `data/global/exposure_caravan.parquet` |
| 3.4, 4.4 Observing-system simulation, S5, Tables S4-S5 | `osse.py`, `osse_tests.py`, `osse_edge.py` | `data/osse_*.parquet`, `fig/rev/osse_*.csv` |

Event and pairing tables: `data/usgs/events.csv` (all annual-peak events of candidate gauges), `data/usgs/peaks.parquet`
(annual peaks used for rating curves), `data/usgs/windows.parquet` (daily discharge around each crest),
`data/usgs/iv.parquet` (15-minute stage, parameter 00065, and discharge, 00060, around the crests with instantaneous records), `data/usgs/sites.csv` (site attributes from NWIS, keyed by site number);
`data/global/caravan_matched_dedup.parquet` and `caravan_ana.parquet` (Caravan crests with MODIS clear-sky fractions);
`data/s2/s2_scl.parquet` and `s2_scl_classes_valid.parquet` (Sentinel-2/GOES-16 scene pairs, 8,800 valid scenes).

| Figure | Script | Input tables |
|---|---|---|
| 1 | `v2/f01_data.py` | `main_sample`, `clim_month.npy`, `global/caravan_ana`, `global/clim_*.npy`, `landmask_1deg.npy` |
| 2 | `v2/f02_workflow.py` (+ `f02_icons.py`, `f02_terrain.py`) | schematic |
| 3 | `v2/f03_case.py` | `terra.npy`, `aqua.npy`, `case_goes.npz`, `usgs/iv`, `usgs/events` |
| 4 | `v2/f04_us.py` | `main_sample`, `stage_first`, `extent_first`, `fig/fig1_bins.csv` (from `fig1.py`) |
| 5 | `v2/f05_iv.py` | `iv_capture_ext`, `iv_null` |
| 6 | `v2/f06_robust.py` | `main_sample`, `mrms_feat`, `s2/s2_scl`, `fig/fig1_bins.csv` |
| 7 | `v2/f07_mech.py` | `mech`, `fig/fig3_*.csv` (from `fig3.py`), `fig/rev/mech_*_US.csv` |
| 8 | `v2/f08_global.py` | `global/*`, `fig/fig4_*.csv` (from `global_mech2.py`), `fig/rev/mech_*_global.csv` |
| 9 | `v2/f09_twcc.py` | `rev_recon*`, `recon_cv`, `recon_map.npz`, `v3_twcc*`, `v3_exposure` |
| 10 | `v2/f10_exposure.py` | `v3_exposure`, `global/exposure_caravan`, `fig/exposure_global.csv`, `fig/rev/hand_*.csv` |
| 11 | `v2/f11_osse.py` | `osse_*`, `osse_edge`, `fig/rev/osse_*.csv` |
| S1 | `v2/fS1_modis.py` | `global/usgs_goes_modis`, `fig/rev/S1_subset.csv` |

## 6. Public sources of the raw inputs

USGS annual peaks, daily values and 15-minute instantaneous values (NWIS); GOES-16 ABI L2 clear-sky mask (ACMC, NOAA
open data on AWS); MODIS Terra and Aqua daily cloud fraction at 1 degree; Caravan and its extensions; MRMS
multi-sensor precipitation; ERA5-Land; Copernicus GLO-30 HAND; Meta High Resolution Settlement Layer; Sentinel-2 L2A
scene classification (cloud-optimised GeoTIFFs on AWS open data). Data citations are in the reference list of the paper.

## 7. Notes on the data files

- `data/clear_YYYY.npz` hold the daily GOES-16 clear-sky fractions at the Terra and Aqua overpass hours on the 0.1 degree
  grid (uint8); `code/cubes_from_yearly.py` joins them into the cubes read by the analysis and figure scripts.
- `data/goes_1deg.npz` is not included; `code/coarsen_test.py` writes it from the daily cubes.
- Station names are left out of `data/usgs/sites.csv`; NWIS lists them by site number.
- Natural Earth boundaries (`data/ne_50m_*.geojson`, public domain) keep their geometry and English attributes only.

## 8. Licence

Code: MIT (`LICENSE`). Data files in `data/` and `fig/`: CC BY 4.0 (`LICENSE-DATA`), except the Natural Earth files,
which are in the public domain, and records derived from USGS, NOAA and NASA products, which remain subject to the
terms of their providers.
