#!/usr/bin/env bash
# Rebuild the TWCC evaluation, Tables S2, S3 and S5, the residents totals and all figures from the files in this repository.
set -euo pipefail
cd "$(dirname "$0")"
python3 code/cubes_from_yearly.py
python3 code/v3/twcc.py
python3 code/v3/twcc_folds.py
python3 code/v3/twcc_supp.py
python3 code/v3/twcc_overpass_direct.py
python3 code/osse_edge.py
python3 code/global_exposure_nolook.py
mkdir -p fig/v2
for f in f01_data f02_workflow f03_case f04_us f05_iv f06_robust f07_mech f08_global f09_twcc f10_exposure f11_osse fS1_modis; do
  PYTHONPATH=code/v2 python3 code/v2/$f.py
done
echo "done: outputs in data/, fig/rev/ and fig/v2/"
