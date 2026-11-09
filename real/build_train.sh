#!/bin/bash
# Builds the simulation grids of the 17 training fires (2019-2020) used by the past-only recalibration.
python3 - <<'P' > train_fires.txt
import json; print('\n'.join(json.load(open('calib_prospective.json'))['fires']))
P
while IFS= read -r f; do
  tag=train_$(echo "$f" | tr ' A-Z-' '_a-z_')
  [ -f fire_$tag/met.npz ] || python3 build_fire.py "$f" "$tag" 2>&1 | grep -v Warn | tail -n 2
done < train_fires.txt
echo ALLDONE
