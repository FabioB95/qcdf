#!/usr/bin/env bash
# Downloads the public data used by the Dixie Fire hindcast into $QCDF_DATA (default ../realdata).
set -e
D=${QCDF_DATA:-../realdata}; mkdir -p "$D"; cd "$D"
curl -L -o perimeters.zip "https://zenodo.org/records/15492982/files/DixieFire_2021_perimeters.zip?download=1"
mkdir -p peri && unzip -o -j -q perimeters.zip -d peri
for v in fm1000 erc vs th pr; do curl -L -O "https://www.northwestknowledge.net/metdata/data/${v}_2021.nc"; done
curl -L -O https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_06_tabblock20.zip
curl -L -O https://www2.census.gov/geo/tiger/TIGER2021/PRISECROADS/tl_2021_06_prisecroads.zip
echo "done (elevation is fetched by build_grid.py from the Open-Meteo API and cached in elev.json)"
