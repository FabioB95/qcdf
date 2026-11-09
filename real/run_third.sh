#!/bin/bash
# Third wildfire (Monument Fire 2021): grid, damage records, decision dates, allocation and calibration.
set -e
[ -f fire_monument/met.npz ] || python3 build_fire.py Monument monument
python3 - <<'P'
import requests, pandas as pd, h3
B = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query"
rows, off = [], 0
while True:
    r = requests.get(B, params=dict(where="INCIDENTNAME like '%Monument%' AND INCIDENTSTARTDATE >= DATE '2021-07-01' AND INCIDENTSTARTDATE < DATE '2021-12-31'",
                                    outFields="DAMAGE,STRUCTURECATEGORY,INCIDENTNAME,LATITUDE,LONGITUDE", returnGeometry='false',
                                    resultOffset=off, resultRecordCount=1000, f='json'), timeout=120).json()
    f = r.get('features', []); rows += [x['attributes'] for x in f]; off += len(f)
    if not f or not r.get('exceededTransferLimit'): break
d = pd.DataFrame(rows)
if len(d):
    d = d.dropna(subset=['LATITUDE', 'LONGITUDE']); d['h3'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(d.LATITUDE, d.LONGITUDE)]
    d['destroyed'] = d.DAMAGE.str.startswith('Destroyed'); d['res_destroyed'] = d.destroyed & d.STRUCTURECATEGORY.isin(['Single Residence', 'Multiple Residence'])
    d.groupby('h3').agg(dins_destroyed=('destroyed', 'sum'), dins_res_destroyed=('res_destroyed', 'sum'), dins_total=('destroyed', 'size')).reset_index().to_csv('fire_monument/dins_cells.csv', index=False)
print('DINS Monument records', len(d), 'destroyed', int(d.destroyed.sum()) if len(d) else 0)
days = open('fire_monument/dates.txt').read().split(',')
dec = days[1:25:2][:12]; open('fire_monument/decision_dates.txt', 'w').write(','.join(dec)); print('dates', dec)
P
D=$(cat fire_monument/decision_dates.txt)
QCDF_FIRE=fire_monument/ QCDF_MODEL=prospective python3 fullscale.py main $D fsM.json
python3 - <<'P'
import json; json.dump([{'T0': d} for d in open('fire_monument/decision_dates.txt').read().split(',')], open('fsM_dates.json', 'w'))
P
QCDF_FIRE=fire_monument/ CAL_DATES=fsM_dates.json CAL_MODELS=prospective CAL_OUT=calib_monument.json python3 calib_check2.py
echo THIRD_DONE
