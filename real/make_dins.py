"""Download CAL FIRE DINS records of the Dixie Fire and aggregate destroyed structures to H3 res-7 cells."""
import os, requests, pandas as pd, h3
D = os.environ.get('QCDF_DATA', '../realdata'); os.makedirs(D + '/dins', exist_ok=True)
B = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query"
rows, off = [], 0
while True:
    r = requests.get(B, params=dict(where="INCIDENTNAME like '%Dixie%'", outFields="DAMAGE,STRUCTURECATEGORY,COMMUNITY,LATITUDE,LONGITUDE",
                                    returnGeometry='false', resultOffset=off, resultRecordCount=1000, f='json'), timeout=120).json()
    f = r.get('features', []); rows += [x['attributes'] for x in f]; off += len(f)
    if not f or not r.get('exceededTransferLimit'): break
d = pd.DataFrame(rows).dropna(subset=['LATITUDE', 'LONGITUDE']); d.to_csv(D + '/dins/dins_dixie.csv', index=False)
d['h3'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(d.LATITUDE, d.LONGITUDE)]
d['destroyed'] = d.DAMAGE.str.startswith('Destroyed'); d['res'] = d.STRUCTURECATEGORY.isin(['Single Residence', 'Multiple Residence'])
d['res_destroyed'] = d.destroyed & d.res
d.groupby('h3').agg(dins_destroyed=('destroyed', 'sum'), dins_res_destroyed=('res_destroyed', 'sum'), dins_total=('destroyed', 'size')).reset_index().to_csv('dins_cells.csv', index=False)
print(len(d), 'records;', int(d.destroyed.sum()), 'destroyed;', int(d.res_destroyed.sum()), 'residences')
