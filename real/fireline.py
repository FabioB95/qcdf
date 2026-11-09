"""Real containment lines of the Dixie Fire from the USFS fireline engagement archive (RDS-2025-0011): what the archive records (line types, dates, engagement outcomes), and how the lines relate to the QCDF plans and to the grid cells. Descriptive only: outcomes are classified against the final perimeter and lines are built where the fire was worst, so the archive does not identify the effect of a unit."""
import json, numpy as np, pandas as pd, geopandas as gpd, h3
d = gpd.read_file('../realdata/fireline/Data/FirelineEngagement.gdb', layer='Firelines_Engagement_17_24', where="IncidentName LIKE '%Dixie%' OR IncidentName LIKE '%DIXIE%'", bbox=(-121.8, 39.6, -120.4, 40.6))
d = d[d.Year.astype(str).str.startswith('2021')].copy()
n_rows = len(d)
d['wkb'] = d.geometry.apply(lambda g: g.wkb if g is not None else b'')
d['from_'] = pd.to_datetime(d.GDB_FROM_DATE, errors='coerce', utc=True); d['to_'] = pd.to_datetime(d.GDB_TO_DATE, errors='coerce', utc=True)
first = d.groupby('wkb').from_.min()
d = d.sort_values('to_').groupby('wkb').tail(1).copy(); d['first_mapped'] = d.wkb.map(first)
d['km'] = d.to_crs(3310).length / 1000
out = dict(n_rows_archive=int(n_rows), n_segments=int(len(d)), km_total=float(d.km.sum()))
out['by_type_km'] = d.groupby('FeatureCategory').km.sum().sort_values(ascending=False).round(1).to_dict()
out['by_engagement_km'] = d.groupby('FirelineEngagement').km.sum().sort_values(ascending=False).round(1).to_dict()
dt = pd.to_datetime(d.LineDateTime, errors='coerce', utc=True)
cd = pd.to_datetime(d.CreateDate, errors='coerce', utc=True)
out['frac_with_linedate'] = float(dt.notna().mean()); out['frac_with_createdate'] = float(cd.notna().mean())
out['linedate_range'] = [str(dt.min()), str(dt.max())] if dt.notna().any() else None
out['createdate_range'] = [str(cd.min()), str(cd.max())]
fm = d.first_mapped.dt.tz_convert('America/Los_Angeles').dt.date
out['km_mapped_by_active_end'] = float(d.km[pd.to_datetime(fm) <= pd.Timestamp('2021-08-05')].sum())
out['km_mapped_by_week'] = d.groupby(pd.to_datetime(fm).dt.to_period('W').astype(str)).km.sum().round(0).to_dict()
G = gpd.read_file('grid_r7.gpkg'); G['ign_date'] = pd.to_datetime(G['ign_date'])
dd = d.drop(columns=['wkb']).to_crs(3310); Gm = G.to_crs(3310)
inter = gpd.overlay(dd[['FeatureCategory', 'FirelineEngagement', 'geometry']].explode(index_parts=False), Gm[['h3', 'geometry']], how='intersection', keep_geom_type=False)
inter['km'] = inter.geometry.length / 1000
cell_km = inter.groupby('h3').km.sum()
held = inter[inter.FirelineEngagement.astype(str).str.contains('Held', case=False, na=False)].groupby('h3').km.sum()
out['cells_with_line'] = int((cell_km > 0.1).sum()); out['cells_total'] = int(len(G))
inter2 = gpd.overlay(dd[['first_mapped', 'geometry']].explode(index_parts=False), Gm[['h3', 'geometry']], how='intersection', keep_geom_type=False)
inter2['km'] = inter2.geometry.length / 1000; inter2['fm'] = inter2.first_mapped.dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
R = json.load(open('fsP_act.json')) + json.load(open('fsP_late.json'))
rows = []
for r in R:
    for m in ('QCDF_MILP', 'B4_value_at_risk', 'B1_communities', 'B2_nearest_front'):
        p = r['K']['10']['eval'][m]['plan']
        T0 = pd.Timestamp(r['T0']); w = inter2[(inter2.fm > T0) & (inter2.fm <= T0 + pd.Timedelta(days=10))].groupby('h3').km.sum()
        rows.append(dict(T0=r['T0'], plan=m, frac_line=float(np.mean([cell_km.get(c, 0) > 0.1 for c in p])), km=float(sum(cell_km.get(c, 0) for c in p)),
                         frac_line_10d=float(np.mean([w.get(c, 0) > 0.1 for c in p]))))
P = pd.DataFrame(rows)
out['plan_overlap'] = P.groupby('plan')[['frac_line', 'km', 'frac_line_10d']].mean().round(3).to_dict()
out['plan_overlap_active'] = P[P.T0 <= '2021-08-05'].groupby('plan')[['frac_line', 'frac_line_10d']].mean().round(3).to_dict()

E = G.set_index('h3')
el = E[E.index.isin(cell_km.index)]
out['line_cells_burned_frac'] = float(el.ign_date.notna().mean()); out['line_cells_housing_mean'] = float(el.HOUSING20.mean())
json.dump(out, open('fireline.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:3000])
