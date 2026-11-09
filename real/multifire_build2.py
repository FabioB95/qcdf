"""Multi-fire calibration data: 28 California wildfires 2019-2021 from GOFER (Liu et al. 2024, ESSD; Zenodo 10.5281/zenodo.14642378, GOES hourly progression, CC BY 4.0), H3 resolution 7. For each fire: cells within 10 km of the final perimeter; ignition day = first local day whose end-of-day cumulative perimeter covers >= 50% of the cell. Daily (Delta = 1) neighbour-ignition transitions with gridMET covariates (wind speed/direction, 1000-h fuel moisture, ERC) and elevation (Copernicus GLO-90, AWS open-data tiles). Output: mf_transitions.npz"""
import os, json, time, math, rasterio, numpy as np, pandas as pd, geopandas as gpd, h3, requests, xarray as xr
from shapely.geometry import Polygon
RD = os.environ.get('QCDF_DATA', '../realdata') + '/'
P = gpd.read_file(RD + 'gofer/GOFER/GOFER_Combined/GOFERC_fireProg.shp')
P['tLocal'] = pd.to_datetime(P.tLocal); P['day'] = P.tLocal.dt.floor('D')
ELEV = json.load(open('mf_elev.json')) if os.path.exists('mf_elev.json') else {}

TILES = {}
def _tile(lat, lon):
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    key = (la, lo)
    if key not in TILES:
        name = f"Copernicus_DSM_COG_30_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
        fn = RD + f'dem/{name}.tif'
        if not os.path.exists(fn):
            os.makedirs(RD + 'dem', exist_ok=True)
            r = requests.get(f'https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif', timeout=300); r.raise_for_status()
            open(fn, 'wb').write(r.content)
        TILES[key] = rasterio.open(fn)
    return TILES[key]

def elevation(cells):
    need = [c for c in cells if c not in ELEV]
    for c in need:
        la, lo = h3.cell_to_latlng(c); ds = _tile(la, lo)
        ELEV[c] = float(next(ds.sample([(lo, la)]))[0])
    json.dump(ELEV, open('mf_elev.json', 'w'))
    return np.array([ELEV[c] for c in cells], float)

MET = {}
def met(var, year):
    if (var, year) not in MET:
        name = dict(vs='wind_speed', th='wind_from_direction', fm1000='dead_fuel_moisture_1000hr', erc='energy_release_component-g')[var]
        MET[(var, year)] = xr.open_dataset(RD + f'{var}_{year}.nc')[name]
    return MET[(var, year)]

def bearing(la1, lo1, la2, lo2):
    dlon = np.radians(lo2 - lo1); a1, a2 = np.radians(la1), np.radians(la2)
    x = np.sin(dlon) * np.cos(a2); y = np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

rows = dict(fire=[], t=[], y=[], dj=[], ej=[], npair=[], lat=[], lon=[], day=[], year=[]); pairs = dict(row=[], W=[], S=[])
summary = []
for fid, (fname, F) in enumerate(P.groupby('fname', sort=True)):
    year = int(F.fyear.iloc[0])
    daily = F.sort_values('tLocal').groupby('day').tail(1).sort_values('day')
    final = daily.geometry.iloc[-1]
    reg = gpd.GeoSeries([final], crs=4326).to_crs(3310).buffer(10000).to_crs(4326).iloc[0]
    polys = [reg] if reg.geom_type == 'Polygon' else list(reg.geoms)
    cells = sorted(set(c for pg in polys for c in h3.geo_to_cells(pg, 7)))
    G = gpd.GeoDataFrame(dict(h3=cells), geometry=[Polygon([(b, a) for a, b in h3.cell_to_boundary(c)]) for c in cells], crs=4326).to_crs(3310)
    area = G.area.values; days = list(daily.day); ign = np.full(len(cells), -1)
    D3 = daily.to_crs(3310).geometry.simplify(100).values
    for t, geom in enumerate(D3):
        fr = G.intersection(geom).area.values / area
        new = (ign < 0) & (fr >= 0.5); ign[new] = t
    idx = {c: k for k, c in enumerate(cells)}
    nb = [[idx[x] for x in h3.grid_ring(c, 1) if x in idx] for c in cells]
    lat = np.array([h3.cell_to_latlng(c)[0] for c in cells]); lon = np.array([h3.cell_to_latlng(c)[1] for c in cells])
    elev = elevation(cells)
    sel = lambda v, d: met(v, year).sel(day=d, method='nearest').sel(lat=xr.DataArray(lat, dims='c'), lon=xr.DataArray(lon, dims='c'), method='nearest').values.astype(float)
    npos = 0
    for t in range(len(days) - 1):
        burned = (ign >= 0) & (ign <= t)
        if not burned.any(): continue
        d0 = days[t]; vs, th, fm, er = sel('vs', d0), sel('th', d0), sel('fm1000', d0), sel('erc', d0)
        for j in np.where(~burned)[0]:
            B = [i for i in nb[j] if burned[i]]
            if not B: continue
            r = len(rows['y']); y = int(ign[j] == t + 1); npos += y
            rows['lat'].append(lat[j]); rows['lon'].append(lon[j]); rows['day'].append(str(pd.Timestamp(d0).date())); rows['year'].append(year)
            rows['fire'].append(fid); rows['t'].append(t); rows['y'].append(y); rows['dj'].append(-fm[j]); rows['ej'].append(er[j]); rows['npair'].append(len(B))
            for i in B:
                bt = bearing(lat[i], lon[i], lat[j], lon[j])
                pairs['row'].append(r); pairs['W'].append(vs[i] * math.cos(math.radians(bt - (th[i] + 180) % 360)))
                pairs['S'].append((elev[j] - elev[i]) / 2400.0)
    summary.append(dict(fire=fname, year=year, cells=len(cells), burned=int((ign >= 0).sum()), days=len(days), positives=npos))
    print(summary[-1], 'rows so far', len(rows['y']), flush=True)
np.savez(os.environ.get('MF_OUT', 'mf_transitions.npz'), **{k: np.array(v) for k, v in rows.items()}, **{'p_' + k: np.array(v) for k, v in pairs.items()})
pd.DataFrame(summary).to_csv('mf_summary.csv', index=False)
