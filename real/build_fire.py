"""Build the full-scale case data for another fire from GOFER daily perimeters (replication). Usage: python build_fire.py Caldor caldor -> fire_caldor/grid_r7.gpkg, fire_caldor/met.npz, fire_caldor/dates.txt Grid: H3 res 7 over the final perimeter + 15 km; ignition day = first local day whose end-of-day GOFER perimeter covers >= 50% of the cell; elevation: Copernicus GLO-90 (AWS tiles); housing: Census 2020 blocks; road distance: TIGER 2021; weather: gridMET 2021 at cell centres."""
import os, sys, json, numpy as np, pandas as pd, geopandas as gpd, h3, xarray as xr, rasterio, requests
from shapely.geometry import Polygon
RD = os.environ.get('QCDF_DATA', '../realdata') + '/'
name, tag = sys.argv[1], sys.argv[2]; OUT = f'fire_{tag}/'; os.makedirs(OUT, exist_ok=True)
P = gpd.read_file(RD + 'gofer/GOFER/GOFER_Combined/GOFERC_fireProg.shp'); P = P[P.fname == name].copy()
P['tLocal'] = pd.to_datetime(P.tLocal); P['day'] = P.tLocal.dt.floor('D')
daily = P.sort_values('tLocal').groupby('day').tail(1).sort_values('day')
year = int(P.fyear.iloc[0])
final = daily.to_crs(3310).geometry.iloc[-1]
region = gpd.GeoSeries([final.buffer(15000)], crs=3310).to_crs(4326).iloc[0]
polys = [region] if region.geom_type == 'Polygon' else list(region.geoms)
cells = sorted(set(c for pg in polys for c in h3.geo_to_cells(pg, 7)))
lat = np.array([h3.cell_to_latlng(c)[0] for c in cells]); lon = np.array([h3.cell_to_latlng(c)[1] for c in cells])
G = gpd.GeoDataFrame(dict(h3=cells, lat=lat, lon=lon), geometry=[Polygon([(b, a) for a, b in h3.cell_to_boundary(c)]) for c in cells], crs=4326)
Gm = G.to_crs(3310); area = Gm.area.values
ign = np.full(len(G), np.datetime64('NaT'), 'datetime64[ns]')
for d, geom in zip(daily.day, daily.to_crs(3310).geometry):
    fr = Gm.geometry.intersection(geom.simplify(100)).area.values / area
    new = (fr >= 0.5) & np.isnat(ign); ign[new] = np.datetime64(d)
G['ign_date'] = ign
print(name, 'cells', len(G), 'burned', int((~np.isnat(ign)).sum()), 'days', len(daily), flush=True)

TILES = {}
def tile(la, lo):
    a, b = int(np.floor(la)), int(np.floor(lo))
    if (a, b) not in TILES:
        nm = f"Copernicus_DSM_COG_30_{'N' if a >= 0 else 'S'}{abs(a):02d}_00_{'E' if b >= 0 else 'W'}{abs(b):03d}_00_DEM"
        fn = RD + f'dem/{nm}.tif'
        if not os.path.exists(fn):
            os.makedirs(RD + 'dem', exist_ok=True); r = requests.get(f'https://copernicus-dem-90m.s3.amazonaws.com/{nm}/{nm}.tif', timeout=300); r.raise_for_status(); open(fn, 'wb').write(r.content)
        TILES[(a, b)] = rasterio.open(fn)
    return TILES[(a, b)]
G['elev'] = [float(next(tile(a, b).sample([(b, a)]))[0]) for a, b in zip(lat, lon)]
blk = gpd.read_file(f'zip://{RD}tl_2020_06_tabblock20.zip', bbox=tuple(G.total_bounds)).to_crs(4326)
bc = blk.geometry.representative_point(); blk['h3'] = [h3.latlng_to_cell(p.y, p.x, 7) for p in bc]
G = G.join(blk.groupby('h3')[['POP20', 'HOUSING20']].sum(), on='h3'); G[['POP20', 'HOUSING20']] = G[['POP20', 'HOUSING20']].fillna(0)
roads = gpd.read_file(f'zip://{RD}tl_2021_06_prisecroads.zip').to_crs(3310)
roads = roads[roads.intersects(Gm.unary_union.envelope.buffer(30000))]; ru = roads.unary_union; cent = Gm.geometry.centroid
G['road_km'] = [cent.iloc[i].distance(ru) / 1000 for i in range(len(G))]
G.to_file(OUT + 'grid_r7.gpkg', driver='GPKG')
met = {}
for v, nmv in [('fm1000', 'dead_fuel_moisture_1000hr'), ('erc', 'energy_release_component-g'), ('vs', 'wind_speed'), ('th', 'wind_from_direction'), ('pr', 'precipitation_amount')]:
    d = xr.open_dataset(RD + f'{v}_{year}.nc')[nmv].sel(day=slice(f'{year}-06-01', f'{year}-11-01'))
    met[v] = d.sel(lat=xr.DataArray(lat, dims='c'), lon=xr.DataArray(lon, dims='c'), method='nearest').values.astype(float)
    days = pd.to_datetime(d.day.values)
np.savez(OUT + 'met.npz', days=days.values.astype('datetime64[D]').astype(str), **met)
open(OUT + 'dates.txt', 'w').write(','.join(str(x.date()) for x in daily.day))
print(G[['POP20', 'HOUSING20', 'elev', 'road_km']].sum().round(0).to_dict(), flush=True)
