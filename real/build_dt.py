"""Decision-time domain for the Dixie Fire."""
import os, numpy as np, pandas as pd, geopandas as gpd, h3, xarray as xr, rasterio
from shapely.geometry import Polygon
RD = '../realdata/'; OUT = 'fire_dixie_dt/'; os.makedirs(OUT, exist_ok=True)
G0 = gpd.read_file('grid_r7.gpkg'); G0['ign_date'] = pd.to_datetime(G0['ign_date']); M0 = np.load('met.npz')
burned = G0.h3[G0.ign_date.notna()].tolist()
cells = set(G0.h3)
for c in burned: cells |= set(h3.grid_disk(c, 12))
new = sorted(cells - set(G0.h3)); print('old', len(G0), 'new', len(new), flush=True)
lat = np.array([h3.cell_to_latlng(c)[0] for c in new]); lon = np.array([h3.cell_to_latlng(c)[1] for c in new])
N = gpd.GeoDataFrame(dict(h3=new, lat=lat, lon=lon), geometry=[Polygon([(b, a) for a, b in h3.cell_to_boundary(c)]) for c in new], crs=4326)
N['ign_date'] = pd.NaT
TILES = {}
def tile(la, lo):
    a, b = int(np.floor(la)), int(np.floor(lo))
    if (a, b) not in TILES:
        nm = f"Copernicus_DSM_COG_30_N{a:02d}_00_W{abs(b):03d}_00_DEM"; TILES[(a, b)] = rasterio.open(RD + f'dem/{nm}.tif')
    return TILES[(a, b)]
N['elev'] = [float(next(tile(a, b).sample([(b, a)]))[0]) for a, b in zip(lat, lon)]
blk = gpd.read_file(f'zip://{RD}tl_2020_06_tabblock20.zip', bbox=tuple(N.total_bounds)).to_crs(4326)
bc = blk.geometry.representative_point(); blk['h3'] = [h3.latlng_to_cell(p.y, p.x, 7) for p in bc]
N = N.join(blk.groupby('h3')[['POP20', 'HOUSING20']].sum(), on='h3'); N[['POP20', 'HOUSING20']] = N[['POP20', 'HOUSING20']].fillna(0)
Nm = N.to_crs(3310); roads = gpd.read_file(f'zip://{RD}tl_2021_06_prisecroads.zip').to_crs(3310)
roads = roads[roads.intersects(Nm.unary_union.envelope.buffer(30000))]; ru = roads.unary_union
N['road_km'] = [p.distance(ru) / 1000 for p in Nm.geometry.centroid]
G = pd.concat([G0[N.columns.intersection(G0.columns).tolist() + [c for c in G0.columns if c not in N.columns]], N], ignore_index=True)
G = gpd.GeoDataFrame(G, geometry='geometry', crs=4326)
G.to_file(OUT + 'grid_r7.gpkg', driver='GPKG')
days = pd.to_datetime(M0['days']); met = {}
for v, nmv in [('fm1000', 'dead_fuel_moisture_1000hr'), ('erc', 'energy_release_component-g'), ('vs', 'wind_speed'), ('th', 'wind_from_direction'), ('pr', 'precipitation_amount')]:
    d = xr.open_dataset(RD + f'{v}_2021.nc')[nmv].sel(day=slice(str(days[0].date()), str(days[-1].date())))
    x = d.sel(lat=xr.DataArray(lat, dims='c'), lon=xr.DataArray(lon, dims='c'), method='nearest').values.astype(float)
    assert x.shape[0] == len(days)
    met[v] = np.concatenate([M0[v], x], 1)
np.savez(OUT + 'met.npz', days=M0['days'], **met)
print('grid', len(G), 'housing new cells', N.HOUSING20.sum(), flush=True)
