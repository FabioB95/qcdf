"""Reads the Dixie Fire infrared perimeters and writes the cumulative daily burned area (perims_cum.gpkg)."""
import geopandas as gpd, glob, os, re, pandas as pd
from shapely.ops import unary_union
D='../realdata/peri'
by={}
for f in sorted(glob.glob(D+'/*.shp')):
    d=os.path.basename(f)[:8]
    g=gpd.read_file(f)
    if g.crs is None: g=g.set_crs(4326)
    g=g.to_crs(3310); g=g[g.geometry.notna()]
    by.setdefault(d,[]).append(unary_union(g.geometry.buffer(0)))
rows=[]; cum=None
for d in sorted(by):
    geom=unary_union(by[d]); cum=geom if cum is None else unary_union([cum,geom])
    rows.append(dict(date=pd.to_datetime(d), area_km2=geom.area/1e6, cum_km2=cum.area/1e6, geometry=cum))
G=gpd.GeoDataFrame(rows,crs=3310)
print(G[['date','area_km2','cum_km2']].to_string())
G.to_file('perims_cum.gpkg',driver='GPKG')
