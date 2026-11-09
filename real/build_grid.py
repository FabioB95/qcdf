"""Builds the Dixie Fire H3 resolution-7 grid: ignition dates from perimeters, elevation, Census 2020 housing and road distance."""
import geopandas as gpd, pandas as pd, numpy as np, h3, json, time, requests, xarray as xr
from shapely.geometry import Polygon, Point
RD='../realdata/'
P=gpd.read_file('perims_cum.gpkg')
final=P.geometry.iloc[-1]
region=gpd.GeoSeries([final.buffer(15000)],crs=3310).to_crs(4326).iloc[0]
cells=set()
polys=list(region.geoms) if region.geom_type=='MultiPolygon' else [region]
for poly in polys:
    cells|=set(h3.h3shape_to_cells(h3.LatLngPoly([(la,lo) for lo,la in poly.exterior.coords]),7))
cells=sorted(cells)
lat=np.array([h3.cell_to_latlng(c)[0] for c in cells]); lon=np.array([h3.cell_to_latlng(c)[1] for c in cells])
import os
G=gpd.GeoDataFrame(dict(h3=cells,lat=lat,lon=lon),geometry=[Polygon([(b,a) for a,b in h3.cell_to_boundary(c)]) for c in cells],crs=4326)
Gm=G.to_crs(3310); cent=Gm.geometry.centroid
print('cells',len(G))
if os.path.exists('grid_r7_step1.gpkg'):
    G=gpd.read_file('grid_r7_step1.gpkg'); Gm=G.to_crs(3310); cent=Gm.geometry.centroid; lat=G.lat.values; lon=G.lon.values
if not os.path.exists('grid_r7_step1.gpkg'):

    dates=list(P.date)
    ign=np.full(len(G),np.datetime64('NaT'),'datetime64[ns]'); frac_final=np.zeros(len(G))
    for d,geom in zip(P.date,P.geometry):
        geom=geom.simplify(50); fr=Gm.geometry.intersection(geom).area.values/Gm.geometry.area.values
        newly=(fr>=0.5)&np.isnat(ign); ign[newly]=np.datetime64(d)
    G['ign_date']=ign
    print('burned cells',(~np.isnat(ign)).sum(),flush=True)
    G.to_file('grid_r7_step1.gpkg',driver='GPKG')

elev=json.load(open('elev.json')) if os.path.exists('elev.json') else []
for i in (range(0,len(G),100) if not elev else []):
    la=','.join(f'{x:.5f}' for x in lat[i:i+100]); lo=','.join(f'{x:.5f}' for x in lon[i:i+100])
    for a in range(30):
        try:
            r=requests.get(f'https://api.open-meteo.com/v1/elevation?latitude={la}&longitude={lo}',timeout=60)
            e=r.json()['elevation']; assert len(e)==len(lat[i:i+100]); elev+=e; break
        except Exception as ex:
            print('elev retry',i,a,str(ex)[:80],flush=True); time.sleep(5+5*a)
    else: raise SystemExit('elevation failed')
    time.sleep(1.0)
json.dump(elev,open('elev.json','w'))
G['elev']=elev

bb=tuple(G.total_bounds)
blk=gpd.read_file(f'zip://{RD}tl_2020_06_tabblock20.zip', bbox=bb)
blk=blk.to_crs(4326); bc=blk.geometry.representative_point()
blk['h3']=[h3.latlng_to_cell(p.y,p.x,7) for p in bc]
agg=blk.groupby('h3')[['POP20','HOUSING20']].sum()
G=G.join(agg,on='h3'); G[['POP20','HOUSING20']]=G[['POP20','HOUSING20']].fillna(0)

roads=gpd.read_file(f'zip://{RD}tl_2021_06_prisecroads.zip').to_crs(3310)
roads=roads[roads.intersects(Gm.unary_union.envelope.buffer(30000))]
ru=roads.unary_union
G['road_km']=[cent.iloc[i].distance(ru)/1000 for i in range(len(G))]
G.to_file('grid_r7.gpkg',driver='GPKG')
print(G[['POP20','HOUSING20','elev','road_km']].describe().round(1))
print('pop total',G.POP20.sum())
