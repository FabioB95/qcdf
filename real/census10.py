"""Census 2010 housing units per H3 cell of the Dixie grid: the value field that was public before the 2021 decisions."""
import geopandas as gpd, h3, pandas as pd
G = gpd.read_file('grid_r7.gpkg')
b = gpd.read_file('zip://../realdata/census2010/tabblock2010_06_pophu.zip', bbox=tuple(G.total_bounds)).to_crs(4326)
p = b.geometry.representative_point(); b['h3'] = [h3.latlng_to_cell(q.y, q.x, 7) for q in p]
a = b.groupby('h3')[['HOUSING10', 'POP10']].sum()
G = G.join(a, on='h3'); G[['HOUSING10', 'POP10']] = G[['HOUSING10', 'POP10']].fillna(0)
G[['h3', 'HOUSING10', 'POP10', 'HOUSING20']].to_csv('housing10.csv', index=False)
print(G.HOUSING10.sum(), G.HOUSING20.sum(), G[['HOUSING10', 'HOUSING20']].corr().iloc[0, 1])
