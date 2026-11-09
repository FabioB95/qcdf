"""Calibrate a neighbour-ignition (stochastic cellular automaton) fire-spread model on the observed NIFC perimeters of the 2021 Dixie Fire, H3 resolution 7."""
import geopandas as gpd, pandas as pd, numpy as np, h3, xarray as xr, json, math
from scipy.optimize import minimize
RD = '../realdata/'
G = gpd.read_file('grid_r7.gpkg'); P = gpd.read_file('perims_cum.gpkg')
G['ign_date'] = pd.to_datetime(G['ign_date'])
idx = {c: k for k, c in enumerate(G.h3)}
nb = [[idx[x] for x in h3.grid_ring(c, 1) if x in idx] for c in G.h3]
lat, lon = G.lat.values, G.lon.values

met = {}
for v, name in [('fm1000', 'dead_fuel_moisture_1000hr'), ('erc', 'energy_release_component-g'),
                ('vs', 'wind_speed'), ('th', 'wind_from_direction'), ('pr', 'precipitation_amount')]:
    d = xr.open_dataset(RD + f'{v}_2021.nc')[name].sel(day=slice('2021-06-01', '2021-10-01'))
    sub = d.sel(lat=xr.DataArray(lat, dims='c'), lon=xr.DataArray(lon, dims='c'), method='nearest')
    met[v] = sub.values.astype(float)
    days = pd.to_datetime(d.day.values)
dayi = {dd: k for k, dd in enumerate(days)}
np.savez('met.npz', days=days.values.astype('datetime64[D]').astype(str), **met)
elev = G.elev.values.astype(float)

def bearing(i, j):
    dlon = math.radians(lon[j] - lon[i]); la1, la2 = math.radians(lat[i]), math.radians(lat[j])
    x = math.sin(dlon) * math.cos(la2); y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360
DIST = 2.4

dates = list(pd.to_datetime(P.date))
rows = []
for k in range(len(dates) - 1):
    d0, d1 = dates[k], dates[k + 1]; Delta = (d1 - d0).days
    burned = (G.ign_date <= d0).values
    span = [dayi[d0 + pd.Timedelta(days=s)] for s in range(Delta)]
    for j in np.where(~burned)[0]:
        B = [i for i in nb[j] if burned[i]]
        if not B: continue
        y = int(G.ign_date.iloc[j] == d1)
        feats = []
        for i in B:
            th_to = (met['th'][span, i] + 180) % 360
            W = np.mean(met['vs'][span, i] * np.cos(np.radians(bearing(i, j) - th_to)))
            S = (elev[j] - elev[i]) / (DIST * 1000)
            feats.append([W, S])
        rows.append(dict(k=k, d0=d0, d1=d1, Delta=Delta, j=j, y=y, feats=np.array(feats),
                         fm=np.mean(met['fm1000'][span, j]), erc=np.mean(met['erc'][span, j]),
                         pr30=np.sum(met['pr'][dayi[d0] - 30:dayi[d0], j])))
for k in set(r['k'] for r in rows):
    rr=[r for r in rows if r['k']==k]; m=np.mean([-r['fm'] for r in rr]); me=np.mean([r['erc'] for r in rr])
    for r in rr: r['dsp']=-r['fm']-m; r['esp']=r['erc']-me
print('transitions', len(rows), 'positives', sum(r['y'] for r in rows))
train = [r for r in rows if r['d1'] <= pd.Timestamp('2021-08-23')]; test = [r for r in rows if r['d1'] > pd.Timestamp('2021-08-23')]

allW = np.concatenate([r['feats'][:, 0] for r in train]); allS = np.concatenate([r['feats'][:, 1] for r in train])
fm = np.array([r['fm'] for r in train]); er = np.array([r['erc'] for r in train])
ds=np.array([r['dsp'] for r in train]); es=np.array([r['esp'] for r in train])
norm = dict(W=(allW.mean(), allW.std()), S=(allS.mean(), allS.std()), D=(-fm.mean(), fm.std()), E=(er.mean(), er.std()), Q=(ds.mean(), ds.std()), R=(es.mean(), es.std()))

def design(r, use):
    W = (r['feats'][:, 0] - norm['W'][0]) / norm['W'][1]; S = (r['feats'][:, 1] - norm['S'][0]) / norm['S'][1]
    D = (-r['fm'] - norm['D'][0]) / norm['D'][1]; E = (r['erc'] - norm['E'][0]) / norm['E'][1]
    cols = [np.ones_like(W)]
    if 'W' in use: cols.append(W)
    if 'S' in use: cols.append(S)
    if 'D' in use: cols.append(np.full_like(W, D))
    if 'E' in use: cols.append(np.full_like(W, E))
    if 'Q' in use: cols.append(np.full_like(W, (r['dsp'] - norm['Q'][0]) / norm['Q'][1]))
    if 'R' in use: cols.append(np.full_like(W, (r['esp'] - norm['R'][0]) / norm['R'][1]))
    return np.stack(cols, 1)

def pburn(beta, r, use):
    q = 1 / (1 + np.exp(-design(r, use) @ beta))
    return 1 - np.prod((1 - q) ** r['Delta'])

def nll(beta, data, use, l2=1e-3):
    p = np.clip(np.array([pburn(beta, r, use) for r in data]), 1e-9, 1 - 1e-9)
    y = np.array([r['y'] for r in data])
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + l2 * np.sum(beta[1:] ** 2)

def auc(p, y):
    o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

res = {}
for name, use in [('constant', ''), ('wind+slope', 'WS'), ('wind+slope+dryness', 'WSD'), ('full (wind+slope+dryness+ERC)', 'WSDE'),
                  ('wind+slope+relative dryness', 'WSQ'), ('wind+slope+relative dryness+relative ERC', 'WSQR')]:
    m = 1 + len(use)
    best = minimize(nll, np.r_[-3.0, np.zeros(m - 1)], args=(train, use), method='L-BFGS-B')
    pt = np.array([pburn(best.x, r, use) for r in test]); yt = np.array([r['y'] for r in test])
    ptr = np.array([pburn(best.x, r, use) for r in train]); ytr = np.array([r['y'] for r in train])
    res[name] = dict(use=use, beta=best.x.tolist(), nll_train=float(best.fun),
                     ll_test=float(np.sum(yt * np.log(np.clip(pt, 1e-9, 1)) + (1 - yt) * np.log(np.clip(1 - pt, 1e-9, 1)))),
                     auc_train=float(auc(ptr, ytr)), auc_test=float(auc(pt, yt)), n_train=len(train), n_test=len(test),
                     pos_train=int(ytr.sum()), pos_test=int(yt.sum()), k=m)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items() if k != 'beta'}, np.round(best.x, 3))
json.dump(dict(models=res, norm={k: list(map(float, v)) for k, v in norm.items()}, DIST=DIST), open('calib.json', 'w'), indent=1)
