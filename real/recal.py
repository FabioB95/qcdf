"""Recalibration of the prospective spread model on the training fires only (2019-2020, GOFER)."""
import json, glob, sys, numpy as np, pandas as pd, geopandas as gpd, h3, math
from fastsim import _sim
C = json.load(open('calib_prospective.json')); beta = np.array(C['models']['wind+slope']['beta']); nm = C['norm']
HZ, RINGS, R = 10, 8, 200

def bearing(la1, lo1, la2, lo2):
    dlon = np.radians(lo2 - lo1); a1, a2 = np.radians(la1), np.radians(la2)
    x = np.sin(dlon) * np.cos(a2); y = np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def fire_cases(d):
    G = gpd.read_file(d + 'grid_r7.gpkg'); G['ign_date'] = pd.to_datetime(G['ign_date'])
    M = np.load(d + 'met.npz'); mdays = pd.to_datetime(M['days']); mday = {x: k for k, x in enumerate(mdays)}
    idx = {c: k for k, c in enumerate(G.h3)}; NB = [[idx[x] for x in h3.grid_ring(c, 1) if x in idx] for c in G.h3]
    days = sorted(G.ign_date.dropna().unique()); last = pd.Timestamp(days[-1]); out = []
    for T0 in pd.date_range(pd.Timestamp(days[0]), last - pd.Timedelta(days=2), freq='2D'):
        if T0 + pd.Timedelta(days=HZ - 1) > mdays[-1]: break
        b = (G.ign_date <= T0).values
        if b.sum() == 0: continue
        dist = np.full(len(G), 99); fr = list(np.where(b)[0]); dist[fr] = 0
        for r in range(1, RINGS + 1):
            nx = []
            for i in fr:
                for j in NB[i]:
                    if dist[j] == 99: dist[j] = r; nx.append(j)
            fr = nx
        cells = np.where(dist <= RINGS)[0]; loc = {c: k for k, c in enumerate(cells)}
        src, dst = [], []
        for a, i in enumerate(cells):
            for j in NB[i]:
                if j in loc: src.append(a); dst.append(loc[j])
        src, dst = np.array(src, np.int64), np.array(dst, np.int64); gi, gj = cells[src], cells[dst]
        br = bearing(G.lat.values[gi], G.lon.values[gi], G.lat.values[gj], G.lon.values[gj])
        dd = [mday[T0 + pd.Timedelta(days=t)] for t in range(HZ)]
        W = np.stack([M['vs'][x, gi] * np.cos(np.radians(br - (M['th'][x, gi] + 180) % 360)) for x in dd])
        S = (G.elev.values[gj] - G.elev.values[gi]) / (C['DIST'] * 1000)
        lin = beta[0] + beta[1] * (W - nm['W'][0]) / nm['W'][1] + beta[2] * (S - nm['S'][0]) / nm['S'][1]
        ign = G.ign_date.values[cells]
        y = ((ign > np.datetime64(T0)) & (ign <= np.datetime64(T0 + pd.Timedelta(days=HZ)))).astype(int)
        out.append(dict(src=src, dst=dst, lin=lin, b0=b[cells].astype(np.uint8), y=y, un=~b[cells]))
    return out

def predict(case, a, bb, stream=5):
    q = np.ascontiguousarray(1 / (1 + np.exp(-(a + bb * case['lin']))))
    n = len(case['b0']); P = np.zeros((HZ, n)); L = np.zeros(R); F = np.zeros(n, np.int64)
    _sim(case['src'], case['dst'], q, case['b0'], np.ones(n), P, 0.0, R, stream, L, F)
    return F / R

def score(cases, a, bb):
    ll, P, Y = 0.0, [], []
    for c in cases:
        p = np.clip(predict(c, a, bb)[c['un']], 0.5 / (R + 1), 1 - 0.5 / (R + 1)); y = c['y'][c['un']]
        ll += -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)); P.append(p); Y.append(y)
    return ll, np.concatenate(P), np.concatenate(Y)

def slope_int(p, y):
    from scipy.optimize import minimize
    x = np.log(p / (1 - p)); f = lambda b: -np.sum(y * (b[0] + b[1] * x) - np.logaddexp(0, b[0] + b[1] * x))
    b = minimize(f, [0, 1]).x; return float(b[0]), float(b[1])

if __name__ == '__main__':
    dirs = sorted(glob.glob('fire_train_*/')); cases = []
    for d in dirs: cs = fire_cases(d); cases += cs; print(d, len(cs), flush=True)
    print('cases', len(cases), 'cells at risk', sum(int(c['un'].sum()) for c in cases), 'positives', sum(int(c['y'][c['un']].sum()) for c in cases), flush=True)
    grid = []
    for a in (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0):
        for bb in (0.5, 0.75, 1.0, 1.25):
            ll, p, y = score(cases, a, bb); ci, cs = slope_int(p, y)
            grid.append(dict(a=a, b=bb, logloss=ll / len(y), oe=float(y.sum() / p.sum()), cal_int=ci, cal_slope=cs))
            print(grid[-1], flush=True)
    best = min(grid, key=lambda g: g['logloss'])

    for a in np.arange(best['a'] - 0.4, best['a'] + 0.41, 0.2):
        for bb in np.arange(max(0.3, best['b'] - 0.2), best['b'] + 0.21, 0.1):
            ll, p, y = score(cases, float(a), float(bb)); ci, cs = slope_int(p, y)
            grid.append(dict(a=float(a), b=float(bb), logloss=ll / len(y), oe=float(y.sum() / p.sum()), cal_int=ci, cal_slope=cs))
    best = min(grid, key=lambda g: g['logloss']); base = [g for g in grid if g['a'] == 0.0 and g['b'] == 1.0][0]
    print('best', best, 'base', base, flush=True)
    a, bb = best['a'], best['b']
    Cn = json.loads(json.dumps(C)); Cn['models']['wind+slope']['beta'] = [a + bb * beta[0], bb * beta[1], bb * beta[2]]
    Cn['recal'] = dict(a=a, b=bb, train_cases=len(cases), grid=grid, best=best, base=base)
    json.dump(Cn, open('calib_recal.json', 'w'), indent=1)
