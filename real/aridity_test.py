"""Does aridity add information to the spread model? (P.Z.'s question, tested without future information.) Covariates of the target cell j, all from gridMET, windows ending the day BEFORE the transition day d0: DM_w = (365/w) * P_w / (Tbar_w + 10) De Martonne index over w = 30, 90, 365 days (log1p; 30/90-day versions with an indicator P_w = 0), VPD on day d0, plus 1000-h dryness D and ERC E as before. Models (pre-registered): WS; WS+DM365; WS+VPD; M1=WS+D+E; M1+DM90; M1+DM365; M1+VPD; M1+VPD+DM365. Evaluation: (i) 7-fold fire-grouped CV on 28 GOFER fires with a fire-cluster bootstrap of the CV log-likelihood gain over the reference; (ii) prospective test: fitted on 2019-2020 fires only, evaluated on the NIFC Dixie transitions. Output aridity.json"""
import json, numpy as np, pandas as pd, xarray as xr, os, pickle
RD = os.environ.get('QCDF_DATA', '../realdata') + '/'
Z = np.load('mf_transitions2.npz', allow_pickle=True)
fsrc = open('multifire_fit.py').read().replace("np.load('mf_transitions.npz')", "np.load('mf_transitions2.npz', allow_pickle=True)")
exec(fsrc[:fsrc.index('MODELS = [')])
lat, lon, day, year = Z['lat'], Z['lon'], Z['day'].astype(str), Z['year']
VAR = dict(pr='precipitation_amount', tmmx='air_temperature', tmmn='air_temperature', vpd='mean_vapor_pressure_deficit')
def series(v, lats, lons, y0, y1):
    ds = [xr.open_dataset(RD + f'{v}_{y}.nc')[VAR[v]] for y in range(y0, y1 + 1)]
    out = [d.sel(lat=xr.DataArray(lats, dims='c'), lon=xr.DataArray(lons, dims='c'), method='nearest') for d in ds]
    da = xr.concat(out, dim='day'); return pd.to_datetime(da.day.values), da.values.astype(float)
def features(lat, lon, day, year):
    F = {k: np.full(len(lat), np.nan) for k in ('dm30', 'dm90', 'dm365', 'z30', 'z90', 'vpd')}
    key = np.round(lat, 5) * 1000 + np.round(lon, 5)
    for y in np.unique(year):
        m = year == y; ul, inv = np.unique(np.c_[lat[m], lon[m]], axis=0, return_inverse=True); inv = inv.ravel()
        days, P = series('pr', ul[:, 0], ul[:, 1], y - 1, y); _, TX = series('tmmx', ul[:, 0], ul[:, 1], y - 1, y)
        _, TN = series('tmmn', ul[:, 0], ul[:, 1], y - 1, y); dv, V = series('vpd', ul[:, 0], ul[:, 1], y, y)
        T = (TX + TN) / 2 - 273.15; di = {d: k for k, d in enumerate(days)}; dvi = {d: k for k, d in enumerate(dv)}
        Pc = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)]); Tc = np.vstack([np.zeros((1, T.shape[1])), np.cumsum(T, 0)])
        idx = np.where(m)[0]
        for r, c in zip(idx, inv):
            k = di[pd.Timestamp(str(day[r]))]
            for w, nm in ((30, '30'), (90, '90'), (365, '365')):
                Pw = Pc[k, c] - Pc[k - w, c]; Tw = (Tc[k, c] - Tc[k - w, c]) / w
                F['dm' + nm][r] = np.log1p((365 / w) * Pw / (Tw + 10))
                if nm != '365': F['z' + nm][r] = float(Pw <= 0.0)
            F["vpd"][r] = V[dvi[pd.Timestamp(str(day[r]))], c]
    return F
if not os.path.exists('aridity_feats.npz'):
    Fm = features(lat, lon, day, year); np.savez('aridity_feats.npz', **Fm)
Fm = dict(np.load('aridity_feats.npz'))
print({k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in Fm.items()}, 'P30=0 share', float(Fm['z30'].mean()), flush=True)
COV.update(A=Fm['dm365'], B=Fm['dm90'], C=Fm['dm30'], V=Fm['vpd'], Y=Fm['z90'])
print('corr', pd.DataFrame(dict(D=dj, E=ej, DM365=Fm['dm365'], DM90=Fm['dm90'], VPD=Fm['vpd'])).corr().round(2).to_string(), flush=True)
MODELS = [('wind+slope', 'WS'), ('WS + De Martonne 365 d', 'WSA'), ('WS + VPD', 'WSV'), ('M1 = WS + dryness + ERC', 'WSDE'),
          ('M1 + De Martonne 90 d', 'WSDEBY'), ('M1 + De Martonne 365 d', 'WSDEA'), ('M1 + VPD', 'WSDEV'), ('M1 + VPD + De Martonne 365 d', 'WSDEVA')]
rng = np.random.default_rng(7); perm = rng.permutation(nf); folds = np.array_split(perm, 7)
res = {}; perfire = {}
for name, use in MODELS:
    llf = np.zeros(nf); Ps, Ys = [], []
    for k in range(7):
        te = folds[k]; tr = np.setdiff1d(np.arange(nf), te); o = fit_eval(use, tr, te)
        P_, Y_ = o['test']['P'], o['test']['y']; Ps.append(P_); Ys.append(Y_)
        fr_ = fire[np.isin(fire, te)]
        for f in te: m_ = fr_ == f; llf[f] = np.sum(Y_[m_] * np.log(P_[m_]) + (1 - Y_[m_]) * np.log(1 - P_[m_]))
    P_, Y_ = np.concatenate(Ps), np.concatenate(Ys)
    full = fit_eval(use, np.arange(nf)); fe = fit_eval(use, np.arange(nf), fe=True)
    res[name] = dict(use=use, cv_ll=float(llf.sum()), cv_auc=auc(P_, Y_), beta=full['beta'], beta_fe=fe['beta'])
    perfire[name] = llf
    print(name, round(llf.sum(), 1), round(res[name]['cv_auc'], 4), {k: round(v, 3) for k, v in full['beta'].items()}, flush=True)
def boot_gain(a, b, B=2000):
    d = perfire[a] - perfire[b]; g = [d[rng.integers(0, nf, nf)].sum() for _ in range(B)]
    return float(d.sum()), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))
gains = {}
for a, b in [('WS + De Martonne 365 d', 'wind+slope'), ('WS + VPD', 'wind+slope'), ('M1 + De Martonne 90 d', 'M1 = WS + dryness + ERC'),
             ('M1 + De Martonne 365 d', 'M1 = WS + dryness + ERC'), ('M1 + VPD', 'M1 = WS + dryness + ERC'), ('M1 + VPD + De Martonne 365 d', 'M1 = WS + dryness + ERC'),
             ('M1 = WS + dryness + ERC', 'wind+slope')]:
    gains[f'{a} vs {b}'] = boot_gain(a, b); print('gain', a, 'vs', b, np.round(gains[f'{a} vs {b}'], 1), flush=True)

if not os.path.exists('nifc_rows_full.pkl'):
    ns = {}; src = open('calibrate.py').read(); exec(src[:src.index("print('transitions'")], ns)
    pickle.dump([dict(d0=r['d0'], d1=r['d1'], Delta=r['Delta'], y=r['y'], feats=r['feats'], fm=r['fm'], erc=r['erc'], j=r['j']) for r in ns['rows']], open('nifc_rows_full.pkl', 'wb'))
NR = pickle.load(open('nifc_rows_full.pkl', 'rb'))
import geopandas as gpd
GD = gpd.read_file('grid_r7.gpkg')
la = np.array([GD.lat.values[r['j']] for r in NR]); lo = np.array([GD.lon.values[r['j']] for r in NR])
dd = np.array([str(pd.Timestamp(r['d0']).date()) for r in NR]); yy = np.full(len(NR), 2021)
if not os.path.exists('aridity_feats_dixie.npz'): np.savez('aridity_feats_dixie.npz', **features(la, lo, dd, yy))
FD = dict(np.load('aridity_feats_dixie.npz'))
train = np.where(summ.year.values <= 2020)[0]; prosp = {}
for name, use in MODELS:
    o = fit_eval(use, train); nm = o['norm']; b = o['beta']
    def raw(c, k, r):
        if c == 'W': return r['feats'][:, 0]
        if c == 'S': return r['feats'][:, 1]
        val = {'D': -r['fm'], 'E': r['erc'], 'A': FD['dm365'][k], 'B': FD['dm90'][k], 'C': FD['dm30'][k], 'V': FD['vpd'][k], 'Y': FD['z90'][k]}[c]
        return np.full(len(r['feats']), val)
    p = []
    for k, r in enumerate(NR):
        eta = o['b0'] + sum(b[c] * (raw(c, k, r) - nm[c][0]) / nm[c][1] for c in use)
        q = 1 / (1 + np.exp(-eta)); p.append(1 - np.prod((1 - q) ** r['Delta']))
    p = np.clip(np.array(p), 1e-9, 1 - 1e-9); yv = np.array([r['y'] for r in NR])
    act = np.array([pd.Timestamp(r['d1']) <= pd.Timestamp('2021-08-05') for r in NR])
    prosp[name] = dict(ll=float(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p))), auc=auc(p, yv), oe=float(yv.sum() / p.sum()),
                       ll_active=float(np.sum((yv * np.log(p) + (1 - yv) * np.log(1 - p))[act])), auc_active=auc(p[act], yv[act]), oe_active=float(yv[act].sum() / p[act].sum()))
    print('prospective Dixie', name, {k: round(v, 3) for k, v in prosp[name].items()}, flush=True)
json.dump(dict(models=res, gains=gains, prospective_dixie=prosp, p30_zero=float(Fm['z30'].mean()), p90_zero=float(Fm['z90'].mean()),
               corr=pd.DataFrame(dict(D=dj, E=ej, DM365=Fm['dm365'], DM90=Fm['dm90'], VPD=Fm['vpd'])).corr().round(3).to_dict()), open('aridity.json', 'w'), indent=1)
