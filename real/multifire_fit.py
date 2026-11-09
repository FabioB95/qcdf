"""Multi-fire calibration of the neighbour-ignition model (28 GOFER fires, daily steps). P(j ignites on day t+1) = 1 - prod_{i in B_j} (1 - q_ij), q_ij = sigmoid(eta_ij), eta_ij = b0 [+ fire effect] + bW W_ij + bS S_ij + bD D_j + bE E_j. Evaluation: 7-fold cross-validation grouped by fire (held-out fires never seen in fitting); dryness coefficient with a cluster (fire) bootstrap. Output: mf_calib.json"""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
Z = np.load('mf_transitions.npz'); summ = pd.read_csv('mf_summary.csv')
fire, t, y, dj, ej = Z['fire'], Z['t'], Z['y'].astype(float), Z['dj'], Z['ej']
prow, W, S = Z['p_row'], Z['p_W'], Z['p_S']
nf = fire.max() + 1; nrow = len(y)

key = fire * 1000 + t
dfr = pd.DataFrame(dict(k=key, d=dj, e=ej))
drel = dj - dfr.groupby('k').d.transform('mean').values; erel = ej - dfr.groupby('k').e.transform('mean').values
COV = dict(D=dj, E=ej, Q=drel, R=erel)

def design(use, rows_mask, norm):
    pm = rows_mask[prow]
    cols = []
    for c in use:
        if c == 'W': x = W[pm]
        elif c == 'S': x = S[pm]
        else: x = COV[c][prow[pm]]
        cols.append((x - norm[c][0]) / norm[c][1])
    X = np.stack(cols, 1) if cols else np.zeros((pm.sum(), 0))
    return X, pm

def fit_eval(use, train_fires, test_fires=None, fe=False, l2=1e-3, mult=None):
    tr = np.isin(fire, train_fires)
    wr = np.ones(tr.sum()) if mult is None else np.array([mult[f] for f in fire[tr]], float)
    norm = {}
    for c in use:
        x = W[tr[prow]] if c == 'W' else S[tr[prow]] if c == 'S' else COV[c][tr]
        norm[c] = (x.mean(), x.std() + 1e-12)
    X, pm = design(use, tr, norm)
    rows = prow[pm]; ridx = np.unique(rows, return_inverse=True)[1]
    R = tr.sum(); yy = y[tr]; ff = fire[prow[pm]]
    fmap = {f: k for k, f in enumerate(sorted(set(train_fires)))}; fi = np.array([fmap[f] for f in ff])
    nfe = len(fmap) if fe else 1
    def unpack(th): return th[:nfe], th[nfe:]
    def f(th):
        a, b = unpack(th); eta = (a[fi] if fe else a[0]) + X @ b
        sp = np.logaddexp(0, eta); q = 1 / (1 + np.exp(-eta))
        L = -np.bincount(ridx, sp, R)
        P = np.clip(-np.expm1(L), 1e-12, 1)
        ll = np.sum(wr * (yy * np.log(P) + (1 - yy) * L)) - l2 * np.sum(b ** 2)
        g_r = wr * ((1 - yy) - yy * np.exp(L) / P)
        geta = -q * g_r[ridx]
        ga = np.bincount(fi, geta, nfe) if fe else np.array([geta.sum()])
        gb = X.T @ geta - 2 * l2 * b
        return -ll, -np.concatenate([ga, gb])
    th0 = np.concatenate([np.full(nfe, -3.0), np.zeros(X.shape[1])])
    r = minimize(f, th0, jac=True, method='L-BFGS-B')
    a, b = unpack(r.x); out = dict(b0=float(np.mean(a)), beta=dict(zip(use, map(float, b))), nll=float(r.fun), n=int(R), norm={k: list(map(float, v)) for k, v in norm.items()})
    if test_fires is not None:
        te = np.isin(fire, test_fires); Xt, pmt = design(use, te, norm)
        rt = np.unique(prow[pmt], return_inverse=True)[1]; Rt = te.sum()
        eta = np.mean(a) + Xt @ b
        L = -np.bincount(rt, np.logaddexp(0, eta), Rt); P = np.clip(-np.expm1(L), 1e-12, 1 - 1e-12)
        out['test'] = dict(P=P, y=y[te])
    return out

def auc(p, yv):
    s = pd.Series(p).rank().values; n1 = yv.sum(); n0 = len(yv) - n1
    return float((s[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

MODELS = [('constant', ''), ('wind+slope', 'WS'), ('+ dryness', 'WSD'), ('+ ERC', 'WSE'), ('+ dryness + ERC', 'WSDE'),
          ('+ relative dryness', 'WSQ'), ('+ relative dryness + relative ERC', 'WSQR')]
rng = np.random.default_rng(7); perm = rng.permutation(nf); folds = np.array_split(perm, 7)
res = {}
for name, use in MODELS:
    Ps, Ys = [], []
    for k in range(7):
        te = folds[k]; tr = np.setdiff1d(np.arange(nf), te)
        o = fit_eval(use, tr, te); Ps.append(o['test']['P']); Ys.append(o['test']['y'])
    P = np.concatenate(Ps); Yv = np.concatenate(Ys)
    full = fit_eval(use, np.arange(nf))
    fe = fit_eval(use, np.arange(nf), fe=True) if use else None
    res[name] = dict(use=use, beta=full['beta'], b0=full['b0'], beta_fe=fe['beta'] if fe else {},
                     cv_ll=float(np.sum(Yv * np.log(P) + (1 - Yv) * np.log(1 - P))), cv_ll_per=float(np.mean(Yv * np.log(P) + (1 - Yv) * np.log(1 - P))),
                     cv_auc=auc(P, Yv), n=int(len(Yv)), pos=int(Yv.sum()))
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items() if k not in ('beta', 'beta_fe')},
          {k: round(v, 3) for k, v in full['beta'].items()}, 'FE:', {k: round(v, 3) for k, v in res[name]['beta_fe'].items()}, flush=True)

only = '--boot' in __import__('sys').argv
boot = {'WSD': [], 'WSD_fe': [], 'WSE': [], 'WSDE_D': [], 'WSDE_E': []}
for bb in range(200):
    sf = rng.choice(nf, nf, replace=True); u, cnt = np.unique(sf, return_counts=True); m = dict(zip(u, cnt))
    boot['WSD'].append(fit_eval('WSD', u, mult=m)['beta']['D']); boot['WSE'].append(fit_eval('WSE', u, mult=m)['beta']['E'])
    boot['WSD_fe'].append(fit_eval('WSD', u, fe=True, mult=m)['beta']['D'])
    o = fit_eval('WSDE', u, mult=m)['beta']; boot['WSDE_D'].append(o['D']); boot['WSDE_E'].append(o['E'])
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
print('bootstrap 95% CI', ci)

per = []
for f in range(nf):
    if y[fire == f].sum() >= 50:
        o = fit_eval('WSD', [f]); per.append(dict(fire=summ.fire.iloc[f], pos=int(y[fire == f].sum()), **{k: v for k, v in o['beta'].items()}))
print(pd.DataFrame(per).round(3).to_string())
json.dump(dict(models=res, boot_ci=ci, per_fire=per, n_fires=int(nf), n_rows=int(nrow), n_pos=int(y.sum())), open('mf_calib.json', 'w'), indent=1)
