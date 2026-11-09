"""Astronomical counterparts of the wildfire validation protocol (v11)."""
import sys, json, math, numpy as np, healpy as hp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
import astro_full as AF

rng = np.random.default_rng(7)
d2, n2, X2, y2 = AF.transition('gll_psc_v27.fit', 'gll_psc_v31.fit')
d3, n3, X3, y3 = AF.transition('gll_psc_v31.fit', 'gll_psc_v35.fit')
ok2 = np.isfinite(X2).all(1); ok3 = np.isfinite(X3).all(1)
def fit(X, y):
    sc = StandardScaler().fit(X); return sc, LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(X), y)
def pred(model, X, prior):
    sc, M = model; P = np.full(len(X), prior); ok = np.isfinite(X).all(1); P[ok] = M.predict_proba(sc.transform(X[ok]))[:, 1]; return P
base_model = fit(X2[ok2], y2[ok2]); prior = y2[ok2].mean()
P3 = pred(base_model, X3, prior)
S = AF.Sky(d3, P3, rng)

cell_src = [[] for _ in range(S.n)]
for s_, (r, v) in enumerate(zip(S.rows, S.vals)):
    for a, f in zip(r, v): cell_src[a].append((s_, f))
cell_src = [(np.array([x[0] for x in L], int), np.array([x[1] for x in L])) for L in cell_src]
ra_bin = (np.degrees(hp.pix2ang(AF.NSIDE, S.cells)[1]) // 60).astype(int)
lgal, bgal = hp.Rotator(coord=['C', 'G'])(*hp.pix2ang(AF.NSIDE, S.cells))
src_b = np.abs(d3['GLAT'].astype(float)); src_l = d3['GLON'].astype(float)

def gains(w, eps, resid):
    return np.array([np.sum(w[ix] * eps[ix] * f * resid[ix]) if len(ix) else 0.0 for ix, f in cell_src])

def greedy(w, K, eps=None, lazy=True, cap=None):
    eps = np.ones(len(w)) if eps is None else eps; resid = np.ones(len(w)); Z = []; used = np.zeros(6, int)
    g = gains(w, eps, resid); stale = np.zeros(S.n, bool); evals = S.n
    while len(Z) < K:
        if lazy:
            while True:
                i = int(np.argmax(g))
                if cap is not None and used[ra_bin[i]] >= cap: g[i] = -1; continue
                if not stale[i]: break
                ix, f = cell_src[i]; g[i] = np.sum(w[ix] * eps[ix] * f * resid[ix]) if len(ix) else 0.0; stale[i] = False; evals += 1
        else:
            g = gains(w, eps, resid); evals += S.n; g[Z] = -1
            if cap is not None: g[used[ra_bin] >= cap] = -1
            i = int(np.argmax(g))
        Z.append(i); used[ra_bin[i]] += 1
        ix, f = cell_src[i]; resid[ix] *= (1 - eps[ix] * f); g[i] = -1
        stale[:] = True; stale[Z] = False
    return Z, evals

def F(Z, w, eps=None):
    eps = np.ones(len(w)) if eps is None else eps; resid = np.ones(len(w))
    for i in Z: ix, f = cell_src[i]; resid[ix] *= (1 - eps[ix] * f)
    return float(np.sum(w * (1 - resid)))

def hits(Z, eps=None): return F(Z, y3.astype(float), eps)
def n_hit(Z):
    Zs = set(Z); return int(sum(1 for t, r in zip(y3, S.rows) if t and any(a in Zs for a in r)))

def slope_int(p, y):
    from scipy.optimize import minimize
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)); f = lambda b: -np.sum(y * (b[0] + b[1] * x) - np.logaddexp(0, b[0] + b[1] * x))
    b = minimize(f, [0, 1]).x; return float(b[0]), float(b[1])

KS = (20, 50, 200); out = {}

oof = np.zeros(ok2.sum()); Xa, ya = X2[ok2], y2[ok2]
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, ya):
    m = fit(Xa[tr], ya[tr]); oof[te] = m[1].predict_proba(m[0].transform(Xa[te]))[:, 1]
a_, b_ = slope_int(oof, ya)
P3r = expit(a_ + b_ * logit(np.clip(P3, 1e-6, 1 - 1e-6)))
cal = lambda P: dict(auc=float(roc_auc_score(y3, P)), oe=float(y3.sum() / P.sum()), slope=slope_int(P, y3)[1], intercept=slope_int(P, y3)[0])
out['R'] = dict(a=a_, b=b_, before=cal(P3), after=cal(P3r), K={})
for K in KS:
    Zb, _ = greedy(P3, K); Zr, _ = greedy(P3r, K)
    out['R']['K'][K] = dict(overlap=len(set(Zb) & set(Zr)), hits_base=hits(Zb), hits_recal=hits(Zr), n_base=n_hit(Zb), n_recal=n_hit(Zr))
print('R', out['R'], flush=True)

K = 50; Z0, _ = greedy(P3, K)
def perturbed_X(r):
    X = X3.copy(); d = d3
    F1 = d['Flux1000'].astype(float); U1 = np.nan_to_num(d['Unc_Flux1000'].astype(float), nan=0.0)
    X[:, 0] = np.log10(np.clip(F1 + r.standard_normal(len(F1)) * U1, F1 * 0.1, None))
    X[:, 1] = d['PL_Index'].astype(float) + r.standard_normal(len(F1)) * np.nan_to_num(d['Unc_PL_Index'].astype(float), nan=0.0)
    X[:, 2] = np.clip(X3[:, 2] + r.standard_normal(len(F1)), 0, 30)
    return X
inp, mod = [], []
for b in range(100):
    r = np.random.default_rng(100 + b)
    Pi = pred(base_model, perturbed_X(r), prior); Zi, _ = greedy(Pi, K)
    inp.append((len(set(Zi) & set(Z0)), F(Zi, P3), hits(Zi), float(np.std(Pi - P3))))
    ib = r.integers(len(ya), size=len(ya)); mb = fit(Xa[ib], ya[ib]); Pm = pred(mb, X3, prior); Zm, _ = greedy(Pm, K)
    mod.append((len(set(Zm) & set(Z0)), F(Zm, P3), hits(Zm), float(np.std(Pm - P3))))
inp, mod = np.array(inp), np.array(mod)
ro = np.random.default_rng(5); outc = [float(np.sum(ro.random(len(P3)) < P3 * 0 + (1 - np.ones(len(P3))) )) for _ in range(1)]

cov0 = np.zeros(len(P3)); res0 = np.ones(len(P3))
for i in Z0: ix, f = cell_src[i]; res0[ix] *= (1 - f)
cov0 = 1 - res0; sim_hits = [float(np.sum((ro.random(len(P3)) < P3) * cov0)) for _ in range(2000)]
out['U'] = dict(K=K, input=dict(overlap_mean=float(inp[:, 0].mean()), F_rel=float(inp[:, 1].mean() / F(Z0, P3)), hits_sd=float(inp[:, 2].std()), dP_sd=float(inp[:, 3].mean())),
                model=dict(overlap_mean=float(mod[:, 0].mean()), F_rel=float(mod[:, 1].mean() / F(Z0, P3)), hits_sd=float(mod[:, 2].std()), dP_sd=float(mod[:, 3].mean())),
                outcome=dict(hits_mean=float(np.mean(sim_hits)), hits_sd=float(np.std(sim_hits))), hits_Z0=hits(Z0))
print('U', out['U'], flush=True)

Tsky = 3.5 + 25 * np.exp(-src_b / 3.0) * np.where(np.abs(((src_l + 180) % 360) - 180) < 90, 1.0, 0.5); eps = 18 / (18 + Tsky)
out['E'] = dict(eps_min=float(eps.min()), eps_med=float(np.median(eps)), K={})
for K in KS:
    Zc, _ = greedy(P3, K); Ze, _ = greedy(P3, K, eps=eps)
    out['E']['K'][K] = dict(overlap=len(set(Zc) & set(Ze)), F_const_plan=F(Zc, P3, eps), F_eps_plan=F(Ze, P3, eps), hits_const=hits(Zc, eps), hits_eps=hits(Ze, eps))
print('E', out['E'], flush=True)

out['O'] = {}
for K in KS:
    cap = math.ceil(K / 6); Zu, _ = greedy(P3, K); Zo, _ = greedy(P3, K, cap=cap)
    out['O'][K] = dict(cap=cap, overlap=len(set(Zu) & set(Zo)), F_free=F(Zu, P3), F_cap=F(Zo, P3), n_free=n_hit(Zu), n_cap=n_hit(Zo), bins_free=np.bincount(ra_bin[Zu], minlength=6).tolist())
print('O', out['O'], flush=True)

pri = (src_b > 10) & (X3[:, 2] > 3); w = P3 * np.where(pri, 3.0, 1.0)
out['C'] = dict(n_priority=int(pri.sum()), n_priority_id=int((pri & (y3 == 1)).sum()), K={})
for K in KS:
    Zu, _ = greedy(P3, K); Zp, _ = greedy(w, K)
    cov = lambda Z: float(np.sum(P3[pri] * (1 - np.prod([1 - 0 for _ in []])) )) if False else None
    out['C']['K'][K] = dict(overlap=len(set(Zu) & set(Zp)), Fpri_base=F(Zu, P3 * pri), Fpri_w=F(Zp, P3 * pri), Fall_base=F(Zu, P3), Fall_w=F(Zp, P3),
                            hits_pri_base=F(Zu, (y3 * pri).astype(float)), hits_pri_w=F(Zp, (y3 * pri).astype(float)))
print('C', out['C'], flush=True)

import json as _j
A = _j.load(open('astro_full_prospective.json'))
out['G'] = {}
for K in KS:
    Zl, el = greedy(P3, K, lazy=True); Zf, ef = greedy(P3, K, lazy=False)
    milp_cells = A['K'][str(K)]['plans']['QCDF_MILP']; Zm = [S.loc[c] for c in milp_cells]
    out['G'][K] = dict(same=len(set(Zl) & set(Zf)), F_lazy=F(Zl, P3), F_full=F(Zf, P3), F_milp=F(Zm, P3), evals_lazy=el, evals_full=ef)
print('G', out['G'], flush=True)

milp20 = [S.loc[c] for c in A['K']['20']['plans']['QCDF_MILP']]
expl = []
for i in milp20:
    ix, f = cell_src[i]; o = np.argsort(-P3[ix] * f)[:3]
    expl.append(dict(cell=int(S.cells[i]), ra=float(np.degrees(hp.pix2ang(AF.NSIDE, S.cells[i])[1])), dec=float(90 - np.degrees(hp.pix2ang(AF.NSIDE, S.cells[i])[0])),
                     n_src=int(len(ix)), expected=float(np.sum(P3[ix] * f)),
                     top=[dict(name=str(n3[ix[k]]), P=float(P3[ix[k]]), f=float(f[k]), identified=bool(y3[ix[k]])) for k in o]))
out['X'] = dict(plan20=expl)
rank = np.argsort(np.argsort(-P3)) + 1; om = {}
for K in (50, 200):
    Zs = set(A['K'][str(K)]['plans']['QCDF_MILP']); rows = []
    for s_ in np.where(y3 == 1)[0]:
        dec = float(d3['DEJ2000'][s_]); covered = any(S.cells[a] in Zs for a in S.rows[s_])
        reason = 'covered' if covered else ('not visible' if dec >= AF.DEC_MAX or not len(S.rows[s_]) else ('low P' if rank[s_] > 2 * K else 'beyond budget'))
        rows.append(dict(name=str(n3[s_]), P=float(P3[s_]), rank=int(rank[s_]), dec=dec, reason=reason))
    om[K] = dict(rows=rows, counts={r: sum(1 for x in rows if x['reason'] == r) for r in ('covered', 'not visible', 'low P', 'beyond budget')})
out['X']['omissions'] = om
print('X', {K: v['counts'] for K, v in om.items()}, flush=True)
json.dump(out, open('astro_v11.json', 'w'), indent=1, default=float)
np.savez('astro_map.npz', ra=d3['RAJ2000'].astype(float), dec=d3['DEJ2000'].astype(float), P=P3, y=y3, cells=S.cells,
         plan20=np.array(A['K']['20']['plans']['QCDF_MILP']), plan50=np.array(A['K']['50']['plans']['QCDF_MILP']), plan200=np.array(A['K']['200']['plans']['QCDF_MILP']))
