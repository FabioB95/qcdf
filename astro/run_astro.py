"""Prospective identification model (DR2->DR3), test on DR3->DR4, and the full-scale allocation study."""
import sys, json, numpy as np, healpy as hp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import astro_full as AF

rng = np.random.default_rng(7)
import os
TR = os.environ.get('ASTRO_TRAIN', 'gll_psc_v27.fit,gll_psc_v31.fit').split(','); TE = os.environ.get('ASTRO_TEST', 'gll_psc_v31.fit,gll_psc_v35.fit').split(',')
SUF = os.environ.get('ASTRO_SUFFIX', '')
d2, n2, X2, y2 = AF.transition(*TR)
d3, n3, X3, y3 = AF.transition(*TE)
ok2 = np.isfinite(X2).all(1); ok3 = np.isfinite(X3).all(1)
sc = StandardScaler().fit(X2[ok2]); M = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(X2[ok2]), y2[ok2])
P3 = np.full(len(X3), y2[ok2].mean()); P3[ok3] = M.predict_proba(sc.transform(X3[ok3]))[:, 1]

from scipy.special import logit
lp = logit(np.clip(P3, 1e-6, 1 - 1e-6))
slope = LogisticRegression(C=1e6).fit(lp[:, None], y3).coef_[0, 0]
cal = dict(n_train=int(ok2.sum()), pos_train=int(y2[ok2].sum()), n_test=int(len(y3)), pos_test=int(y3.sum()),
           auc_test=float(roc_auc_score(y3, P3)), brier=float(brier_score_loss(y3, P3)), obs_exp=float(y3.sum() / P3.sum()),
           slope=float(slope), coef=dict(zip(AF.FEAT, M.coef_[0].round(3).tolist())))

cal['auc_curv_only'] = float(roc_auc_score(y3, X3[:, 2])); cal['auc_flux_only'] = float(roc_auc_score(y3, np.nan_to_num(X3[:, 0], nan=-20)))
print(json.dumps(cal, indent=1), flush=True)

sc4 = StandardScaler().fit(X3[ok3]); M4 = LogisticRegression(C=0.3, max_iter=2000).fit(sc4.transform(X3[ok3]), y3[ok3])
P3o = np.full(len(X3), y3.mean()); P3o[ok3] = M4.predict_proba(sc4.transform(X3[ok3]))[:, 1]
cal['auc_oracle_insample'] = float(roc_auc_score(y3, P3o))
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'prospective'
if MODEL == 'oracle': P3 = P3o
S = AF.Sky(d3, P3, rng)
print('cells', S.n, 'J', len(S.J), 'nonzero J>1e-4', sum(v > 1e-4 for v in S.J.values()), flush=True)
truth = y3.astype(float)
hp_counts = np.zeros(S.n)
for r in S.rows:
    for a in r: hp_counts[a] += 1
res = dict(cal=cal, n_cells=S.n, n_J=len(S.J), K={})
for K in (10, 20, 50, 100, 200):
    plans = {}
    plans['QCDF_MILP'], _ = S.milp(K); plans['QCDF_h_only'], _ = S.milp(K, use_J=False)
    plans['greedy_exact'] = S.greedy(K)
    plans['most_sources'] = list(np.argsort(-hp_counts)[:K])
    bright = np.argsort(-np.nan_to_num(X3[:, 0], nan=-20)); Zb = []
    for sidx in bright:
        if not len(S.rows[sidx]): continue
        a = int(S.rows[sidx][np.argmax(S.vals[sidx])])
        if a not in Zb: Zb.append(a)
        if len(Zb) == K: break
    plans['brightest'] = Zb
    curv = np.argsort(-X3[:, 2]); Zc = []
    for sidx in curv:
        if not len(S.rows[sidx]): continue
        a = int(S.rows[sidx][np.argmax(S.vals[sidx])])
        if a not in Zc: Zc.append(a)
        if len(Zc) == K: break
    plans['most_curved'] = Zc
    rr = np.random.default_rng(K); plans['random'] = list(rr.choice(S.n, K, replace=False))
    out = {}
    for k, Z in plans.items():
        out[k] = dict(F_model=S.F(Z), E=S.energy(Z), hits=S.F(Z, truth), n_hit_any=int(sum(1 for t, r, v in zip(truth, S.rows, S.vals) if t and len(r) and np.isin(r, Z).any())))

    B = 4000; rb = np.random.default_rng(1); pos = np.where(truth > 0)[0]
    cov = {k: np.array([1 - np.prod(1 - S.vals[s][np.isin(S.rows[s], Z)]) if len(S.rows[s]) else 0 for s in pos]) for k, Z in plans.items()}
    for k in plans:
        if k == 'QCDF_MILP': continue
        dd = cov['QCDF_MILP'] - cov[k]; bs = [dd[rb.integers(len(dd), size=len(dd))].sum() for _ in range(B)]
        out[k]['hits_diff_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res['K'][K] = dict(plans={k: [int(S.cells[i]) for i in Z] for k, Z in plans.items()}, eval=out)
    print(K, {k: (round(v['F_model'], 2), round(v['hits'], 2), v['n_hit_any']) for k, v in out.items()}, flush=True)
json.dump(res, open(f'astro_full_{MODEL}{SUF}.json', 'w'), indent=1)
import pickle; pickle.dump(dict(h=S.h, J=S.J, adj=S.adj, cells=S.cells), open(f'astro_ham_{MODEL}{SUF}.pkl', 'wb'))
