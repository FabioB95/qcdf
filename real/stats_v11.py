"""Statistical robustness of the main Dixie comparison (prospective model, K=10, 29 dates) and the separation of three sources of uncertainty. * Interval sensitivity: circular block bootstrap with block length 3, 6 (main), 10 dates; moving (non-circular) blocks of 6; calendar blocks (resampling whole 10-day calendar periods); per-date paired differences listed. * Monte Carlo precision: standard error of each date's difference from the 2,000 evaluation realisations (paired, common random numbers), against the spread of the differences between dates. * Between fires: Dixie against Caldor (same protocol)."""
import json, numpy as np, pandas as pd
R = json.load(open('fsP_act.json')) + json.load(open('fsP_late.json'))
R = sorted(R, key=lambda r: r['T0'])
def diff(r, a, b, K='10'):
    e = r['K'][K]['eval']; return e[a]['saved_pct'] - e[b]['saved_pct']
def mcse(r, a, b, K='10'):
    e = r['K'][K]['eval']; return np.hypot(e[a]['se'], e[b]['se']) * 100 / max(e['_none']['loss'], 1e-9)
pairs = [('QCDF_MILP', 'QCDF_h_only'), ('QCDF_MILP', 'B4_value_at_risk'), ('QCDF_MILP', 'CELF_greedy')]
dates = pd.to_datetime([r['T0'] for r in R]); n = len(R); rng = np.random.default_rng(11); B = 10000
def cbb(x, L, circular=True):
    out = np.empty(B); nb = int(np.ceil(n / L))
    for b in range(B):
        if circular: st = rng.integers(n, size=nb); idx = np.concatenate([(s + np.arange(L)) % n for s in st])[:n]
        else: st = rng.integers(n - L + 1, size=nb); idx = np.concatenate([s + np.arange(L) for s in st])[:n]
        out[b] = x[idx].mean()
    return np.percentile(out, [2.5, 97.5])
def calendar(x):
    per = ((dates - dates[0]).days // 10).values; groups = [np.where(per == g)[0] for g in np.unique(per)]
    out = np.empty(B)
    for b in range(B):
        g = rng.integers(len(groups), size=len(groups)); idx = np.concatenate([groups[k] for k in g]); out[b] = x[idx].mean()
    return np.percentile(out, [2.5, 97.5]), len(groups)
res = {}
for a, b in pairs:
    x = np.array([diff(r, a, b) for r in R]); m = np.array([mcse(r, a, b) for r in R])
    ci_cal, ng = calendar(x)
    res[f'{a}-{b}'] = dict(mean=float(x.mean()), per_date=x.round(3).tolist(),
                           cbb3=cbb(x, 3).tolist(), cbb6=cbb(x, 6).tolist(), cbb10=cbb(x, 10).tolist(), mbb6=cbb(x, 6, False).tolist(),
                           calendar=ci_cal.tolist(), n_calendar_blocks=ng, mc_se_median=float(np.median(m)), mc_se_upper=float(np.max(m)),
                           between_date_sd=float(x.std(ddof=1)), n_pos=int((x > 0).sum()), n=n)

C = json.load(open('fsC.json'))
for a, b in pairs:
    x = np.array([diff(r, a, b) for r in C]); res[f'{a}-{b}']['caldor_mean'] = float(x.mean()); res[f'{a}-{b}']['caldor_sd'] = float(x.std(ddof=1))
json.dump(dict(dates=[str(d.date()) for d in dates], res=res), open('stats_v11.json', 'w'), indent=1)
for k, v in res.items():
    print(k, {kk: (np.round(vv, 2).tolist() if isinstance(vv, list) and kk != 'per_date' else vv) for kk, vv in v.items() if kk != 'per_date'})
