"""Forecast skill and probability calibration of the three spread models (oracle = Dixie <=23 Aug, prospective = GOFER 2019-2020, rolling = Dixie <= T0 with prospective prior) at each decision date."""
import sys, json, numpy as np, pandas as pd
sys.argv = ['x']; import fullscale as FS
from scipy.optimize import minimize
dates = open('dates.txt').read().split('\n')[2].split(',')
rows = []; cellrows = []
for m in ('oracle', 'prospective', 'rolling'):
    for d in dates:
        P = FS.Problem(d, model=m); L, pb = P.sim.losses([], 1000, 21)
        un = ~P.D.burned0; y = P.obs_burn[un].astype(int); p = pb[un]; H = P.D.value[un]
        obsH = float((H * y).sum())
        rows.append(dict(model=m, T0=d, n=int(un.sum()), obs_cells=int(y.sum()), pred_cells=float(p.sum()), obs_H=obsH, pred_H=float((H * p).sum()),
                         brier=float(np.mean((p - y) ** 2)), q05=float(np.quantile(L, 0.05)), q95=float(np.quantile(L, 0.95)),
                         pit=float(np.mean(L <= obsH))))
        for pi, yi, hi in zip(p, y, H): cellrows.append((m, d, pi, yi, hi))
    print(m, 'done', flush=True)
R = pd.DataFrame(rows); C = pd.DataFrame(cellrows, columns=['model', 'T0', 'p', 'y', 'H'])
def auc(p, y):
    s = pd.Series(p).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return float((s[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def slope_int(p, y):
    pc = np.clip(p, 0.5 / 1001, 1 - 0.5 / 1001); x = np.log(pc / (1 - pc))
    f = lambda b: -np.sum(y * (b[0] + b[1] * x) - np.logaddexp(0, b[0] + b[1] * x))
    b = minimize(f, [0, 1]).x; return float(b[0]), float(b[1])
out = []
for m in ('oracle', 'prospective', 'rolling'):
    for ph, f in (('active', lambda d: d <= '2021-08-05'), ('late', lambda d: d > '2021-08-05'), ('all', lambda d: True)):
        r = R[(R.model == m) & R.T0.map(f)]; c = C[(C.model == m) & C.T0.map(f)]
        a, b = slope_int(c.p.values, c.y.values)
        out.append(dict(model=m, phase=ph, n_dates=len(r), auc_median=float(np.median([auc(g.p.values, g.y.values) for _, g in c.groupby('T0') if 0 < g.y.sum() < len(g)])),
                        auc_pooled=auc(c.p.values, c.y.values), brier=float(np.mean((c.p - c.y) ** 2)),
                        oe_cells=float(r.obs_cells.sum() / r.pred_cells.sum()), oe_H=float(r.obs_H.sum() / r.pred_H.sum()),
                        cal_int=a, cal_slope=b, cover90=float(np.mean((r.obs_H >= r.q05) & (r.obs_H <= r.q95)))))
        print(out[-1], flush=True)
bins = [0, .01, .05, .1, .2, .4, .6, 1.0001]; rel = []
for m in ('oracle', 'prospective', 'rolling'):
    for ph, f in (('active', lambda d: d <= '2021-08-05'), ('late', lambda d: d > '2021-08-05')):
        c = C[(C.model == m) & C.T0.map(f)]; k = np.digitize(c.p, bins) - 1
        for b_ in range(len(bins) - 1):
            s = c[k == b_]
            if len(s): rel.append(dict(model=m, phase=ph, bin=b_, n=len(s), p=float(s.p.mean()), y=float(s.y.mean())))
json.dump(dict(per_date=rows, summary=out, reliability=rel), open('calib_check.json', 'w'), indent=1)
