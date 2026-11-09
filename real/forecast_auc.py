"""10-day forecast skill of the calibrated spread model at each decision date: AUC of P_i(burn within 10 days | no action) against the observed burns, over unburned domain cells; and the same for H_i*P_i against observed housing burned (ranking of the candidates)."""
import sys, json, numpy as np, pandas as pd
sys.argv = [sys.argv[0]] + sys.argv[1:]
dates = sys.argv[1].split(','); out = sys.argv[2]
import realcase as RC
def auc(p, y):
    o = np.argsort(p, kind='mergesort'); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)

    s = pd.Series(p).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return float((s[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
res = []
for d in dates:
    T0 = pd.Timestamp(d); rng = np.random.default_rng(int(T0.strftime('%m%d')))
    D = RC.Domain(T0, rng); _, pb = D.run([], D.UA)
    gi = D.cells; un = ~D.burned0
    y = ((RC.G.ign_date.values[gi] > np.datetime64(T0)) & (RC.G.ign_date.values[gi] <= np.datetime64(T0 + pd.Timedelta(days=10)))).astype(int)
    var = D.value * pb; cand = np.argsort(-np.where(un, var, -1))[:20]
    res.append(dict(T0=d, auc=auc(pb[un], y[un]), n=int(un.sum()), pos=int(y[un].sum()),
                    pred=float(pb[un].sum()), cand_burn_obs=float(y[cand].mean()), cand_burn_pred=float(pb[cand].mean()),
                    hous_burn_obs=float((y * D.value)[un].sum()), hous_burn_pred=float(var[un].sum())))
    print(res[-1], flush=True)
json.dump(res, open(out, 'w'), indent=1)
