"""Thinning law for q-oracle sub-solvers in LNS: prediction vs observation."""
import json, sys, glob, numpy as np
from scipy.stats import binom

def analyse(recs, eps=0.002):
    rows = []
    for r in recs:
        A = r['arms']; B = len(A['exact'][0]['gap'])
        g_ex = np.mean([x['gap'] for x in A['exact']], 0)
        g_ex0 = r['gap0']
        p = np.nanmean([x['p'] for x in A['exact'] if x['p'] is not None]); u = np.nanmean([x['u'] for x in A['exact'] if x['u'] is not None])

        rho = u / p if p > 0 else 1.0
        dimF = np.mean([np.mean(x['dimF']) for x in A['exact'] if x['dimF']])
        frac_used = np.mean([x['used'] for x in A['exact']]) / B
        tr = np.r_[g_ex0, g_ex]
        obs_uni = np.mean([x['gap'][-1] for x in A['uniform']])
        for arm in [a for a in A if a.startswith('q')] + ['uniform']:
            q = float(arm[1:]) if arm.startswith('q') else 0.0
            rr = q + (1 - q) * rho
            n = np.arange(B + 1); pred = float((binom.pmf(n, B, rr) * tr).sum()); pred0 = float((binom.pmf(n, B, q) * tr).sum())
            obs = float(np.mean([x['gap'][-1] for x in A[arm]]))
            rows.append(dict(domain=r.get('domain', 'fire'), T0=r.get('T0', ''), K=r['K'], w=r['w'], S=r['S'], dimF=float(dimF), rho=float(rho),
                             q=q, arm=arm, pred=pred, pred_thin=pred0, obs=obs, exact=float(g_ex[-1]), gap0=float(g_ex0), frac_used=float(frac_used)))
    return rows

if __name__ == '__main__':
    recs = []
    for f in sys.argv[1:]: recs += json.load(open(f))
    R = analyse(recs)
    import pandas as pd
    D = pd.DataFrame(R); D['excess_obs'] = D.obs - D.exact; D['excess_pred'] = D.pred - D.exact
    print(D.groupby(['domain', 'w', 'S'])[['dimF', 'rho']].mean().round(3))
    print('corr(pred, obs) excess:', np.corrcoef(D.excess_pred, D.excess_obs)[0, 1].round(3), ' MAE', (D.excess_pred - D.excess_obs).abs().mean().round(5))
    D.to_csv('analyse_q.csv', index=False)
