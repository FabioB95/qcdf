"""Remaining protocol checks on the Dixie hindcast (prospective model, active phase). thr coupling selection: keep J_ij when |I| >= c SE for c in {0, 1, 2 (main), 3} and with a Benjamini-Hochberg false-discovery rate of 5%; K=10 plan and its value. nest budgets: overlap of the K=5, 10, 20 plans (are they nested?). reinf reinforcements: 5 units placed first, then 5 more given the first five (S_prev), against the K=10 plan from scratch; same for 10 + 10 against 20. c2010 value known at the decision: plans built with Census 2010 housing (Census 2020 was published on 12 August 2021), evaluated with Census 2020 housing. delay construction time: protection starts on day 2 of the horizon instead of day 0. last5 line maintenance: protection lasts 5 of the 10 days. All plans are evaluated on the same held-out stream."""
import sys, json, math, time, numpy as np, pandas as pd
from scipy.stats import norm
_ARGV = list(sys.argv); sys.argv = ['x']
import fullscale as FS
sys.argv = _ARGV
from ops import Pv
from fastsim import Sim
H10 = pd.read_csv('housing10.csv').set_index('h3').HOUSING10

def val(P, S, base=None):
    b = P.L(set(), FS.R_B, 99).mean() if base is None else base
    return float(100 * (b - P.L(set(S), FS.R_B, 99).mean()) / b)

def run(T0):
    t0 = time.time(); out = dict(T0=T0)
    P = Pv(T0).build_qubo(); base = P.L(set(), FS.R_B, 99).mean()

    raw = P.Jraw; Jmain = dict(P.J); res = {}
    for lab, c in (('all', 0.0), ('1se', 1.0), ('2se', 2.0), ('3se', 3.0)):
        P.J = {p: -I for p, (I, se) in raw.items() if se > 0 and abs(I) >= c * se}
        S, _ = P.milp(10); res[lab] = dict(nJ=len(P.J), val=val(P, S, base), plan=sorted(S))
    pv = {p: 2 * (1 - norm.cdf(abs(I) / se)) for p, (I, se) in raw.items() if se > 0}
    ks = sorted(pv, key=pv.get); m = len(ks); keep = 0
    for r, p in enumerate(ks, 1):
        if pv[p] <= 0.05 * r / m: keep = r
    P.J = {p: -raw[p][0] for p in ks[:keep]}; S, _ = P.milp(10); res['fdr5'] = dict(nJ=len(P.J), val=val(P, S, base), plan=sorted(S))
    P.J = Jmain
    out['thr'] = {k: {kk: vv for kk, vv in v.items() if kk != 'plan'} | dict(overlap_main=len(set(v['plan']) & set(res['2se']['plan']))) for k, v in res.items()}

    S5, _ = P.milp(5); S10, _ = P.milp(10); S20, _ = P.milp(20)
    out['nest'] = dict(o5_10=len(set(S5) & set(S10)), o10_20=len(set(S10) & set(S20)), o5_20=len(set(S5) & set(S20)))
    Pr = Pv(T0, S_prev=S5).build_qubo(); add, _ = Pr.milp(5); S55 = list(S5) + list(add)
    Pr2 = Pv(T0, S_prev=S10).build_qubo(); add2, _ = Pr2.milp(10); S1010 = list(S10) + list(add2)
    out['reinf'] = dict(v10=val(P, S10, base), v5p5=val(P, S55, base), v20=val(P, S20, base), v10p10=val(P, S1010, base),
                        o10=len(set(S55) & set(S10)), o20=len(set(S1010) & set(S20)))

    Pc = Pv(T0); h10 = np.array([H10.get(FS.G.h3.iloc[g], 0.0) for g in Pc.D.cells]); Pc.D.value = h10.astype(float); Pc.sim = Sim(Pc.D, 0.85)
    Pc.build_qubo(); Sc, _ = Pc.milp(10); bl = Pc.baselines(10)
    out['c2010'] = dict(qcdf=val(P, Sc, base), var=val(P, bl['B4_value_at_risk'], base), qcdf2020=val(P, S10, base), overlap=len(set(Sc) & set(S10)))

    H = P.sim.H
    for lab, mask in (('delay', np.r_[np.zeros(2), np.ones(H - 2)]), ('last5', np.r_[np.ones(5), np.zeros(H - 5)])):
        Pe = Pv(T0); Pe.eta_mat = 0.85 * mask[:, None] * np.ones((H, Pe.n)); Pe.build_qubo()
        Sq, _ = Pe.milp(10); Sh, _ = Pe.milp(10, use_J=False); b2 = Pe.baselines(10)
        be = Pe.L(set(), FS.R_B, 99).mean(); ve = lambda S: float(100 * (be - Pe.L(set(S), FS.R_B, 99).mean()) / be)
        out[lab] = dict(qcdf=ve(Sq), h_only=ve(Sh), var=ve(b2['B4_value_at_risk']), front=ve(b2['B2_nearest_front']), comm=ve(b2['B1_communities']), main_plan=ve(S10))
    print(T0, out['thr']['2se']['val'], {k: v['val'] for k, v in out['thr'].items()}, out['nest'], out['reinf'], out['c2010'], out['delay'], out['last5'], f'[{time.time()-t0:.0f}s]', flush=True)
    return out

if __name__ == '__main__':
    dates = sys.argv[1].split(','); outf = sys.argv[2]; res = []
    for d in dates:
        res.append(run(d)); json.dump(res, open(outf, 'w'), default=float)
