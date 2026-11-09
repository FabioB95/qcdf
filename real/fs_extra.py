"""Full-scale extras: (1) closed- vs open-loop re-planning over the active phase; (2) sensitivity to protection effectiveness eta (constant and weather-dependent) and crew number K; (3) Greenville timeline."""
import sys, json, time, numpy as np, pandas as pd
import fullscale as FS
G = FS.G
ACTIVE = ['2021-07-24', '2021-07-25', '2021-07-27', '2021-07-28', '2021-07-29', '2021-07-30', '2021-07-31', '2021-08-01', '2021-08-04', '2021-08-05']
GREENVILLE = '87281472affffff'

def to_local(P, gl):
    return [P.loc[g] for g in gl if g in P.loc and not P.D.burned0[P.loc[g]]]

def closed_open(Knew=5):
    out = []; closed_g, var_g = [], []
    P0 = FS.Problem(ACTIVE[0]).build_qubo()
    Zo, _ = P0.milp(Knew * len(ACTIVE))
    order = [int(P0.D.cells[i]) for i in sorted(Zo, key=lambda i: P0.h[i])]
    for k, d in enumerate(ACTIVE):
        t0 = time.time()

        Pbase = FS.Problem(d); prev = to_local(Pbase, closed_g)
        Pc = FS.Problem(d, S_prev=prev).build_qubo()
        new, _ = Pc.milp(Knew); closed_g += [int(Pc.D.cells[i]) for i in new]

        pv = to_local(Pbase, var_g)
        _, pb = Pbase.sim.losses(set(pv) if pv else [], FS.R_A, 13, eta=Pbase.eta)
        var = Pbase.D.value * pb; cand = [i for i in Pbase.elig[np.argsort(-var[Pbase.elig])] if i not in set(pv)][:Knew]
        var_g += [int(Pbase.D.cells[i]) for i in cand]
        open_g = order[:Knew * (k + 1)]
        plans = {'closed_QCDF': to_local(Pbase, closed_g), 'open_QCDF_24Jul': to_local(Pbase, open_g), 'closed_VaR': to_local(Pbase, var_g)}
        ev = Pbase.evaluate(plans)
        rec = dict(T0=d, n_units=Knew * (k + 1), **{k_: dict(saved_pct=v['saved_pct'], n_active=len(plans[k_]), prec=v['prec']) for k_, v in ev.items() if not k_.startswith('_')})
        out.append(rec); print('loop', d, {k_: round(v['saved_pct'], 1) for k_, v in ev.items() if not k_.startswith('_')}, f"[{time.time()-t0:.0f}s]", flush=True)
    return out

def sensitivity(dates=('2021-07-27', '2021-07-31', '2021-08-04')):
    out = []
    for d in dates:
        for eta, mode in [(0.3, 'const'), (0.5, 'const'), (0.7, 'const'), (0.85, 'const'), (0.85, 'weather')]:
            o, _ = FS.run_date(d, Ks=(5, 10, 20), eta=eta, eta_mode=mode, do_fid=False, verbose=False)
            for K, r in o['K'].items():
                e = r['eval']; out.append(dict(T0=d, eta=eta, mode=mode, K=K, **{k: v['saved_pct'] for k, v in e.items() if not k.startswith('_')}))
            print('sens', d, eta, mode, {K: (round(r['eval']['QCDF_MILP']['saved_pct'], 1), round(r['eval']['B4_value_at_risk']['saved_pct'], 1), round(r['eval']['CELF_greedy']['saved_pct'], 1)) for K, r in o['K'].items()}, flush=True)
    return out

def greenville(dates=('2021-07-27', '2021-07-28', '2021-07-29', '2021-07-30', '2021-07-31', '2021-08-01', '2021-08-04')):
    out = []; g = int(np.where(G.h3.values == GREENVILLE)[0][0])
    for d in dates:
        P = FS.Problem(d).build_qubo()
        if g not in P.loc: out.append(dict(T0=d, in_domain=False)); continue
        i = P.loc[g]; E = P.elig
        rk = lambda score: int((score[E] > score[i]).sum() + 1) if i in set(E) else None
        _, pb = P.sim.losses([], FS.R_A, 13, eta=P.eta)
        rec = dict(T0=d, in_domain=True, eligible=bool(i in set(E)), ring=int(P.dist[i]), P_burn10=float(pb[i]), housing=float(P.D.value[i]),
                   rank_delta=rk(P.delta), rank_var=rk(P.D.value * pb), rank_communities=rk(P.D.value.astype(float)), n_elig=int(len(E)))
        for K in (5, 10, 20):
            Z, _ = P.milp(K); rec[f'in_QCDF_K{K}'] = bool(i in Z)
        out.append(rec); print('greenville', rec, flush=True)
    return out

if __name__ == '__main__':
    what = sys.argv[1]; res = {}; suf = '' if FS.MODEL_MODE == 'oracle' else '_' + FS.MODEL_MODE
    if 'loop' in what: res['loop'] = closed_open(); json.dump(res, open(f'fs_extra_loop{suf}.json', 'w'), indent=1)
    if 'green' in what: res['greenville'] = greenville(); json.dump(res, open(f'fs_extra_green{suf}.json', 'w'), indent=1)
    if 'sens' in what: res['sens'] = sensitivity(); json.dump(res, open(f'fs_extra_sens{suf}.json', 'w'), indent=1)
