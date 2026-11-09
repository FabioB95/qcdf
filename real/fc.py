"""Operational replay without future information (Dixie, decision-time domain, prospective model)."""
import os, sys, json, math, time, numpy as np, pandas as pd
_ARGV = list(sys.argv); sys.argv = ['x']
import fullscale as FS
sys.argv = _ARGV
RC = FS.RC
from fastsim import Sim
from scipy.interpolate import RegularGridInterpolator

FC = np.load('../gefs/gefs_fc.npz'); AN = np.load('../gefs/gefs_an.npz', allow_pickle=True)
R1, R2 = 60, 100

def to_cells(field, la, lo, lat, lon):
    la_ = la[:, 0]; lo_ = lo[0] - 360 if lo[0, 0] > 180 else lo[0]
    o = np.argsort(la_); f = RegularGridInterpolator((la_[o], lo_), field[o], bounds_error=False, fill_value=None)
    return f(np.c_[lat, lon])

def member_weather(T0, cells):
    k = list(FC['dates']).index(str(T0.date())); la, lo = FC['lat'], FC['lon']
    lat, lon = RC.lat[cells], RC.lon[cells]
    U = FC['u'][k]; V = FC['v'][k]
    M, Lh = U.shape[:2]
    vs = np.zeros((M, 10, len(cells))); th = np.zeros((M, 10, len(cells)))
    for m in range(M):
        for d in range(10):
            us = [to_cells(U[m, 4 * d + j], la, lo, lat, lon) for j in range(4)]
            vv = [to_cells(V[m, 4 * d + j], la, lo, lat, lon) for j in range(4)]
            vs[m, d] = np.mean([np.hypot(a, b) for a, b in zip(us, vv)], 0)
            th[m, d] = (np.degrees(np.arctan2(-np.mean(us, 0), -np.mean(vv, 0))) + 360) % 360

    times = pd.to_datetime(AN['times']); sp = []
    obs = []
    for d in pd.date_range(T0 - pd.Timedelta(days=14), T0 - pd.Timedelta(days=1)):
        sel = np.where((times >= d + pd.Timedelta(hours=7)) & (times < d + pd.Timedelta(hours=31)))[0]
        sp.append(np.mean([np.hypot(to_cells(AN['u'][t], la, lo, lat, lon), to_cells(AN['v'][t], la, lo, lat, lon)) for t in sel], 0))
        obs.append(RC.met['vs'][RC.mday[d], cells])
    ratio = np.clip(np.mean(obs, 0) / np.maximum(np.mean(sp, 0), 0.3), 0.5, 3.0)
    return vs * ratio[None, None], th, ratio

def set_weather(P, vs, th):
    C = json.load(open('calib_prospective.json')); beta = np.array(C['models']['wind+slope']['beta']); nm = C['norm']
    D = P.D; gi, gj = D.src, D.dst
    br = np.array([RC.bearing(i, j) for i, j in zip(D.cells[gi], D.cells[gj])])
    W = np.stack([vs[t, gi] * np.cos(np.radians(br - (th[t, gi] + 180) % 360)) for t in range(10)])
    S = (RC.elev[D.cells[gj]] - RC.elev[D.cells[gi]]) / (2.4 * 1000)
    D.q = 1 / (1 + np.exp(-(beta[0] + beta[1] * (W - nm['W'][0]) / nm['W'][1] + beta[2] * (S - nm['S'][0]) / nm['S'][1])))
    P.sim = Sim(D, 0.85)

def run(T0):
    t0 = time.time(); T0 = pd.Timestamp(T0)
    Pobs = FS.Problem(T0).build_qubo()
    vs, th, ratio = member_weather(T0, Pobs.D.cells); M = vs.shape[0]
    Pm = FS.Problem(T0)
    E = Pobs.elig; n = Pobs.n
    deltas = np.zeros((M, n)); bases = []; pbs = np.zeros((M, n))
    for m in range(M):
        set_weather(Pm, vs[m], th[m])
        sa = 11 + 1000 * m
        base, pb = Pm.sim.losses([], R1, sa, eta=Pm.eta); bases.append(base.mean()); pbs[m] = pb
        for i in E: deltas[m, i] = base.mean() - Pm.L({i}, R1, sa).mean()
    delta = deltas.mean(0)
    top = list(E[np.argsort(-delta[E])[:FS.M_PAIR]]); tops = set(top)
    pairs = sorted({(min(i, j), max(i, j)) for i in top for j in Pm.adj[i] if j in tops})
    DD = {p: [] for p in pairs}
    for m in range(M):
        set_weather(Pm, vs[m], th[m])
        sp = 12 + 1000 * m
        b = Pm.L(set(), R2, sp); single = {i: Pm.L({i}, R2, sp) for i in top}
        for i, j in pairs: DD[(i, j)].append(Pm.L({i, j}, R2, sp) - single[i] - single[j] + b)
    J = {}
    for p, v in DD.items():
        v = np.concatenate(v); I = -v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
        if abs(I) >= 2 * se: J[p] = -I
    if os.environ.get('FC_MEMBERS'):

        rng = np.random.default_rng(1); halves = []
        for b in range(10):
            idx = rng.permutation(M)[:M // 2]; d_ = deltas[idx].mean(0)
            Jh = {}
            for p, v in DD.items():
                vv = np.concatenate([v[k] for k in idx]); I = -vv.mean(); se = vv.std(ddof=1) / math.sqrt(len(vv))
                if abs(I) >= 2 * se: Jh[p] = -I
            Pm.h = -d_; Pm.J = Jh; Pm.elig = E; Sh, _ = Pm.milp(10); halves.append(Sh)
        member_overlap = [len(set(a) & set(b)) for i, a in enumerate(halves) for b in halves[i + 1:]]
    else:
        halves, member_overlap = [], []
    Pm.h = -delta; Pm.J = J; Pm.elig = E
    K = 10; plans = {}
    plans['FC_QCDF'], _ = Pm.milp(K); plans['FC_h_only'], _ = Pm.milp(K, use_J=False)
    var = Pobs.D.value * pbs.mean(0); plans['FC_VaR'] = list(E[np.argsort(-var[E])[:K]])
    plans['FC_control_QCDF'] = None

    Pc = FS.Problem(T0); set_weather(Pc, vs[0], th[0]); Pc.build_qubo(); plans['FC_control_QCDF'], _ = Pc.milp(K)

    plans['OBS_QCDF'], _ = Pobs.milp(K); plans['OBS_h_only'], _ = Pobs.milp(K, use_J=False)
    bl = Pobs.baselines(K); plans['OBS_VaR'] = bl['B4_value_at_risk']; plans['B1_communities'] = bl['B1_communities']; plans['B2_nearest_front'] = bl['B2_nearest_front']
    Ev = Pobs.evaluate(plans)

    Lm = []
    for m in range(M):
        set_weather(Pm, vs[m], th[m]); Lm.append(Pm.L(set(plans['FC_QCDF']), 100, 31 + 1000 * m))
    Lm = np.array(Lm); within = Lm.var(1, ddof=1).mean(); between = Lm.mean(1).var(ddof=1)

    obs_vs = np.stack([RC.met['vs'][RC.mday[T0 + pd.Timedelta(days=t)], Pobs.D.cells] for t in range(10)])
    err = vs.mean(0) - obs_vs
    out = dict(T0=str(T0.date()), n=int(n), n_elig=int(len(E)), n_J=len(J), overlap_fc_obs=len(set(plans['FC_QCDF']) & set(plans['OBS_QCDF'])),
               saved={k: v['saved_pct'] for k, v in Ev.items() if not k.startswith('_')}, none=Ev['_none']['loss'],
               var_weather=float(between), var_spread=float(within), frac_weather=float(between / (between + within)),
               wind_bias=float(err.mean()), wind_mae=float(np.abs(err).mean()), wind_mae_day=[float(np.abs(err[t]).mean()) for t in range(10)],
               ratio_med=float(np.median(ratio)), half_overlap=member_overlap, half_plans=[[int(Pobs.D.cells[i]) for i in h] for h in halves], member_base_cv=float(np.std(bases) / np.mean(bases)),
               plans={k: [int(Pobs.D.cells[i]) for i in v] for k, v in plans.items()})
    print(out['T0'], {k: round(v, 1) for k, v in out['saved'].items()}, 'wx frac', round(out['frac_weather'], 2), 'MAE', round(out['wind_mae'], 2), f'[{time.time()-t0:.0f}s]', flush=True)
    return out

if __name__ == '__main__':
    dates = sys.argv[1].split(','); outf = sys.argv[2]; res = []
    for d in dates:
        res.append(run(d)); json.dump(res, open(outf, 'w'))
