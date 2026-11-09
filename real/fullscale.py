"""Full-scale QCDF on the Dixie Fire: all domain cells, K crews, persistent protection."""
import sys, json, time, math, itertools, numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import spearmanr
sys.argv_saved = list(sys.argv); sys.argv = ['x']
import realcase as RC
sys.argv = sys.argv_saved
from fastsim import Sim

REACH, M_PAIR, R_A, R_P, R_B = 3, 120, 300, 1000, 2000
G, NB = RC.G, RC.NB

def rings_from_burned(burned):
    dist = np.full(len(G), 99); fr = list(np.where(burned)[0]); dist[fr] = 0
    for r in range(1, 9):
        nx = []
        for i in fr:
            for j in NB[i]:
                if dist[j] == 99: dist[j] = r; nx.append(j)
        fr = nx
    return dist

def weather_eta(D, eta_max, mode):
    if mode == 'const': return np.full(D.q.shape[0], eta_max)
    days = [RC.mday[D.T0 + pd.Timedelta(days=t)] for t in range(D.q.shape[0])]
    gi = D.cells
    erc = np.array([RC.met['erc'][d, gi].mean() for d in days]); ws = np.array([RC.met['vs'][d, gi].mean() for d in days])
    allerc = RC.met['erc'][:, gi].mean(1); allws = RC.met['vs'][:, gi].mean(1)
    ext = (erc >= np.percentile(allerc, 90)) | (ws >= np.percentile(allws, 90))
    return eta_max * np.where(ext, 0.4, 1.0)

import os
MODEL_MODE = os.environ.get('QCDF_MODEL', 'oracle')
def apply_model(D, mode):
    if mode == 'oracle': return
    if mode == 'prospective':
        C = json.load(open('calib_prospective.json')); beta = np.array(C['models']['wind+slope']['beta']); nm = C['norm']
    elif mode.startswith('file:'):
        C = json.load(open(mode[5:])); beta = np.array(C['models']['wind+slope']['beta']); nm = C['norm']
    else:
        C = json.load(open('calib_rolling.json')); beta = np.array(C['dates'][str(D.T0.date())]['beta']); nm = C['norm']
    days = [RC.mday[D.T0 + pd.Timedelta(days=t)] for t in range(RC.HORIZON)]
    gi, gj = D.cells[D.src], D.cells[D.dst]
    br = np.array([RC.bearing(i, j) for i, j in zip(gi, gj)])
    W = np.stack([RC.met['vs'][d, gi] * np.cos(np.radians(br - (RC.met['th'][d, gi] + 180) % 360)) for d in days])
    S = (RC.elev[gj] - RC.elev[gi]) / (2.4 * 1000)
    eta_ = beta[0] + beta[1] * (W - nm['W'][0]) / nm['W'][1] + beta[2] * (S - nm['S'][0]) / nm['S'][1]
    D.q = 1 / (1 + np.exp(-eta_))

class Problem:
    def __init__(s, T0, eta=0.85, eta_mode='const', S_prev=(), model=None):
        s.T0 = pd.Timestamp(T0); rng = np.random.default_rng(int(s.T0.strftime('%m%d')))
        s.D = RC.Domain(s.T0, rng); s.model = model or MODEL_MODE; apply_model(s.D, s.model)
        s.sim = Sim(s.D, eta); s.n = s.D.n
        s.eta = weather_eta(s.D, eta, eta_mode)
        s.loc = {int(g): k for k, g in enumerate(s.D.cells)}
        dist = rings_from_burned((G.ign_date <= s.T0).values); s.dist = dist[s.D.cells]
        s.elig = np.where((~s.D.burned0) & (s.dist >= 1) & (s.dist <= REACH))[0]
        s.S_prev = set(S_prev)
        s.adj = {k: [s.loc[j] for j in NB[g] if j in s.loc] for k, g in enumerate(s.D.cells)}
        obs = ((G.ign_date.values[s.D.cells] > np.datetime64(s.T0)) &
               (G.ign_date.values[s.D.cells] <= np.datetime64(s.T0 + pd.Timedelta(days=10))))
        s.obs_burn = obs
    def L(s, S, R, stream):
        return s.sim.losses(set(S) | s.S_prev if (S or s.S_prev) else [], R, stream, eta=s.eta)[0]
    def build_qubo(s, t_budget=None):
        t0 = time.time()
        base = s.L(set(), R_A, 11); s.base_A = base.mean()
        s.delta = np.zeros(s.n)
        for i in s.elig:
            s.delta[i] = base.mean() - s.L({i}, R_A, 11).mean()
        order = s.elig[np.argsort(-s.delta[s.elig])]
        top = list(order[:M_PAIR]); tops = set(top)
        pairs = sorted({(min(i, j), max(i, j)) for i in top for j in s.adj[i] if j in tops})
        baseP = s.L(set(), R_P, 12); single = {i: s.L({i}, R_P, 12) for i in top}
        s.J = {}; s.Jraw = {}
        for i, j in pairs:
            dd = s.L({i, j}, R_P, 12) - single[i] - single[j] + baseP
            I = -dd.mean(); se = dd.std(ddof=1) / math.sqrt(R_P)
            s.Jraw[(i, j)] = (float(I), float(se))
            if abs(I) >= 2 * se: s.J[(i, j)] = -I
        s.h = -s.delta
        s.t_qubo = time.time() - t0
        return s

    def energy(s, Z, J=None):
        J = s.J if J is None else J
        e = s.h[list(Z)].sum() if len(Z) else 0.0
        Zs = set(Z)
        for (i, j), v in J.items():
            if i in Zs and j in Zs: e += v
        return e
    def milp(s, K, use_J=True, cand=None):
        cand = s.elig if cand is None else cand
        idx = {c: k for k, c in enumerate(cand)}; n = len(cand)
        J = {p: v for p, v in s.J.items() if p[0] in idx and p[1] in idx} if use_J else {}
        P = list(J); m = len(P)
        c = np.concatenate([s.h[cand], np.array([J[p] for p in P])])
        A, lb, ub = [], [], []
        row = np.zeros(n + m); row[:n] = 1; A.append(row); lb.append(K); ub.append(K)
        for k, (i, j) in enumerate(P):
            a, b = idx[i], idx[j]
            if J[(i, j)] < 0:
                for x in (a, b):
                    r = np.zeros(n + m); r[n + k] = 1; r[x] = -1; A.append(r); lb.append(-np.inf); ub.append(0)
            else:
                r = np.zeros(n + m); r[n + k] = 1; r[a] = -1; r[b] = -1; A.append(r); lb.append(-1); ub.append(np.inf)
        res = milp(c, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.r_[np.ones(n), np.zeros(m)],
                   bounds=Bounds(0, 1), options=dict(time_limit=120))
        z = np.round(res.x[:n]).astype(int)
        return [int(cand[k]) for k in np.where(z == 1)[0]], res
    def sa(s, K, iters=20000, seed=0, init=None):
        rng = np.random.default_rng(seed); cand = list(s.elig)
        Z = list(init) if init is not None else list(rng.choice(cand, K, replace=False))
        Jn = {}
        for (i, j), v in s.J.items(): Jn.setdefault(i, []).append((j, v)); Jn.setdefault(j, []).append((i, v))
        inZ = set(Z); e = s.energy(Z); best = (e, list(Z))
        scale = np.std(s.h[cand]) + 1e-9
        for it in range(iters):
            T = scale * 2.0 * (0.005 / 2.0) ** (it / iters)
            a = Z[rng.integers(K)]; b = cand[rng.integers(len(cand))]
            if b in inZ: continue
            de = s.h[b] - s.h[a] + sum(v for (j, v) in Jn.get(b, []) if j in inZ and j != a) - sum(v for (j, v) in Jn.get(a, []) if j in inZ)
            if de <= 0 or rng.random() < math.exp(-de / T):
                Z[Z.index(a)] = b; inZ.discard(a); inZ.add(b); e += de
                if e < best[0]: best = (e, list(Z))
        return best[1]
    def celf(s, K, R=R_A, stream=11):
        base = s.L(set(), R, stream).mean(); S = []; cur = base
        gains = [(-(base - s.L({i}, R, stream).mean()), int(i), 0) for i in s.elig]
        import heapq; heapq.heapify(gains); evals = len(s.elig)
        while len(S) < K:
            g, i, it = heapq.heappop(gains)
            if it == len(S):
                S.append(i); cur = cur + g
            else:
                val = s.L(set(S) | {i}, R, stream).mean(); evals += 1
                heapq.heappush(gains, (val - cur, i, len(S)))
        return S, evals

    def baselines(s, K, seed=0):
        rng = np.random.default_rng(seed); E = s.elig; H = s.D.value
        out = {}
        out['B1_communities'] = list(E[np.argsort(-H[E], kind='stable')[:K]])
        front = E[s.dist[E] == 1]
        out['B2_nearest_front'] = list(front[np.argsort(-H[front], kind='stable')[:K]]) if len(front) >= K else \
            list(front) + list(E[np.argsort(s.dist[E] * 1e6 - H[E])][len(front):K])

        days = [RC.mday[s.T0 + pd.Timedelta(days=t)] for t in range(3)]
        gi = s.D.cells
        th = np.mean([(RC.met['th'][d, gi] + 180) % 360 for d in days], 0)
        lat, lon = RC.lat[gi], RC.lon[gi]; burned = s.D.burned0
        cl = np.array([np.mean([lat[k] for k in np.where(burned)[0]]), np.mean([lon[k] for k in np.where(burned)[0]])])
        bear = np.array([RC.bearing(int(np.argmin((RC.lat - cl[0]) ** 2 + (RC.lon - cl[1]) ** 2)), int(g)) for g in gi])
        align = np.cos(np.radians(bear - th.mean()))
        sc = (H[E] + 1) * (align[E] + 1.0) / (s.dist[E])
        out['B3_wind_downstream'] = list(E[np.argsort(-sc)[:K]])
        _, pb = s.sim.losses([], R_A, 13)
        var = H * pb
        out['B4_value_at_risk'] = list(E[np.argsort(-var[E])[:K]])
        out['B6_random'] = list(rng.choice(E, K, replace=False))
        return out

    def evaluate(s, plans):
        base = s.L(set(), R_B, 99); res = {}
        for k, S in plans.items():
            Lp = s.L(set(S), R_B, 99); d = base - Lp
            q90 = np.quantile(Lp, 0.9)
            res[k] = dict(saved=float(d.mean()), saved_pct=float(100 * d.mean() / base.mean()), se=float(d.std(ddof=1) / math.sqrt(R_B)),
                          cvar90=float(Lp[Lp >= q90].mean()), prec=float(s.obs_burn[list(S)].mean()) if len(S) else float('nan'),
                          housing_in_plan=float(s.D.value[list(S)].sum()), plan=[str(G.h3.iloc[s.D.cells[i]]) for i in S])
        res['_none'] = dict(loss=float(base.mean()), cvar90=float(base[base >= np.quantile(base, 0.9)].mean()))
        return res
    def fidelity(s, K, n=150, seed=5):
        rng = np.random.default_rng(seed); top = s.elig[np.argsort(-s.delta[s.elig])[:3 * K]]
        base = s.L(set(), 500, 77).mean(); pred, true = [], []
        for _ in range(n):
            S = list(rng.choice(top, K, replace=False))
            pred.append(-s.energy(S)); true.append(base - s.L(set(S), 500, 77).mean())
        pred, true = np.array(pred), np.array(true)
        r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)
        predh = np.array([-s.h[list(S)].sum() for S in [list(rng.choice(top, K, replace=False)) for _ in range(0)]]) if False else None
        return dict(spearman=float(spearmanr(pred, true)[0]), r2=float(r2), mean_pred=float(pred.mean()), mean_true=float(true.mean()))

def run_date(T0, Ks=(5, 10, 20), eta=0.85, eta_mode='const', S_prev=(), do_fid=True, verbose=True):
    t0 = time.time(); P = Problem(T0, eta, eta_mode, S_prev).build_qubo()
    out = dict(T0=str(pd.Timestamp(T0).date()), n=int(P.n), n_elig=int(len(P.elig)), n_pairs=len(P.Jraw), n_J=len(P.J),
               n_compl=int(sum(1 for v in P.J.values() if v < 0)), n_redund=int(sum(1 for v in P.J.values() if v > 0)),
               eta_mode=eta_mode, eta=eta, eta_days=P.eta.tolist(), t_qubo=P.t_qubo, K={})
    for K in Ks:
        tk = time.time(); plans = P.baselines(K)
        plans['QCDF_MILP'], res = P.milp(K); plans['QCDF_h_only'], _ = P.milp(K, use_J=False)
        plans['QCDF_SA'] = P.sa(K)
        plans['CELF_greedy'], ev = P.celf(K)
        E = P.evaluate(plans)
        rec = dict(eval=E, celf_evals=ev, milp_status=int(res.status), E_milp=float(P.energy(plans['QCDF_MILP'])),
                   E_sa=float(P.energy(plans['QCDF_SA'])), E_honly=float(P.energy(plans['QCDF_h_only'])), t=time.time() - tk)
        if do_fid: rec['fidelity'] = P.fidelity(K)
        out['K'][K] = rec
        if verbose:
            print(out['T0'], f"K={K} n={P.n} elig={len(P.elig)} J={len(P.J)} none={E['_none']['loss']:.0f} | " +
                  " ".join(f"{k.split('_')[0] if k.startswith('B') else k}={v['saved_pct']:.1f}" for k, v in E.items() if not k.startswith('_')) +
                  (f" | fid rho={rec['fidelity']['spearman']:.2f} R2={rec['fidelity']['r2']:.2f}" if do_fid else '') + f" [{time.time()-t0:.0f}s]", flush=True)
    return out, P

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        out, P = run_date('2021-07-31', Ks=(10,))
    elif mode == 'main':
        dates = sys.argv[2].split(','); outf = sys.argv[3]; res = []
        for d in dates:
            o, _ = run_date(d, Ks=(5, 10, 20)); res.append(o); json.dump(res, open(outf, 'w'))
    elif mode == 'pert':

        dates = sys.argv[2].split(','); files = sys.argv[3].split(','); outf = sys.argv[4]; res = []
        for f in files:
            for d in dates:
                P = Problem(d, model='file:' + f).build_qubo(); K = 10; plans = P.baselines(K)
                plans['QCDF_MILP'], _ = P.milp(K); plans['QCDF_h_only'], _ = P.milp(K, use_J=False)
                E = P.evaluate({k: plans[k] for k in ('QCDF_MILP', 'QCDF_h_only', 'B4_value_at_risk', 'B1_communities', 'B2_nearest_front')})
                res.append(dict(model=f, T0=d, none=E['_none']['loss'], n_J=len(P.J), **{k: v['saved_pct'] for k, v in E.items() if not k.startswith('_')}))
                print(f, d, {k: round(v['saved_pct'], 1) for k, v in E.items() if not k.startswith('_')}, flush=True)
                json.dump(res, open(outf, 'w'))
    elif mode == 'k10':
        dates = sys.argv[2].split(','); outf = sys.argv[3]; res = []
        for d in dates:
            o, _ = run_date(d, Ks=(10,), do_fid=False); res.append(o); json.dump(res, open(outf, 'w'))
    elif mode == 'late':
        dates = sys.argv[2].split(','); outf = sys.argv[3]; res = []
        for d in dates:
            o, _ = run_date(d, Ks=(10,), do_fid=False); res.append(o); json.dump(res, open(outf, 'w'))
