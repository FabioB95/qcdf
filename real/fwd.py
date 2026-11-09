"""Forward (non-anchored) closed-loop study on the Dixie Fire, prospective spread model."""
import sys, json, time, math, numpy as np, pandas as pd, numba as nb
sys.argv_saved = list(sys.argv); sys.argv = ['x']
import realcase as RC
sys.argv = sys.argv_saved
from fastsim import _sim, _u01
from scipy.optimize import milp, LinearConstraint, Bounds

G, NB = RC.G, RC.NB
T0 = pd.Timestamp('2021-07-24'); TEND = pd.Timestamp('2021-08-15')
ACTIVE = ['2021-07-24', '2021-07-25', '2021-07-27', '2021-07-28', '2021-07-29', '2021-07-30', '2021-07-31', '2021-08-01', '2021-08-04', '2021-08-05']
DOFF = [(pd.Timestamp(d) - T0).days for d in ACTIVE]
H_ALL = (TEND - T0).days + 1
HZ, REACH, ETA, KNEW = 10, 3, 0.85, 5
R_IN, R_PAIR, M_PAIR = 200, 400, 60

burn_end = (G.ign_date <= TEND).values
dist = np.full(len(G), 99); fr = list(np.where(burn_end)[0]); dist[fr] = 0
for r in range(1, 9):
    nx = []
    for i in fr:
        for j in NB[i]:
            if dist[j] == 99: dist[j] = r; nx.append(j)
    fr = nx
cells = np.where(dist <= 8)[0]; loc = {int(c): k for k, c in enumerate(cells)}; n = len(cells)
src, dst = [], []
for a, i in enumerate(cells):
    for j in NB[i]:
        if j in loc: src.append(a); dst.append(loc[j])
src, dst = np.array(src, np.int64), np.array(dst, np.int64)
adj = [[] for _ in range(n)]
for a, b in zip(src, dst): adj[a].append(b)
value = RC.HOUS[cells].astype(np.float64)
b_obs0 = (G.ign_date <= T0).values[cells].astype(np.uint8)

C = json.load(open('calib_prospective.json')); beta = np.array(C['models']['wind+slope']['beta']); nm = C['norm']
days = [RC.mday[T0 + pd.Timedelta(days=t)] for t in range(H_ALL + HZ)]
gi, gj = cells[src], cells[dst]
br = np.array([RC.bearing(i, j) for i, j in zip(gi, gj)])
W = np.stack([RC.met['vs'][d, gi] * np.cos(np.radians(br - (RC.met['th'][d, gi] + 180) % 360)) for d in days])
S = (RC.elev[gj] - RC.elev[gi]) / (2.4 * 1000)
Q = 1 / (1 + np.exp(-(beta[0] + beta[1] * (W - nm['W'][0]) / nm['W'][1] + beta[2] * (S - nm['S'][0]) / nm['S'][1])))
Q = np.ascontiguousarray(Q)

@nb.njit(cache=True)
def segment(b, q, prot, t0, t1, r, stream, src, dst):
    n = b.shape[0]; nb_ = np.zeros(n, np.uint8)
    for t in range(t0, t1):
        for k in range(n): nb_[k] = 0
        for e in range(src.shape[0]):
            i = src[e]; j = dst[e]
            if b[i] and not b[j]:
                if _u01(r, t, e, stream) < q[t, e] * (1.0 - prot[j]): nb_[j] = 1
        for k in range(n):
            if nb_[k]: b[k] = 1

def inner_losses(b, t, units, R, stream):
    P = np.zeros((HZ, n)); idx = np.fromiter(units, np.int64) if units else np.zeros(0, np.int64)
    P[:, idx] = ETA
    L = np.zeros(R); F = np.zeros(n, np.int64)
    _sim(src, dst, np.ascontiguousarray(Q[t:t + HZ]), b.astype(np.uint8), value, P, 0.0, R, stream, L, F)
    return L, F / R

def eligible(b):
    d = np.full(n, 99); fr = list(np.where(b)[0]); d[fr] = 0
    for r in range(1, REACH + 1):
        nx = []
        for i in fr:
            for j in adj[i]:
                if d[j] == 99: d[j] = r; nx.append(j)
        fr = nx
    return np.where((b == 0) & (d >= 1) & (d <= REACH))[0]

def milp_k(h, J, cand, K):
    idx = {c: k for k, c in enumerate(cand)}; m_ = len(cand)
    Jc = {p: v for p, v in J.items() if p[0] in idx and p[1] in idx}; Pl = list(Jc); m = len(Pl)
    c = np.concatenate([h[cand], np.array([Jc[p] for p in Pl])])
    A, lb, ub = [], [], []
    row = np.zeros(m_ + m); row[:m_] = 1; A.append(row); lb.append(K); ub.append(K)
    for k, (i, j) in enumerate(Pl):
        a, b_ = idx[i], idx[j]
        if Jc[(i, j)] < 0:
            for x in (a, b_):
                r_ = np.zeros(m_ + m); r_[m_ + k] = 1; r_[x] = -1; A.append(r_); lb.append(-np.inf); ub.append(0)
        else:
            r_ = np.zeros(m_ + m); r_[m_ + k] = 1; r_[a] = -1; r_[b_] = -1; A.append(r_); lb.append(-1); ub.append(np.inf)
    res = milp(c, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.r_[np.ones(m_), np.zeros(m)],
               bounds=Bounds(0, 1), options=dict(time_limit=60))
    z = np.round(res.x[:m_]).astype(int)
    return [int(cand[k]) for k in np.where(z == 1)[0]]

def plan(b, t, units, K, stream, need='QCDF'):
    units = set(units); cand = np.array([c for c in eligible(b) if c not in units])
    if len(cand) <= K: c_ = [int(c) for c in cand]; return dict(QCDF=c_, h_only=c_, VaR=c_)
    base, pb = inner_losses(b, t, units, R_IN, stream)
    var = value * pb
    if need == 'VaR': return dict(VaR=[int(c) for c in cand[np.argsort(-var[cand])][:K]])
    h = np.zeros(n)
    for i in cand: h[i] = -(base.mean() - inner_losses(b, t, units | {int(i)}, R_IN, stream)[0].mean())
    if need == 'h_only': return dict(h_only=milp_k(h, {}, cand, K))
    top = list(cand[np.argsort(h[cand])[:M_PAIR]]); tops = set(top)
    pairs = sorted({(min(i, j), max(i, j)) for i in top for j in adj[i] if j in tops})
    bP = inner_losses(b, t, units, R_PAIR, stream + 1)[0]; single = {i: inner_losses(b, t, units | {int(i)}, R_PAIR, stream + 1)[0] for i in top}
    J = {}
    for i, j in pairs:
        dd = inner_losses(b, t, units | {int(i), int(j)}, R_PAIR, stream + 1)[0] - single[i] - single[j] + bP
        I = -dd.mean(); se = dd.std(ddof=1) / math.sqrt(R_PAIR)
        if abs(I) >= 2 * se: J[(i, j)] = -I
    var = value * pb
    return dict(QCDF=milp_k(h, J, cand, K), h_only=milp_k(h, {}, cand, K),
                VaR=[int(c) for c in cand[np.argsort(-var[cand])][:K]], _h=h, _J=J, _cand=cand)

def run_outer(r, open_orders, stream_out=777):
    res = {}
    for name in ['none', 'OL_QCDF', 'OL_VaR', 'CL_QCDF', 'CL_h_only', 'CL_VaR']:
        b = b_obs0.copy(); units = []; tprev = 0
        for k, t in enumerate(DOFF):
            prot = np.zeros(n); prot[units] = ETA
            segment(b, Q, prot, tprev, t, r, stream_out, src, dst); tprev = t
            if name == 'none': continue
            if name.startswith('OL'): units = list(open_orders[name][:KNEW * (k + 1)])
            else:
                key = name[3:]
                pl = plan(b, t, units, KNEW, 5000 + 10 * k, need=key)
                units = units + [c for c in pl[key] if c not in units]
        prot = np.zeros(n); prot[units] = ETA
        segment(b, Q, prot, tprev, H_ALL, r, stream_out, src, dst)
        res[name] = float(((b == 1) & (b_obs0 == 0)) @ value); res[name + '_cells'] = int(((b == 1) & (b_obs0 == 0)).sum())
    return res

def open_orders():
    b = b_obs0.copy(); pl = plan(b, 0, [], KNEW * len(DOFF), 4000)
    h = pl['_h']; ol_q = sorted(pl['QCDF'], key=lambda i: h[i])
    return {'OL_QCDF': ol_q, 'OL_VaR': pl['VaR']}

if __name__ == '__main__':
    r0, r1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    print('domain cells', n, 'edges', len(src), 'burned at T0', int(b_obs0.sum()), 'burned obs by 15 Aug', int(burn_end[cells].sum()), flush=True)
    t0 = time.time(); OO = open_orders(); print('open plans', f'[{time.time()-t0:.0f}s]', flush=True)
    res = []
    for r in range(r0, r1):
        t0 = time.time(); x = run_outer(r, OO); x['r'] = r; res.append(x)
        print(r, {k: round(v) for k, v in x.items() if not k.endswith('_cells')}, f'[{time.time()-t0:.0f}s]', flush=True)
        json.dump(dict(open=OO, runs=res), open(out, 'w'))
