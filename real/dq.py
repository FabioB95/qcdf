"""Decision-quality checks of the simulator-derived surrogate (Dixie, prospective model, K=10)."""
import sys, json, math, time, numpy as np, pandas as pd, h3
_ARGV = list(sys.argv); sys.argv = ['x']
import fullscale as FS
sys.argv = _ARGV
from scipy.optimize import milp, LinearConstraint, Bounds
K = 10

def build(P, sa, sp):
    base = P.L(set(), FS.R_A, sa); delta = np.zeros(P.n)
    for i in P.elig: delta[i] = base.mean() - P.L({i}, FS.R_A, sa).mean()
    top = list(P.elig[np.argsort(-delta[P.elig])[:FS.M_PAIR]]); tops = set(top)
    pairs = sorted({(min(i, j), max(i, j)) for i in top for j in P.adj[i] if j in tops})
    bP = P.L(set(), FS.R_P, sp); single = {i: P.L({i}, FS.R_P, sp) for i in top}; J = {}; raw = {}
    for i, j in pairs:
        dd = P.L({i, j}, FS.R_P, sp) - single[i] - single[j] + bP; I = -dd.mean(); se = dd.std(ddof=1) / math.sqrt(FS.R_P)
        raw[(i, j)] = (I, se)
        if abs(I) >= 2 * se: J[(i, j)] = -I
    return -delta, J, raw, top, single, bP

def topk_plans(P, h, J, n_plans=30):
    cand = P.elig; idx = {c: k for k, c in enumerate(cand)}; n = len(cand)
    Jc = {p: v for p, v in J.items() if p[0] in idx and p[1] in idx}; Pl = list(Jc); m = len(Pl)
    c = np.concatenate([h[cand], np.array([Jc[p] for p in Pl])])
    A, lb, ub = [], [], []
    row = np.zeros(n + m); row[:n] = 1; A.append(row); lb.append(K); ub.append(K)
    for k, (i, j) in enumerate(Pl):
        a, b = idx[i], idx[j]
        if Jc[(i, j)] < 0:
            for x in (a, b):
                r = np.zeros(n + m); r[n + k] = 1; r[x] = -1; A.append(r); lb.append(-np.inf); ub.append(0)
        else:
            r = np.zeros(n + m); r[n + k] = 1; r[a] = -1; r[b] = -1; A.append(r); lb.append(-1); ub.append(np.inf)
    plans = []
    for _ in range(n_plans):
        res = milp(c, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.r_[np.ones(n), np.zeros(m)], bounds=Bounds(0, 1), options=dict(time_limit=60))
        if res.x is None: break
        z = np.round(res.x[:n]).astype(int); S = [int(cand[k]) for k in np.where(z == 1)[0]]; plans.append((float(res.fun), S))
        r = np.zeros(n + m); r[np.where(z == 1)[0]] = 1; A.append(r); lb.append(-np.inf); ub.append(K - 1)
    return plans

def full_greedy(P, R=FS.R_A, stream=11):
    S = []; cur = P.L(set(), R, stream).mean(); evals = 0
    for _ in range(K):
        best = None
        for i in P.elig:
            if i in S: continue
            v = P.L(set(S) | {int(i)}, R, stream).mean(); evals += 1
            if best is None or v < best[0]: best = (v, int(i))
        S.append(best[1]); cur = best[0]
    return S, evals

def run(T0):
    t0 = time.time(); P = FS.Problem(T0); out = dict(T0=T0)
    h, J, raw, top, single, bP = build(P, 11, 12); P.h, P.J = h, J
    base = P.L(set(), FS.R_B, 99).mean()
    val = lambda S: float(100 * (base - P.L(set(S), FS.R_B, 99).mean()) / base)

    plans = topk_plans(P, h, J); sim = [val(S) for _, S in plans]
    celf, _ = P.celf(K); fg, ev = full_greedy(P)
    out['a'] = dict(surr_E=[e for e, _ in plans], sim=sim, opt=sim[0], best_top=max(sim), argbest=int(np.argmax(sim)),
                    spearman=float(pd.Series([-e for e, _ in plans]).corr(pd.Series(sim), method='spearman')),
                    celf=val(celf), full_greedy=val(fg), same_celf_fg=len(set(celf) & set(fg)), fg_evals=ev)

    stab = []
    for sa, sp in ((21, 22), (31, 32)):
        h2, J2, raw2, top2, _, _ = build(P, sa, sp)
        common = set(raw) & set(raw2)
        sign = np.mean([np.sign(raw[p][0]) == np.sign(raw2[p][0]) for p in common if abs(raw[p][0]) >= 2 * raw[p][1]]) if common else float('nan')
        P2h, P2J = P.h, P.J; P.h, P.J = h2, J2; S2, _ = P.milp(K); P.h, P.J = P2h, P2J
        S1, _ = P.milp(K)
        stab.append(dict(corr_h=float(np.corrcoef(h[P.elig], h2[P.elig])[0, 1]), sign_agree=float(sign), nJ=(len(J), len(J2)),
                         overlap=len(set(S1) & set(S2)), val1=val(S1), val2=val(S2)))
    out['b'] = stab

    rng = np.random.default_rng(3); cells = [str(P.D.cells[i]) for i in top]
    hx = {i: FS.G.h3.iloc[P.D.cells[i]] for i in top}; cand = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            d = h3.grid_distance(hx[top[a]], hx[top[b]])
            if 2 <= d <= 4: cand.append((top[a], top[b], d))
    sel = [cand[k] for k in rng.choice(len(cand), min(150, len(cand)), replace=False)]
    na = []
    for i, j, d in sel:
        dd = P.L({i, j}, FS.R_P, 12) - single[i] - single[j] + bP; na.append((i, j, d, float(-dd.mean()), float(dd.std(ddof=1) / math.sqrt(FS.R_P))))
    adjI = [abs(v[0]) for p, v in raw.items() if v[1] > 0 and abs(v[0]) >= 2 * v[1]]
    sig = [x for x in na if x[4] > 0 and abs(x[3]) >= 2 * x[4]]
    J3 = dict(J); J3.update({(min(i, j), max(i, j)): -I for i, j, d, I, se in sig})
    P.J = J3; S3, _ = P.milp(K); P.J = J; S1, _ = P.milp(K)
    out['c'] = dict(pairs=na, n_zero=sum(1 for x in na if x[4] == 0), n_pairs=len(na), n_sig=len(sig), n_sig_red=sum(1 for x in sig if x[3] < 0), n_sig_comp=sum(1 for x in sig if x[3] > 0),
                    by_dist={d: [sum(1 for x in sig if x[2] == d), sum(1 for x in na if x[2] == d)] for d in (2, 3, 4)},
                    med_abs_sig=float(np.median([abs(x[3]) for x in sig])) if sig else 0.0, med_abs_adj=float(np.median(adjI)) if adjI else 0.0,
                    overlap=len(set(S1) & set(S3)), val_with=val(S3), val_without=val(S1))
    print(T0, 'a', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out['a'].items() if k not in ('surr_E', 'sim')},
          'b', [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for s in stab], 'c', {k: v for k, v in out['c'].items()}, f'[{time.time()-t0:.0f}s]', flush=True)
    return out

if __name__ == '__main__':
    dates = sys.argv[1].split(','); outf = sys.argv[2]; res = []
    for d in dates:
        res.append(run(d)); json.dump(res, open(outf, 'w'), default=float)
