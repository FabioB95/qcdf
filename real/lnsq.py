"""Success-probability threshold for constraint-preserving sub-solvers inside LNS."""
import sys, json, time, math, itertools, numpy as np

KMAX, ITERS, SEEDS = 6, 500, 8
S_SHOTS = 128
QS = (0.01, 0.03, 0.1, 0.3, 0.6)
_Z = {}
def zmat(w, k):
    if (w, k) not in _Z:
        Z = np.zeros((math.comb(w, k), w), np.int8)
        for a, c in enumerate(itertools.combinations(range(w), k)): Z[a, list(c)] = 1
        _Z[(w, k)] = Z
    return _Z[(w, k)]

def window(P, rng, Z, w):
    elig = P.eset
    pool = list(Z) + list(P.elig[np.argsort(P.h[P.elig])[:3 * len(Z)]])
    seed = pool[rng.integers(len(pool))]
    W = [seed]; seen = {seed}; k = 0
    while len(W) < w and k < len(W):
        for j in P.adj[W[k]]:
            if j in elig and j not in seen and len(W) < w: seen.add(j); W.append(j)
        k += 1
    return W

def subproblem(P, W, Z):
    Zs = set(Z); inW = set(W); ks = len(Zs & inW)
    if ks == 0 or ks > KMAX or len(W) < 4: return None
    loc = {c: a for a, c in enumerate(W)}
    heff = P.h[W].astype(float).copy(); Zm = zmat(len(W), ks).astype(float)
    E = Zm @ heff
    for (i, j), v in P.J.items():
        if i in inW and j in inW: E = E + v * Zm[:, loc[i]] * Zm[:, loc[j]]
        elif i in inW and j in Zs: E = E + v * Zm[:, loc[i]]
        elif j in inW and i in Zs: E = E + v * Zm[:, loc[j]]
    cur = np.zeros(len(W)); cur[[loc[c] for c in Zs & inW]] = 1
    e_cur = float(cur @ heff + sum(v for (i, j), v in P.J.items() if i in inW and j in inW and cur[loc[i]] and cur[loc[j]]))

    for (i, j), v in P.J.items():
        if i in inW and j in Zs and j not in inW and cur[loc[i]]: e_cur += v
        elif j in inW and i in Zs and i not in inW and cur[loc[j]]: e_cur += v
    return dict(W=W, ks=ks, Zm=Zm, E=E, e_cur=e_cur, rest=[c for c in Z if c not in inW])

def run(P, arm, q, seed, Z0, Eref, w, S):
    rng = np.random.default_rng(seed); Z = list(Z0); e = P.energy(Z); trace = []; pe, ue = [], []; used = 0; dims = []
    for it in range(ITERS):
        pr = subproblem(P, window(P, rng, Z, w), Z)
        if pr is None: trace.append(e); continue
        used += 1; E = pr['E']; o = int(np.argmin(E)); dims.append(len(E))
        s = rng.integers(len(E), size=S); uni = int(s[np.argmin(E[s])])
        pe.append(E[o] < pr['e_cur'] - 1e-9); ue.append(E[uni] < pr['e_cur'] - 1e-9)
        if arm == 'exact': a = o
        elif arm == 'uniform': a = uni
        else: a = o if rng.random() < q else uni
        if E[a] < pr['e_cur'] - 1e-9:
            Z = pr['rest'] + [pr['W'][k] for k in np.where(pr['Zm'][a] == 1)[0]]; e = P.energy(Z)
        trace.append(e)
    return dict(gap=[(x - Eref) / abs(Eref) for x in trace], dimF=dims, p=float(np.mean(pe)) if pe else None, u=float(np.mean(ue)) if ue else None,
                pe=[bool(x) for x in pe], ue=[bool(x) for x in ue], used=used)

if __name__ == '__main__':
    import fullscale as FS
    dates = sys.argv[1].split(','); out = sys.argv[2]; res = []
    for d in dates:
        t0 = time.time(); P = FS.Problem(d).build_qubo(); P.eset = set(P.elig.tolist())
        print(d, 'qubo', f'[{time.time()-t0:.0f}s]', 'elig', len(P.elig), 'J', len(P.J), flush=True)
        for K in (10, 20):
            Zm, _ = P.milp(K); Em = P.energy(Zm); Z0 = list(P.elig[np.argsort(P.h[P.elig])[:K]])
            for w, S in [(w, S) for S in (16, 128) for w in (8, 12, 16, 20, 24)]:
                rec = dict(T0=d, K=K, w=w, S=S, E_milp=float(Em), gap0=float((P.energy(Z0) - Em) / abs(Em)), arms={})
                for arm in ['exact', 'uniform'] + [f'q{q}' for q in QS]:
                    q = float(arm[1:]) if arm.startswith('q') else None
                    rec['arms'][arm] = [run(P, arm, q, s, Z0, Em, w, S) for s in range(SEEDS)]
                fin = {a: np.mean([r['gap'][-1] for r in v]) for a, v in rec['arms'].items()}
                print(d, K, w, S, 'gap0', round(rec['gap0'], 4), {a: round(v, 4) for a, v in fin.items()}, f'[{time.time()-t0:.0f}s]', flush=True)
                res.append(rec); json.dump(res, open(out, 'w'))
