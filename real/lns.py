"""Quantum layer at full scale: large-neighbourhood search (LNS) on the simulator-derived QCDF Hamiltonian of a whole Dixie domain, with constraint-preserving sub-solvers on windows of N_SUB cells."""
import sys, json, math, time, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, '../out'); sys.path.insert(0, '../runs')
from qcdf_scaling import DickeSpace, qaoa_state
import anneal as A
import fullscale as FS

N_SUB, KMAX, S_SHOTS, P_DEPTH = 16, 6, 128, 3

def window(P, rng, Z):
    elig = set(P.elig.tolist())
    pool = list(Z) + list(P.elig[np.argsort(P.h[P.elig])[:3 * len(Z)]])
    seed = pool[rng.integers(len(pool))]
    W = [seed]; seen = {seed}; k = 0
    while len(W) < N_SUB and k < len(W):
        for j in P.adj[W[k]]:
            if j in elig and j not in seen and len(W) < N_SUB: seen.add(j); W.append(j)
        k += 1
    return W

def subproblem(P, W, Z):
    Zs = set(Z); inW = set(W); ks = len(Zs & inW)
    if ks == 0 or ks > KMAX or len(W) < 4: return None
    loc = {c: a for a, c in enumerate(W)}
    heff = P.h[W].astype(float).copy(); Jin = []
    for (i, j), v in P.J.items():
        if i in inW and j in inW: Jin.append((loc[i], loc[j], v))
        elif i in inW and j in Zs: heff[loc[i]] += v
        elif j in inW and i in Zs: heff[loc[j]] += v
    ed = sorted({(min(loc[a], loc[b]), max(loc[a], loc[b])) for a in W for b in P.adj[a] if b in inW})
    sp = DickeSpace(len(W), ks, ed if ed else [(0, 1)])
    E = sp.Z @ heff
    for a, b, v in Jin: E = E + v * sp.Z[:, a] * sp.Z[:, b]
    return dict(W=W, ks=ks, sp=sp, E=E, rest=[c for c in Z if c not in inW], heff=heff, Jin=Jin)

def train_angles(problems, p=P_DEPTH, seed=0):
    rng = np.random.default_rng(seed)
    def f(x):
        tot = 0
        for pr in problems:
            En = (pr['E'] - pr['E'].mean()) / (pr['E'].std() + 1e-12); psi = qaoa_state(x, pr['sp'], En)
            tot += np.real(np.vdot(psi, En * psi))
        return tot / len(problems)
    best = None
    for r in range(6):
        x0 = np.concatenate([rng.uniform(0, 1.0, p), rng.uniform(0, 1.0, p)]) if r else np.r_[np.linspace(0.2, 0.8, p), np.linspace(0.8, 0.2, p)]
        res = minimize(f, x0, method='L-BFGS-B', bounds=[(0, 2 * np.pi)] * p + [(0, np.pi)] * p)
        best = res if best is None or res.fun < best.fun else best
    return best.x

def sample_arm(arm, pr, rng, angles=None, q=None):
    E, sp = pr['E'], pr['sp']; o = int(np.argmin(E))
    if arm == 'exact': return o, 1.0
    if arm == 'uniform':
        s = rng.integers(sp.dim, size=S_SHOTS); return int(s[np.argmin(E[s])]), 1 / sp.dim
    if arm.startswith('oracle'):
        if rng.random() < q: return o, q
        s = rng.integers(sp.dim, size=S_SHOTS); return int(s[np.argmin(E[s])]), q
    En = (E - E.mean()) / (E.std() + 1e-12)
    if arm == 'qaoa': pr_ = np.abs(qaoa_state(angles, sp, En)) ** 2
    elif arm == 'anneal':
        class _S: pass
        s_ = _S(); s_.pairs = sp.pairs; s_.dim = sp.dim; A._gs.clear()
        pr_ = A.anneal(s_, En, np.full(len(sp.pairs), 1e9), 20.0, dt=0.1)
    pr_ = pr_ / pr_.sum(); s = rng.choice(sp.dim, size=S_SHOTS, p=pr_)
    return int(s[np.argmin(E[s])]), float(pr_[o])

def run_lns(P, K, arm, iters, seed, Z0, E_ref, angles=None, q=None):
    rng = np.random.default_rng(seed); Z = list(Z0); e = P.energy(Z); trace = []; popt = []; acc = 0; used = 0
    for it in range(iters):
        W = window(P, rng, Z); pr = subproblem(P, W, Z)
        if pr is None: trace.append(e); continue
        a, po = sample_arm(arm, pr, rng, angles, q); used += 1; popt.append(po)
        newZ = pr['rest'] + [pr['W'][k] for k in np.where(pr['sp'].Z[a] == 1)[0]]
        ne = P.energy(newZ)
        if ne < e - 1e-9: Z, e = newZ, ne; acc += 1
        trace.append(e)
    return dict(Z=[int(z) for z in Z], E=float(e), gap=float((e - E_ref) / abs(E_ref)), trace=[float(x) for x in trace],
                acc=acc, used=used, popt_mean=float(np.mean(popt)) if popt else None)

if __name__ == '__main__':
    dates = sys.argv[1].split(','); outf = sys.argv[2]; K = 20; ITERS = 200; SEEDS = 3
    res = []; angles = None
    for d in dates:
        t0 = time.time(); P = FS.Problem(d).build_qubo()
        Zm, _ = P.milp(K); Em = P.energy(Zm)
        Z0 = list(P.elig[np.argsort(P.h[P.elig])[:K]])
        if angles is None:
            rng = np.random.default_rng(1); probs = []
            while len(probs) < 40:
                Zr = list(rng.choice(P.elig, K, replace=False)) if len(probs) % 2 else Z0
                pr = subproblem(P, window(P, rng, Zr), Zr)
                if pr is not None: probs.append(pr)
            angles = train_angles(probs); print('angles', np.round(angles, 4), flush=True)
        rec = dict(T0=d, K=K, E_milp=float(Em), E_start=float(P.energy(Z0)), angles=angles.tolist(), arms={})
        for arm in ['exact', 'uniform', 'qaoa', 'anneal'] + [f'oracle_{q}' for q in (0.02, 0.1, 0.3, 1.0)]:
            q = float(arm.split('_')[1]) if arm.startswith('oracle') else None
            rr = [run_lns(P, K, arm, ITERS, s, Z0, Em, angles, q) for s in range(SEEDS)]
            rec['arms'][arm] = dict(gap=[r['gap'] for r in rr], popt=[r['popt_mean'] for r in rr], acc=[r['acc'] for r in rr],
                                    used=[r['used'] for r in rr], trace=rr[0]['trace'], Z=[r['Z'] for r in rr])
            print(d, arm, 'gap', np.round([r['gap'] for r in rr], 4), 'popt', np.round([r['popt_mean'] or 0 for r in rr], 4), f"[{time.time()-t0:.0f}s]", flush=True)

        plans = {'MILP': Zm, 'start_h_only': Z0}
        for arm in ['exact', 'uniform', 'qaoa', 'anneal']: plans['LNS_' + arm] = rec['arms'][arm]['Z'][0]
        ev = P.evaluate(plans); rec['eval'] = {k: v['saved_pct'] for k, v in ev.items() if not k.startswith('_')}
        print(d, 'eval', {k: round(v, 1) for k, v in rec['eval'].items()}, flush=True)
        res.append(rec); json.dump(res, open(outf, 'w'))
