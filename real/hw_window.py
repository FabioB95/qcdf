"""Export one representative LNS window of the full-scale Dixie problem (12 cells, k=3) as a hardware instance: QUBO coefficients, XY-mixer edges, exact optimum, energy-optimal p=1 and p=2 angles."""
import sys, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '../out')
from qcdf_scaling import DickeSpace, qaoa_state, qaoa_energy
import fullscale as FS, lns as LN
LN.N_SUB = 12
P = FS.Problem('2021-07-31').build_qubo(); Zm, _ = P.milp(10)
rng = np.random.default_rng(3); cands = []
for _ in range(400):
    W = LN.window(P, rng, Zm); pr = LN.subproblem(P, W, Zm)
    if pr is not None and pr['ks'] == 3 and len(W) == 12 and len(pr['sp'].edges) >= 12: cands.append(pr)
gaps = [np.sort(pr['E'])[1] - pr['E'].min() for pr in cands]
pr = cands[int(np.argsort(gaps)[len(gaps) // 2])]
sp, E = pr['sp'], pr['E']; En = (E - E.mean()) / E.std(); o = int(np.argmin(E))
out = dict(heff=pr['heff'].tolist(), Jin=[[int(a), int(b), float(v)] for a, b, v in pr['Jin']], date='2021-07-31', K=10, cells=[str(FS.G.h3.iloc[P.D.cells[c]]) for c in pr['W']], k=3,
           edges=[list(map(int, e)) for e in sp.edges], E_feasible=E.tolist(), Z=sp.Z.tolist(),
           opt=''.join(map(str, sp.Z[o])), P_uniform=1 / sp.dim)
for p in (1, 2):
    best = None; r = np.random.default_rng(p)
    for _ in range(100):
        x0 = np.concatenate([r.uniform(0, 2 * np.pi, p), r.uniform(0, np.pi, p)])
        res = minimize(qaoa_energy, x0, args=(sp, En), method='L-BFGS-B', bounds=[(0, 2 * np.pi)] * p + [(0, np.pi)] * p)
        best = res if best is None or res.fun < best.fun else best
    out[f'angles_p{p}'] = best.x.tolist(); out[f'P_opt_p{p}'] = float((np.abs(qaoa_state(best.x, sp, En)) ** 2)[o])
print({k: v for k, v in out.items() if k not in ('E_feasible', 'Z')})
json.dump(out, open('hw_window.json', 'w'), indent=1)
