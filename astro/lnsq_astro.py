"""Threshold experiment of lnsq.py on the astronomical Hamiltonian (same LNS, windows, arms)."""
import sys, json, pickle, time, numpy as np
sys.path.insert(0, '../real')
import astro_full as AF
import lnsq as LQ

class Ham:
    def __init__(s, f):
        d = pickle.load(open(f, 'rb')); s.h = d['h']; s.J = d['J']; s.adj = d['adj']; s.n = len(s.h)
        s.elig = np.arange(s.n); s.eset = set(range(s.n))
    energy = AF.Sky.energy
    milp = AF.Sky.milp

if __name__ == '__main__':
    P = Ham(sys.argv[1]); out = sys.argv[2]; res = []; t0 = time.time()
    for K in (20, 50):
        Zm, _ = P.milp(K); Em = P.energy(Zm); Z0 = list(np.argsort(P.h)[:K])

        for w, S in [(w, S) for S in (16, 128) for w in (8, 12, 16, 20, 24)]:
            rec = dict(domain='astro', K=K, w=w, S=S, E_milp=float(Em), gap0=float((P.energy(Z0) - Em) / abs(Em)), arms={})
            for arm in ['exact', 'uniform'] + [f'q{q}' for q in LQ.QS]:
                q = float(arm[1:]) if arm.startswith('q') else None
                rec['arms'][arm] = [LQ.run(P, arm, q, s, Z0, Em, w, S) for s in range(LQ.SEEDS)]
            fin = {a: np.mean([r['gap'][-1] for r in v]) for a, v in rec['arms'].items()}
            print(K, w, S, 'gap0', round(rec['gap0'], 4), {a: round(float(v), 4) for a, v in fin.items()}, f'[{time.time()-t0:.0f}s]', flush=True)
            res.append(rec); json.dump(res, open(out, 'w'))
