"""Real-data QCDF wildfire case (Dixie Fire 2021, H3 res 7), using the spread model calibrated on NIFC perimeters (calibrate.py) and observed gridMET weather (hindcast)."""
import sys, json, math, itertools, time
import numpy as np, pandas as pd, geopandas as gpd, h3
from scipy.optimize import minimize
sys.path.insert(0, '../runs'); sys.path.insert(0, '../out')
import anneal as A
from qcdf_scaling import DickeSpace, qaoa_state, qaoa_energy

ETA, HORIZON, T, RINGS, N_CAND, N_STAT, K_STAT = 0.85, 10, 4, 8, 20, 12, 3
RA, RB, M_FIT = 300, 1000, 2000

import os
FIRE_DIR = os.environ.get('QCDF_FIRE', '')
G = gpd.read_file(FIRE_DIR + 'grid_r7.gpkg'); G['ign_date'] = pd.to_datetime(G['ign_date'])
CAL = json.load(open('calib.json')); MODEL = CAL['models']['wind+slope']
beta, norm = np.array(MODEL['beta']), CAL['norm']
met = np.load(FIRE_DIR + 'met.npz'); mdays = pd.to_datetime(met['days']); mday = {d: k for k, d in enumerate(mdays)}
idx = {c: k for k, c in enumerate(G.h3)}
NB = [[idx[x] for x in h3.grid_ring(c, 1) if x in idx] for c in G.h3]
lat, lon, elev = G.lat.values, G.lon.values, G.elev.values.astype(float)
HOUS = G.HOUSING20.values.astype(float)

def bearing(i, j):
    dlon = math.radians(lon[j] - lon[i]); la1, la2 = math.radians(lat[i]), math.radians(lat[j])
    x = math.sin(dlon) * math.cos(la2); y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

class Domain:
    def __init__(s, T0, rng):
        s.T0 = T0
        burned = (G.ign_date <= T0).values

        dist = np.full(len(G), 99); frontier = list(np.where(burned)[0]); dist[frontier] = 0
        for r in range(1, RINGS + 1):
            nxt = []
            for i in frontier:
                for j in NB[i]:
                    if dist[j] == 99: dist[j] = r; nxt.append(j)
            frontier = nxt
        keep = np.where((dist <= RINGS) & ((~burned) | np.array([any(not burned[j] for j in NB[i]) for i in range(len(G))])))[0]
        s.cells = keep; loc = {c: k for k, c in enumerate(keep)}; s.n = len(keep)
        s.burned0 = burned[keep]

        src, dst = [], []
        for a, i in enumerate(keep):
            for j in NB[i]:
                if j in loc: src.append(a); dst.append(loc[j])
        s.src, s.dst = np.array(src), np.array(dst)
        days = [mday[T0 + pd.Timedelta(days=t)] for t in range(HORIZON)]
        gi, gj = keep[s.src], keep[s.dst]
        W = np.stack([met['vs'][d, gi] * np.cos(np.radians(np.array([bearing(i, j) for i, j in zip(gi, gj)]) - (met['th'][d, gi] + 180) % 360)) for d in days])
        S = (elev[gj] - elev[gi]) / (CAL['DIST'] * 1000)
        D = np.stack([-met['fm1000'][d, gj] for d in days]); E = np.stack([met['erc'][d, gj] for d in days])
        feats = dict(W=(W - norm['W'][0]) / norm['W'][1], S=np.broadcast_to((S - norm['S'][0]) / norm['S'][1], W.shape),
                     D=(D - norm['D'][0]) / norm['D'][1], E=(E - norm['E'][0]) / norm['E'][1])
        X = [np.ones_like(W)] + [feats[k] for k in MODEL['use']]
        s.q = 1 / (1 + np.exp(-sum(b * x for b, x in zip(beta, X))))
        s.value = HOUS[keep]
        s.UA = rng.random((RA, HORIZON, len(s.src))); s.UB = rng.random((RB, HORIZON, len(s.src)))
        s.dryness = D.mean(0); s.erc0 = met['erc'][days[0], keep]

    def run(s, prot, U):
        R = U.shape[0]
        burned = np.repeat(s.burned0[None], R, 0)
        for t in range(HORIZON):
            q = s.q[t].copy()
            if t < len(prot) and len(prot[t]):
                q[np.isin(s.dst, prot[t])] *= (1 - ETA)
            fire = burned[:, s.src] & (U[:, t] < q[None])
            burned |= s._ign(fire)
        newly = burned & ~s.burned0[None]
        return float((newly * s.value).sum(1).mean()), newly.mean(0)

    def _ign(s, fire):
        if not hasattr(s, 'Dmat'):
            from scipy.sparse import csr_matrix
            s.Dmat = csr_matrix((np.ones(len(s.dst), np.float32), (np.arange(len(s.dst)), s.dst)), shape=(len(s.dst), s.n))
        return (s.Dmat.T @ fire.T.astype(np.float32)).T > 0

def plan_loss(D, plan, U):
    return D.run([np.array([D.cand[c]]) for c in plan], U)[0]

def solve_static_qcdf(D, rng):
    C = D.cand_all[:N_STAT]; gi = D.cells[C]
    _, pb = D.run([], D.UA)
    p = pb[C]; mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
    v = mm(D.value[C] * p)
    slope = np.array([np.mean([abs(elev[j] - elev[i]) for j in NB[i]]) for i in gi])
    c = mm(mm(G.road_km.values[gi]) + mm(slope))
    r = mm(D.erc0[C])
    u = mm(np.sqrt(np.clip(p * (1 - p), 0, None)))
    h = 1.0 * c + 0.5 * r + 0.3 * u - 2.0 * v
    loc = {int(x): k for k, x in enumerate(gi)}
    ed = sorted({tuple(sorted((loc[i], loc[j]))) for i in gi for j in NB[i] if j in loc})
    sp = DickeSpace(N_STAT, K_STAT, ed if ed else [(0, 1)]); E = sp.energies(h, 0.3 if ed else 0.0)
    best = None
    for _ in range(30):
        x0 = np.concatenate([rng.uniform(0, 2 * np.pi, 4), rng.uniform(0, np.pi, 4)])
        rr = minimize(qaoa_energy, x0, args=(sp, E), method='L-BFGS-B', bounds=[(0, 2 * np.pi)] * 4 + [(0, np.pi)] * 4)
        best = rr if best is None or rr.fun < best.fun else best
    pr = np.abs(qaoa_state(best.x, sp, E)) ** 2; o = int(np.argmin(E))

    actual = ((G.ign_date.values[gi] > np.datetime64(D.T0)) &
              (G.ign_date.values[gi] <= np.datetime64(D.T0 + pd.Timedelta(days=HORIZON)))).astype(float)
    burned_val = sp.Z @ (actual * D.value[C])
    burned_cnt = sp.Z @ actual
    rank = (burned_val > burned_val[o]).sum() + 1
    return dict(n_edges=len(ed), P_opt=float(pr[o]), amp=float(pr[o] * sp.dim), P_unif=1 / sp.dim,
                qcdf_cells=[str(G.h3.iloc[x]) for x in gi[sp.Z[o] == 1]],
                qcdf_burned_cells=int(burned_cnt[o]), qcdf_burned_housing=float(burned_val[o]),
                mean_burned_cells=float(burned_cnt.mean()), mean_burned_housing=float(burned_val.mean()),
                rank_burned_housing=int(rank), n_subsets=int(sp.dim),
                frac_cand_burned=float(actual.mean()), v=v.tolist(), c=c.tolist(), r=r.tolist(), u=u.tolist())

def sequential(D, rng):
    C = D.cand; m = len(C)
    P = list(itertools.product(range(m), repeat=T)); dim = len(P)
    X = np.zeros((dim, T * m), np.int8)
    for a, pl in enumerate(P):
        for t, c in enumerate(pl): X[a, t * m + c] = 1
    fit = rng.choice(dim, M_FIT, replace=False)
    y = np.array([plan_loss(D, P[a], D.UA) for a in fit])
    iu = np.triu_indices(T * m, 1)
    feat = lambda Z: np.hstack([np.ones((len(Z), 1)), Z, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])
    coef, *_ = np.linalg.lstsq(feat(X[fit].astype(float)), y, rcond=None)
    Esur = np.concatenate([feat(X[i:i + 5000].astype(float)) @ coef for i in range(0, dim, 5000)])
    hold = rng.choice(np.setdiff1d(np.arange(dim), fit), 300, replace=False)
    yh = np.array([plan_loss(D, P[a], D.UA) for a in hold])
    r2 = 1 - np.var(yh - Esur[hold]) / np.var(yh)

    ring = [(i, (i + 1) % m) for i in range(m)]
    sys.path.insert(0, '../runs'); import seq as SQ
    PD = SQ.ProductDicke(m, 1, T, ring)
    order = {tuple(PD.X[a].reshape(T, m).argmax(1)): a for a in range(PD.dim)}
    perm = np.array([order[pl] for pl in P])
    Epd = np.empty(PD.dim); Epd[perm] = Esur
    class _S: pass
    sp = _S(); sp.pairs = PD.full_pairs; sp.dim = PD.dim
    En = (Epd - Epd.mean()) / Epd.std()
    pra = A.anneal(sp, En, np.full(len(PD.full_pairs), 1e9), 40.0, dt=0.1)
    a_ann = tuple(PD.X[int(np.argmax(pra))].reshape(T, m).argmax(1))

    myo = []
    for t in range(T):
        best = None
        for c in range(m):
            pl = myo + [c]
            prot = [np.array([C[x]]) for x in pl]
            R = D.UA.shape[0]; burned = np.repeat(D.burned0[None], R, 0)
            for tt in range(t + 2):
                q = D.q[tt].copy()
                if tt < len(prot): q[np.isin(D.dst, prot[tt])] *= (1 - ETA)
                fire = burned[:, D.src] & (D.UA[:, tt] < q[None]); burned |= D._ign(fire)
            val = ((burned & ~D.burned0[None]) * D.value).sum(1).mean()
            if best is None or val < best[0]: best = (val, c)
        myo.append(best[1])
    plans = dict(static_ranking=tuple(range(T)), myopic=tuple(myo), seq_surrogate=P[int(np.argmin(Esur))],
                 seq_anneal=a_ann, best_sampled=P[fit[int(np.argmin(y))]])
    rnd = [P[a] for a in rng.choice(dim, 20, replace=False)]
    out = {k: plan_loss(D, pl, D.UB) for k, pl in plans.items()}
    out['none'] = D.run([], D.UB)[0]; out['random'] = float(np.mean([plan_loss(D, pl, D.UB) for pl in rnd]))
    return dict(loss=out, r2_holdout=float(r2), p_anneal_sur_opt=float(pra[perm[int(np.argmin(Esur))]]),
                plans={k: list(map(int, v)) for k, v in plans.items()})

if __name__ == '__main__':
    dates = [pd.Timestamp(d) for d in sys.argv[2].split(',')]
    out = []
    for T0 in dates:
        t0 = time.time(); rng = np.random.default_rng(int(T0.strftime('%m%d')))
        D = Domain(T0, rng)
        _, pb = D.run([], D.UA)
        var = D.value * pb; var[D.burned0] = -1
        D.cand_all = np.argsort(-var); D.cand = D.cand_all[:N_CAND]
        st = solve_static_qcdf(D, rng)
        sq = sequential(D, rng)
        rec = dict(T0=str(T0.date()), n_domain=D.n, housing_at_risk=float((D.value * pb).sum()), static=st, seq=sq)
        out.append(rec)
        L = sq['loss']; b = L['none']
        print(T0.date(), f"domain={D.n} R2={sq['r2_holdout']:.3f} static: QCDF burned {st['qcdf_burned_cells']}/3 cells "
              f"(mean {st['mean_burned_cells']:.2f}), rank {st['rank_burned_housing']}/{st['n_subsets']} P_opt={st['P_opt']:.3f} | seq saved: "
              + " ".join(f"{k}={100*(b-v)/max(b,1e-9):.1f}" for k, v in L.items() if k != 'none') + f" [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(sys.argv[1], 'w'), indent=1)
