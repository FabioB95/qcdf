"""Full-scale astronomical QCDF: follow-up campaign of Fermi-LAT unidentified sources."""
import sys, json, math, numpy as np, healpy as hp
from astropy.io import fits
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA = '../realdata/fermi_cat/'
NSIDE, FWHM, DEC_MAX, NMC = 64, 1.0, 44.0, 400
UNID = ('', 'unk', 'spp')

def load(f):
    d = fits.open(DATA + f)[1].data
    nm = np.char.strip(d['Source_Name'].astype(str)); cls = np.char.strip(np.char.lower(d['CLASS1'].astype(str)))
    return d, nm, cls

FEAT = ['logF', 'idx', 'curv', 'logvar', 'absb', 'logerr', 'logsig']
def features(d):
    curv = np.nan_to_num(d['LP_SigCurv'].astype(float), nan=0.0)
    return np.c_[np.log10(d['Flux1000'].astype(float)), d['PL_Index'].astype(float), np.clip(curv, 0, 30),
                 np.log10(np.clip(d['Variability_Index'].astype(float), 1, None)), np.abs(d['GLAT'].astype(float)),
                 np.log10(d['Conf_95_SemiMajor'].astype(float)), np.log10(d['Signif_Avg'].astype(float))]

def transition(fa, fb):
    da, na, ca = load(fa); db, nb, cb = load(fb); mb = dict(zip(nb, cb))
    tgt = np.isin(ca, UNID) & np.isfinite(da['Conf_95_SemiMajor'].astype(float))
    y = np.array([(n in mb) and (mb[n] not in UNID) for n in na])
    return da[tgt], na[tgt], features(da[tgt]), y[tgt]

def beam_overlaps(d, rng):
    ra, de = np.radians(d['RAJ2000'].astype(float)), np.radians(d['DEJ2000'].astype(float))
    a95, b95, pa = d['Conf_95_SemiMajor'].astype(float), d['Conf_95_SemiMinor'].astype(float), np.radians(d['Conf_95_PosAng'].astype(float))
    k = math.sqrt(-2 * math.log(0.05)); sig = 4 * math.log(2) / FWHM ** 2
    out = []
    for s in range(len(d)):

        u = rng.standard_normal((NMC, 2)) * np.array([a95[s] / k, b95[s] / k])
        dx = u[:, 0] * math.sin(pa[s]) + u[:, 1] * math.cos(pa[s]); dy = u[:, 0] * math.cos(pa[s]) - u[:, 1] * math.sin(pa[s])
        dec_s = de[s] + np.radians(dy); ra_s = ra[s] + np.radians(dx) / max(math.cos(de[s]), 1e-3)
        vec = hp.ang2vec(np.pi / 2 - de[s], ra[s])
        pix = hp.query_disc(NSIDE, vec, np.radians(2.0))
        th, ph = hp.pix2ang(NSIDE, pix)

        cosd = (np.sin(dec_s)[:, None] * np.cos(th)[None] + np.cos(dec_s)[:, None] * np.sin(th)[None] * np.cos(ra_s[:, None] - ph[None]))
        dist = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
        f = np.exp(-sig * dist ** 2).mean(0)
        out.append({int(p): float(v) for p, v in zip(pix, f) if v > 0.01})
    return out

class Sky:
    def __init__(s, d, P, rng):
        s.P = P; s.f = beam_overlaps(d, rng)
        th, ph = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
        vis = (90 - np.degrees(th)) < DEC_MAX
        cells = sorted({p for fs in s.f for p in fs if vis[p]}); s.cells = np.array(cells); s.loc = {p: k for k, p in enumerate(cells)}
        n = len(cells); s.n = n
        s.h = np.zeros(n); J = {}
        for Ps, fs in zip(P, s.f):
            ps = [(s.loc[p], v) for p, v in fs.items() if p in s.loc]
            for a, va in ps: s.h[a] -= Ps * va
            for x in range(len(ps)):
                for y in range(x + 1, len(ps)):
                    a, b = sorted((ps[x][0], ps[y][0])); J[(a, b)] = J.get((a, b), 0.0) + Ps * ps[x][1] * ps[y][1]
        s.J = J
        nbr = {}
        for k, p in enumerate(cells):
            nbr[k] = [s.loc[q] for q in hp.get_all_neighbours(NSIDE, p) if q >= 0 and q in s.loc]
        s.adj = nbr

        s.rows = [np.array([s.loc[p] for p in fs if p in s.loc], int) for fs in s.f]
        s.vals = [np.array([v for p, v in fs.items() if p in s.loc]) for fs in s.f]
    def F(s, Z, w=None):
        w = s.P if w is None else w; Zs = np.zeros(s.n, bool); Zs[list(Z)] = True
        return float(sum(w_ * (1 - np.prod(1 - v[Zs[r]])) for w_, r, v in zip(w, s.rows, s.vals) if len(r)))
    def energy(s, Z):
        Zs = set(Z); return float(s.h[list(Z)].sum() + sum(v for (i, j), v in s.J.items() if i in Zs and j in Zs))
    def milp(s, K, use_J=True):
        n = s.n; J = s.J if use_J else {}; Pl = [p for p, v in J.items() if v > 1e-12]; m = len(Pl)
        c = np.concatenate([s.h, np.array([J[p] for p in Pl])])
        rows, lb, ub = [], [], []
        from scipy.sparse import lil_matrix
        A = lil_matrix((1 + m, n + m)); A[0, :n] = 1; lb.append(K); ub.append(K)
        for k, (i, j) in enumerate(Pl):
            A[1 + k, n + k] = 1; A[1 + k, i] = -1; A[1 + k, j] = -1; lb.append(-1); ub.append(np.inf)
        res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.r_[np.ones(n), np.zeros(m)],
                   bounds=Bounds(0, 1), options=dict(time_limit=300))
        return [int(k) for k in np.where(np.round(res.x[:n]) == 1)[0]], res
    def greedy(s, K):
        import heapq
        Z = []; cur = 0.0; H = [(-(-s.h[i]), i, 0) for i in range(s.n)]; heapq.heapify(H)
        while len(Z) < K:
            g, i, it = heapq.heappop(H)
            if it == len(Z): Z.append(i); cur = s.F(Z); continue
            heapq.heappush(H, (-(s.F(Z + [i]) - cur), i, len(Z)))
        return Z
