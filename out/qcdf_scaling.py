"""qcdf_scaling.py — Scaling study + classical baselines for the QCDF paper =========================================================================="""
import argparse
import itertools
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

W_C, W_R, W_U, W_V = 1.0, 0.5, 0.3, 2.0
J_DEFAULT = 0.3

HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

def hex_patch(n):
    cells = [(q, r) for q in range(-6, 7) for r in range(-6, 7)]
    def dist(c):
        q, r = c
        return (abs(q) + abs(r) + abs(q + r)) / 2 + 1e-3 * (q * 7 + r)
    cells = sorted(cells, key=dist)[:n]
    idx = {c: i for i, c in enumerate(cells)}
    edges = set()
    for (q, r), i in idx.items():
        for dq, dr in HEX_DIRS:
            j = idx.get((q + dq, r + dr))
            if j is not None and i < j:
                edges.add((i, j))
    return sorted(edges)

def random_instance(n, rng):
    v, c, r, u = rng.random((4, n))
    h = W_C * c + W_R * r + W_U * u - W_V * v
    return h, hex_patch(n)

class DickeSpace:

    def __init__(self, n, k, edges):
        self.n, self.k, self.edges = n, k, edges
        combos = list(itertools.combinations(range(n), k))
        self.Z = np.zeros((len(combos), n), dtype=np.int8)
        for a, cmb in enumerate(combos):
            self.Z[a, list(cmb)] = 1
        self.dim = len(combos)
        key = {tuple(row): a for a, row in enumerate(self.Z)}

        self.pairs = []
        for (i, j) in edges:
            a01 = np.where((self.Z[:, i] == 0) & (self.Z[:, j] == 1))[0]
            a10 = np.empty_like(a01)
            for t, a in enumerate(a01):
                row = self.Z[a].copy()
                row[i], row[j] = 1, 0
                a10[t] = key[tuple(row)]
            self.pairs.append((a01, a10))

    def energies(self, h, J):
        s = 1 - 2 * self.Z.astype(float)
        e = self.Z @ h
        for (i, j) in self.edges:
            e = e + J * s[:, i] * s[:, j]
        return e

def qaoa_state(params, space, E):
    p = len(params) // 2
    gam, bet = params[:p], params[p:]
    psi = np.full(space.dim, 1 / math.sqrt(space.dim), dtype=complex)
    for g, b in zip(gam, bet):
        psi = psi * np.exp(-1j * g * E)
        cb, sb = math.cos(b), math.sin(b)
        for a01, a10 in space.pairs:
            x01, x10 = psi[a01], psi[a10]

            psi[a01] = cb * x01 - 1j * sb * x10
            psi[a10] = cb * x10 - 1j * sb * x01
    return psi

def qaoa_energy(params, space, E):
    psi = qaoa_state(params, space, E)
    return float(np.real(np.vdot(psi, E * psi)))

def optimise_qaoa(space, E, p_max, restarts, rng):
    bounds_g, bounds_b = (0, 2 * np.pi), (0, np.pi)
    results, prev = {}, None
    for p in range(1, p_max + 1):
        starts = []
        if prev is not None:
            q = p - 1
            def interp(x):
                xo = np.concatenate([[0.0], x, [0.0]])
                return np.array([((i - 1) / q) * xo[i - 1] + ((q - i + 1) / q) * xo[i]
                                 for i in range(1, p + 1)])
            starts.append(np.concatenate([interp(prev[:q]), interp(prev[q:])]))
        for _ in range(restarts):
            starts.append(np.concatenate([rng.uniform(*bounds_g, p), rng.uniform(*bounds_b, p)]))
        best = None
        for x0 in starts:
            res = minimize(qaoa_energy, x0, args=(space, E), method="L-BFGS-B",
                           bounds=[bounds_g] * p + [bounds_b] * p,
                           options={"maxiter": 500})
            if best is None or res.fun < best.fun:
                best = res
        prev = best.x
        results[p] = best
    return results

def energy_of(z, h, edges, J):
    s = 1 - 2 * z
    return float(z @ h + J * sum(s[i] * s[j] for i, j in edges))

def greedy(h, edges, J, k):
    n = len(h)
    z = np.zeros(n, dtype=int)
    for _ in range(k):
        best_i, best_e = None, None
        for i in np.where(z == 0)[0]:
            z[i] = 1
            e = energy_of(z, h, edges, J)
            z[i] = 0
            if best_e is None or e < best_e:
                best_i, best_e = i, e
        z[best_i] = 1
    return energy_of(z, h, edges, J)

def simulated_annealing(h, edges, J, k, rng, sweeps=200, T0=2.0, T1=0.01):
    n = len(h)
    z = np.zeros(n, dtype=int)
    z[rng.choice(n, k, replace=False)] = 1
    e = energy_of(z, h, edges, J)
    best = e
    steps = sweeps * n
    for t in range(steps):
        T = T0 * (T1 / T0) ** (t / steps)
        i = rng.choice(np.where(z == 1)[0]); j = rng.choice(np.where(z == 0)[0])
        z[i], z[j] = 0, 1
        e2 = energy_of(z, h, edges, J)
        if e2 <= e or rng.random() < math.exp(-(e2 - e) / T):
            e = e2; best = min(best, e)
        else:
            z[i], z[j] = 1, 0
    return best

def run_instance(n, h, edges, J, p_max, restarts, rng, label, inst):
    k = n // 2
    space = DickeSpace(n, k, edges)
    E = space.energies(h, J)
    e_min, e_max = E.min(), E.max()
    opt = np.isclose(E, e_min, atol=1e-9)
    t0 = time.time()
    qres = optimise_qaoa(space, E, p_max, restarts, rng)
    t_q = time.time() - t0
    e_gr = greedy(h, edges, J, k)
    e_sa = simulated_annealing(h, edges, J, k, rng)
    rows = []
    for p, res in qres.items():
        psi = qaoa_state(res.x, space, E)
        prob = np.abs(psi) ** 2
        p_opt = float(prob[opt].sum())
        rows.append(dict(
            label=label, instance=inst, N=n, k=k, edges=len(edges), dim=space.dim, p=p,
            E_star=e_min, E_qaoa=res.fun,
            r_qaoa=(e_max - res.fun) / (e_max - e_min),
            P_opt=p_opt, P_uniform=opt.sum() / space.dim,
            amplification=p_opt / (opt.sum() / space.dim),
            P_opt_in_100_shots=1 - (1 - p_opt) ** 100,
            r_greedy=(e_max - e_gr) / (e_max - e_min), greedy_opt=bool(np.isclose(e_gr, e_min)),
            r_sa=(e_max - e_sa) / (e_max - e_min), sa_opt=bool(np.isclose(e_sa, e_min)),
            P_feasible=1.0, t_qaoa_s=t_q))
    return rows

def make_figure(df, out):
    import matplotlib.pyplot as plt
    syn = df[df.label == "synthetic"]
    pmax = syn.p.max()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for p in sorted(syn.p.unique()):
        g = syn[syn.p == p].groupby("N")
        ax[0].errorbar(g.r_qaoa.mean().index, g.r_qaoa.mean(), yerr=g.r_qaoa.std(),
                       marker="o", capsize=3, label=f"QAOA p={p}")
        ax[1].errorbar(g.amplification.mean().index, g.amplification.mean(),
                       yerr=g.amplification.std(), marker="o", capsize=3, label=f"p={p}")
    g = syn[syn.p == pmax].groupby("N")
    ax[0].plot(g.r_greedy.mean().index, g.r_greedy.mean(), "k--s", label="greedy")
    ax[0].plot(g.r_sa.mean().index, g.r_sa.mean(), "k:^", label="sim. annealing")
    ax[0].set(xlabel="N (cells)", ylabel="approximation ratio", title="(a) solution quality")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="N (cells)", ylabel=r"$P(\mathrm{opt})\,/\,P_{\mathrm{uniform}}$",
              yscale="log", title="(b) amplification over uniform-feasible")
    ax[1].legend(fontsize=7)
    ax[2].semilogy(g.P_opt.mean().index, g.P_opt.mean(), "o-", label=f"QAOA p={pmax}")
    ax[2].semilogy(g.P_uniform.mean().index, g.P_uniform.mean(), "k:", label=r"uniform $1/\binom{N}{k}$")
    ax[2].set(xlabel="N (cells)", ylabel=r"$P(\mathrm{opt})$", title="(c) optimum probability")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
    print(f"saved {out}.pdf/.png")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ns", default="6,8,10,12,14,16")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--pmax", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--J", type=float, default=J_DEFAULT)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pilot", help="CSV with columns v,c,r,u (one row per cell)")
    ap.add_argument("--edges", help="CSV with columns i,j (0-based row indices of --pilot)")
    ap.add_argument("--quick", action="store_true", help="smoke test")
    ap.add_argument("--out", default="scaling_results")
    a = ap.parse_args()
    if a.quick:
        a.Ns, a.instances, a.pmax, a.restarts = "6,8", 2, 2, 3
    rng = np.random.default_rng(a.seed)
    rows = []
    if a.pilot:
        cells = pd.read_csv(a.pilot)
        h = (W_C * cells.c + W_R * cells.r + W_U * cells.u - W_V * cells.v).to_numpy()
        ed = [tuple(sorted(map(int, x))) for x in pd.read_csv(a.edges)[["i", "j"]].to_numpy()]
        print(f"pilot: N={len(h)}, |E|={len(ed)}")
        rows += run_instance(len(h), h, ed, a.J, a.pmax, max(a.restarts, 30), rng, "pilot", 0)
    for n in [int(x) for x in a.Ns.split(",")]:
        for inst in range(a.instances):
            h, ed = random_instance(n, rng)
            r = run_instance(n, h, ed, a.J, a.pmax, a.restarts, rng, "synthetic", inst)
            rows += r
            last = r[-1]
            print(f"N={n:2d} inst={inst} |E|={last['edges']:2d} dim={last['dim']:6d}  "
                  f"p={last['p']} r={last['r_qaoa']:.3f} P(opt)={last['P_opt']:.3f} "
                  f"({last['amplification']:.1f}x)  greedy_opt={last['greedy_opt']} "
                  f"SA_opt={last['sa_opt']}  t={last['t_qaoa_s']:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out + ".csv", index=False)
    print(f"saved {a.out}.csv")
    summ = (df[df.label == "synthetic"].groupby(["N", "p"])
            [["r_qaoa", "P_opt", "amplification", "r_greedy", "greedy_opt", "r_sa", "sa_opt"]]
            .mean().round(3))
    print(summ.to_string())
    summ.to_csv(a.out + "_summary.csv")
    make_figure(df, "fig_scaling")

if __name__ == "__main__":
    main()
