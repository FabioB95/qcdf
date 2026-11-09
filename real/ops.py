"""Operational realism checks on the Dixie hindcast (prospective model, 10 active dates, K=10)."""
import sys, json, math, time, numpy as np, pandas as pd, geopandas as gpd, h3
_ARGV = list(sys.argv); sys.argv = ['x']
import fullscale as FS
sys.argv = _ARGV
from fastsim import _sim
RC = FS.RC; G = FS.G
K = 10; W_CRIT = 500.0

class Pv(FS.Problem):
    eta_mat = None
    def L(s, S, R, stream):
        S = set(S) | s.S_prev
        if s.eta_mat is None: return s.sim.losses(S if S else [], R, stream, eta=s.eta)[0]
        P = np.zeros((s.sim.H, s.n)); idx = np.fromiter(S, np.int64) if S else np.zeros(0, np.int64)
        P[:, idx] = s.eta_mat[:, idx]; Lr = np.zeros(R); F = np.zeros(s.n, np.int64)
        _sim(s.sim.src, s.sim.dst, s.sim.q, s.sim.b0, s.sim.v, P, 0.0, R, stream, Lr, F); return Lr
    def freq(s, S, R, stream):
        S = set(S); P = np.zeros((s.sim.H, s.n)); idx = np.fromiter(S, np.int64) if S else np.zeros(0, np.int64)
        P[:, idx] = 0.85 if s.eta_mat is None else s.eta_mat[:, idx]; Lr = np.zeros(R); F = np.zeros(s.n, np.int64)
        _sim(s.sim.src, s.sim.dst, s.sim.q, s.sim.b0, s.sim.v, P, 0.0, R, stream, Lr, F); return F / R

def eta_cond(P):
    days = [RC.mday[P.T0 + pd.Timedelta(days=t)] for t in range(P.sim.H)]
    v = np.stack([RC.met['vs'][d, P.D.cells] for d in days])
    sl = np.zeros(P.n)
    for k, g in enumerate(P.D.cells):
        sl[k] = max([abs(RC.elev[j] - RC.elev[g]) / 2400 for j in FS.NB[g]] + [0])
    return 0.85 / (1 + (v / 6.0) ** 2) * (1 - 0.5 * np.minimum(1, sl / 0.3))[None]

def plans_for(P):
    P.build_qubo(); out = {}
    out['QCDF_MILP'], _ = P.milp(K); out['QCDF_h_only'], _ = P.milp(K, use_J=False)
    bl = P.baselines(K); out['B4_value_at_risk'] = bl['B4_value_at_risk']; out['B1_communities'] = bl['B1_communities']; out['B2_nearest_front'] = bl['B2_nearest_front']
    return out

def evaluate(P, plans):
    base = P.L(set(), FS.R_B, 99).mean()
    return {k: float(100 * (base - P.L(set(S), FS.R_B, 99).mean()) / base) for k, S in plans.items()}, float(base)

hc = pd.read_csv('../realdata/hcai/lhfl_2021_06_30.csv', encoding='latin1', low_memory=False)
hc = hc[hc.LATITUDE.notna() & hc.LICENSE_CATEGORY_DESC.isin(['General Acute Care Hospital', 'Skilled Nursing Facility']) & (hc.FACILITY_STATUS_DESC == 'Open')]
hc['h3'] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(hc.LATITUDE, hc.LONGITUDE)]
hidx = {c: k for k, c in enumerate(G.h3)}; crit = np.zeros(len(G))
for c, nm in zip(hc.h3, hc.FACILITY_NAME):
    if c in hidx: crit[hidx[c]] += 1
places = gpd.read_file('zip://../realdata/tl_2021_06_place.zip', bbox=tuple(G.total_bounds)).to_crs(4326)
cent = gpd.GeoDataFrame(geometry=gpd.points_from_xy(G.lon, G.lat), crs=4326)
pj = gpd.sjoin_nearest(cent.to_crs(3310), places.to_crs(3310)[['NAME', 'geometry']], max_distance=6000, how='left')
pj = pj[~pj.index.duplicated()]
place_of = pj['NAME'].fillna('').values

def attribution(P, plan, R=2000):
    f0 = P.freq(set(), R, 99); rows = []
    for i in plan:
        dp = f0 - P.freq({i}, R, 99); gain = dp * P.D.value
        by = pd.Series(gain).groupby(place_of[P.D.cells]).sum().drop('', errors='ignore').sort_values(ascending=False)
        rows.append(dict(cell=str(G.h3.iloc[P.D.cells[i]]), place=place_of[P.D.cells[i]], housing=float(P.D.value[i]), p_burn=float(f0[i]),
                         gain=float(gain.sum()), own=float(gain[i]), top=[(k, float(v)) for k, v in by.head(3).items() if v > 0.05]))
    return rows

def run(T0):
    t0 = time.time(); r = dict(T0=T0)
    P = Pv(T0); pl = plans_for(P); r['base'], r['none'] = evaluate(P, pl)

    Pe = Pv(T0); Pe.eta_mat = eta_cond(Pe); r['eta_cond_mean'] = float(Pe.eta_mat.mean())
    ple = plans_for(Pe); r['eta_cond'], _ = evaluate(Pe, ple)
    r['eta_cond_constplans'], _ = evaluate(Pe, pl)
    r['eta_overlap'] = len(set(ple['QCDF_MILP']) & set(pl['QCDF_MILP']))

    Pr = Pv(T0); road = G.road_km.values[Pr.D.cells]; Pr.elig = Pr.elig[road[Pr.elig] <= 3.0]
    r['road_frac_elig'] = float(len(Pr.elig) / len(P.elig))
    plr = plans_for(Pr); r['road'], _ = evaluate(Pr, plr)

    Pc = Pv(T0); cv = crit[Pc.D.cells]; Pc.D.value = Pc.D.value + W_CRIT * cv
    from fastsim import Sim; Pc.sim = Sim(Pc.D, 0.85)
    plc = plans_for(Pc); f0 = Pc.freq(set(), 2000, 99)
    ci = np.where(cv > 0)[0]
    r['critical'] = dict(n_fac_cells=int(len(ci)), p_burn=[float(f0[i]) for i in ci], names=[str(G.h3.iloc[Pc.D.cells[i]]) for i in ci],
                         p_after={k: [float(x) for x in Pc.freq(set(S), 2000, 99)[ci]] for k, S in plc.items() if k in ('QCDF_MILP', 'B4_value_at_risk')},
                         housing_saved_with=evaluate(P, {'c': [i for i in plc['QCDF_MILP']]})[0]['c'], housing_saved_without=r['base']['QCDF_MILP'],
                         overlap=len(set(plc['QCDF_MILP']) & set(pl['QCDF_MILP'])))
    if T0 in ('2021-07-31', '2021-08-04'):
        r['attribution'] = attribution(P, pl['QCDF_MILP'])
        gv = [k for k in P.elig if place_of[P.D.cells[k]] == 'Greenville' and P.D.value[k] == P.D.value[P.elig][[place_of[P.D.cells[x]] == 'Greenville' for x in P.elig]].max()] if any(place_of[P.D.cells[x]] == 'Greenville' for x in P.elig) else []
        if gv:
            g = gv[0]; f0 = P.freq(set(), 2000, 99); order = P.elig[np.argsort(-P.delta[P.elig])]
            r['greenville'] = dict(cell=str(G.h3.iloc[P.D.cells[g]]), housing=float(P.D.value[g]), p_burn=float(f0[g]), delta=float(P.delta[g]),
                                   delta_rank=int(np.where(order == g)[0][0] + 1), n_elig=int(len(P.elig)), dist=int(P.dist[g]),
                                   var_rank=int(np.where(P.elig[np.argsort(-(P.D.value * f0)[P.elig])] == g)[0][0] + 1),
                                   in_plan={k: bool(g in S) for k, S in pl.items()}, observed_burn_10d=bool(P.obs_burn[g]))
    print(T0, {k: (round(v['QCDF_MILP'], 1) if isinstance(v, dict) and 'QCDF_MILP' in v else '') for k, v in r.items()}, f'[{time.time()-t0:.0f}s]', flush=True)
    return r

if __name__ == '__main__':
    dates = sys.argv[1].split(','); outf = sys.argv[2]; res = []
    for d in dates:
        res.append(run(d)); json.dump(res, open(outf, 'w'), default=float)
