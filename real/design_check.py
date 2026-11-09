"""Why are the hindcast savings only 3-4%? Same domains/model as realcase.py; vary the intervention design."""
import sys, numpy as np, pandas as pd
sys.argv = ['x']; import realcase as RC
H = RC.HORIZON
def greedy(D, C, per_day, days, persistent):
    chosen, prot = [], []
    for t in range(days):
        today = []
        for _ in range(per_day):
            best = None
            for c in C:
                if c in chosen or c in today: continue
                base = (chosen + today + [c]) if persistent else (today + [c])
                pl = [np.array(x) for x in prot] + [np.array(base)]
                if persistent: pl += [np.array(base)] * (H - len(pl))
                v = D.run(pl, D.UA[:150])[0]
                if best is None or v < best[0]: best = (v, c)
            today.append(best[1])
        if persistent: chosen += today; prot.append(list(chosen))
        else: prot.append(today)
    pl = [np.array(x) for x in prot]
    if persistent: pl += [np.array(chosen)] * (H - len(pl))
    return pl
rows = []
for d in sys.argv[1:] or ['2021-07-24', '2021-07-27', '2021-07-31', '2021-08-04']:
    pass
for d in ['2021-07-24', '2021-07-27', '2021-07-31', '2021-08-04']:
    T0 = pd.Timestamp(d); rng = np.random.default_rng(int(T0.strftime('%m%d')))
    D = RC.Domain(T0, rng); _, pb = D.run([], D.UA)
    var = D.value * pb; var[D.burned0] = -1; C = list(np.argsort(-var)[:20])
    none = D.run([], D.UB)[0]; sv = lambda pl: round(100 * (none - D.run(pl, D.UB)[0]) / none, 1)
    r = dict(date=d, housing_at_risk=round(none))
    r['A_1cell_1day_x4 (paper)'] = sv(greedy(D, C, 1, 4, False))
    r['B_1cell_persistent_x4'] = sv(greedy(D, C, 1, 4, True))
    r['C_3cells_persistent_x4'] = sv(greedy(D, C, 3, 4, True))
    r['D_all20_whole_horizon'] = sv([np.array(C)] * H)
    r['front_cells'] = int(sum(1 for j in range(D.n) if not D.burned0[j] and any(D.burned0[D.src[e]] for e in np.where(D.dst == j)[0])))
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv('design_check.csv', index=False)
