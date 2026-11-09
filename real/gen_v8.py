"""v8 tokens: leakage-free models (prospective primary, rolling secondary, oracle upper bound), calibration of probabilities, dependence-aware inference (circular block bootstrap over decision dates), Caldor replication, aridity test. Fills @@XV_*@@ in ../paper/v4_body_filled.tex (run after gen_full.py)."""
import json, glob, re, numpy as np, pandas as pd, h3
tok = {}; OUT = '../paper/figures/tikz/'
SPLIT = '2021-08-05'
def load(*fs):
    R = []
    for f in fs:
        for g in sorted(glob.glob(f)): R += json.load(open(g))
    return sorted(R, key=lambda r: r['T0'])
ORA = load('fs_[ab].json', 'fs_late_[ab].json'); PRO = load('fsP_act.json', 'fsP_late.json'); ROL = load('fsR.json'); CAL = load('fsC.json')
MOD = {'prospective': PRO, 'rolling': ROL, 'oracle': ORA}
def S(rs, K, m): return np.array([r['K'][K]['eval'][m]['saved_pct'] for r in rs if K in r['K']])
def cbb(d, l=6, B=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(d); nb = int(np.ceil(n / l)); out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n, nb); idx = np.concatenate([(s + np.arange(l)) % n for s in st])[:n]; out[b] = d[idx].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
METH = [('B6_random', 'random'), ('B1_communities', 'communities first'), ('B2_nearest_front', 'nearest to front'),
        ('B3_wind_downstream', 'wind-downstream'), ('B4_value_at_risk', 'value at risk $H_iP_i$'), ('QCDF_h_only', 'QCDF, $J=0$'),
        ('QCDF_SA', 'QCDF, SA in $\\feasible_K$'), ('QCDF_MILP', 'QCDF, exact'), ('CELF_greedy', 'greedy on simulator')]
G = pd.read_csv('dins_cells.csv'); dins = dict(zip(G.h3, G.dins_res_destroyed))
import geopandas as gpd
GR = gpd.read_file('grid_r7.gpkg'); GR['ign_date'] = pd.to_datetime(GR['ign_date']); ign = dict(zip(GR.h3, GR.ign_date))
def dcov(rs, K, m, dins, ign):
    num = den = 0
    for r in rs:
        T0 = pd.Timestamp(r['T0']); hit = lambda c: (c in ign) and pd.notna(ign[c]) and T0 < ign[c] <= T0 + pd.Timedelta(days=10)
        num += sum(dins.get(c, 0) for c in r['K'][K]['eval'][m]['plan'] if hit(c))
        den += sum(v for c, v in dins.items() if hit(c))
    return 100 * num / max(den, 1)

act = [r for r in PRO if r['T0'] <= SPLIT]; late = [r for r in PRO if r['T0'] > SPLIT]
rows = []
for m, lab in METH:
    cells = [f"${S(act, K, m).mean():.1f}\\pm{S(act, K, m).std(ddof=1)/np.sqrt(len(act)):.1f}$" for K in ('5', '10', '20')]
    rows.append(f"{lab} & " + " & ".join(cells) + f" & ${dcov(act, '10', m, dins, ign):.0f}$ & ${S(late, '10', m).mean():.1f}$ & ${S([r for r in ORA if r['T0'] <= SPLIT], '10', m).mean():.1f}$ & ${S([r for r in ROL if r['T0'] <= SPLIT], '10', m).mean():.1f}$ \\\\")
tok['XV_MAINROWS'] = "\n".join(rows); tok['XV_nact'] = str(len(act)); tok['XV_nlate'] = str(len(late))
for K in ('5', '10', '20'):
    for m, _ in METH: tok[f'XV_s_{m}_{K}'] = f"{S(act, K, m).mean():.1f}"
tok['XV_dcov_comm'] = f"{dcov(act, '10', 'B1_communities', dins, ign):.0f}"; tok['XV_dcov_qcdf'] = f"{dcov(act, '10', 'QCDF_MILP', dins, ign):.0f}"
tok['XV_dcov_var'] = f"{dcov(act, '10', 'B4_value_at_risk', dins, ign):.0f}"

rows = []
for a, b, lab in [('QCDF_MILP', 'B4_value_at_risk', 'value at risk'), ('QCDF_MILP', 'QCDF_h_only', 'QCDF, $J=0$'), ('QCDF_MILP', 'CELF_greedy', 'greedy on simulator'),
                  ('QCDF_MILP', 'B2_nearest_front', 'nearest to front'), ('QCDF_MILP', 'B1_communities', 'communities first')]:
    cells = []
    for mname in ('prospective', 'rolling', 'oracle'):
        R = MOD[mname]; da = S([r for r in R if r['T0'] <= SPLIT], '10', a) - S([r for r in R if r['T0'] <= SPLIT], '10', b)
        dall = S(R, '10', a) - S(R, '10', b); lo, hi = cbb(dall)
        cells.append(f"${da.mean():+.1f}$ & ${dall.mean():+.1f}$ [${lo:+.1f}$, ${hi:+.1f}$]")
        tok[f'XV_d_{b}_{mname}'] = f"{da.mean():+.1f}"; tok[f'XV_dall_{b}_{mname}'] = f"{dall.mean():+.1f}"; tok[f'XV_ci_{b}_{mname}'] = f"[{lo:+.1f}, {hi:+.1f}]"
        tok[f'XV_w_{b}_{mname}'] = f"{(da > 0.05).sum()}/{(abs(da) <= 0.05).sum()}/{(da < -0.05).sum()}"
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
tok['XV_ROBROWS'] = "\n".join(rows); tok['XV_nall'] = str(len(PRO))

C = json.load(open('calib_check.json')); rows = []
for s in C['summary']:
    if s['phase'] == 'all': continue
    rows.append(f"{s['model']} & {s['phase']} & ${s['auc_median']:.2f}$ & ${s['brier']:.3f}$ & ${s['oe_cells']:.2f}$ & ${s['oe_H']:.2f}$ & ${s['cal_slope']:.2f}$ & ${100*s['cover90']:.0f}$ \\\\")
    tok[f"XV_cal_{s['model']}_{s['phase']}_oeH"] = f"{s['oe_H']:.2f}"; tok[f"XV_cal_{s['model']}_{s['phase']}_oeC"] = f"{s['oe_cells']:.2f}"
    tok[f"XV_cal_{s['model']}_{s['phase']}_auc"] = f"{s['auc_median']:.2f}"; tok[f"XV_cal_{s['model']}_{s['phase']}_cov"] = f"{100*s['cover90']:.0f}"
    tok[f"XV_cal_{s['model']}_{s['phase']}_slope"] = f"{s['cal_slope']:.2f}"
tok['XV_CALROWS'] = "\n".join(rows)

rel = pd.DataFrame(C['reliability']); pl = []
for mname, col, mk in (('prospective', 'qsel', '*'), ('rolling', 'qblue', 'square*'), ('oracle', 'gray', 'triangle*')):
    for ph, ls in (('active', 'solid'), ('late', 'dashed')):
        x = rel[(rel.model == mname) & (rel.phase == ph) & (rel.n >= 30)]
        pl.append(rf"\addplot[{col},{ls},thick,mark={mk},mark size=1.3pt] coordinates {{" + " ".join(f"({r.p:.4f},{r.y:.4f})" for r in x.itertuples()) + "};")
open(OUT + 'reliab.tex', 'w').write(r"""% reliability diagram, generated by real/gen_v8.py
\begin{tikzpicture}
\begin{axis}[width=0.62\linewidth,height=5.6cm,xmin=0,xmax=0.8,ymin=0,ymax=0.8,grid=major,grid style={gray!20},
 xlabel={predicted 10-day burn probability},ylabel={observed frequency},legend style={font=\tiny,at={(1.03,0.5)},anchor=west},legend cell align=left]
\addplot[black,dotted,forget plot] coordinates {(0,0) (0.8,0.8)};
""" + "\n".join(pl) + r"""
\legend{prospective active,prospective late,rolling active,rolling late,oracle active,oracle late}
\end{axis}
\end{tikzpicture}
""")

for mname, f in (('oracle', 'fs_extra_green.json'), ('prospective', 'fs_extra_green_prospective.json'), ('rolling', 'fs_extra_green_rolling.json')):
    try:
        GV = json.load(open(f))['greenville']; P = [r['P_burn10'] for r in GV if r['T0'] <= '2021-08-01']
        tok[f'XV_gv_{mname}'] = f"{100*min(P):.0f}--{100*max(P):.0f}"
        tok[f'XV_gv10_{mname}'] = ", ".join(pd.Timestamp(r['T0']).strftime('%-d %b') for r in GV if r.get('in_QCDF_K10')) or 'none'
        tok[f'XV_gvlast_{mname}'] = f"{100*GV[-1]['P_burn10']:.0f}"
    except FileNotFoundError: pass

try:
    LP = json.load(open('fs_extra_loop_prospective.json'))['loop']
    c = np.array([r['closed_QCDF']['saved_pct'] for r in LP]); o = np.array([r['open_QCDF_24Jul']['saved_pct'] for r in LP]); v = np.array([r['closed_VaR']['saved_pct'] for r in LP])
    tok['XV_loop_closed'] = f"{c[1:].mean():.1f}"; tok['XV_loop_open'] = f"{o[1:].mean():.1f}"; tok['XV_loop_var'] = f"{v[1:].mean():.1f}"
except FileNotFoundError: pass

if CAL:
    GC = pd.read_csv('fire_caldor/dins_cells.csv'); dinsC = dict(zip(GC.h3, GC.dins_res_destroyed))
    GRC = gpd.read_file('fire_caldor/grid_r7.gpkg'); GRC['ign_date'] = pd.to_datetime(GRC['ign_date']); ignC = dict(zip(GRC.h3, GRC.ign_date))
    rows = []
    for m, lab in METH:
        rows.append(f"{lab} & " + " & ".join(f"${S(CAL, K, m).mean():.1f}\\pm{S(CAL, K, m).std(ddof=1)/np.sqrt(len(CAL)):.1f}$" for K in ('5', '10', '20')) + f" & ${dcov(CAL, '10', m, dinsC, ignC):.0f}$ \\\\")
    tok['XV_CALDROWS'] = "\n".join(rows); tok['XV_cald_n'] = str(len(CAL))
    tok['XV_cald_elig'] = f"{min(r['n_elig'] for r in CAL)}--{max(r['n_elig'] for r in CAL)}"
    for K in ('5', '10', '20'):
        for b in ('B4_value_at_risk', 'QCDF_h_only', 'CELF_greedy', 'B2_nearest_front', 'B1_communities'):
            d = S(CAL, K, 'QCDF_MILP') - S(CAL, K, b); lo, hi = cbb(d)
            tok[f'XV_cd_{b}_{K}'] = f"{d.mean():+.1f}"; tok[f'XV_cci_{b}_{K}'] = f"[{lo:+.1f}, {hi:+.1f}]"; tok[f'XV_cw_{b}_{K}'] = f"{(d>0.05).sum()}/{(abs(d)<=0.05).sum()}/{(d<-0.05).sum()}"
        for m, _ in METH: tok[f'XV_cs_{m}_{K}'] = f"{S(CAL, K, m).mean():.1f}"
    tok['XV_cald_red'] = str(sum(r['n_redund'] for r in CAL)); tok['XV_cald_comp'] = str(sum(r['n_compl'] for r in CAL))
    tok['XV_cald_dcov_comm'] = f"{dcov(CAL, '10', 'B1_communities', dinsC, ignC):.0f}"; tok['XV_cald_dcov_qcdf'] = f"{dcov(CAL, '10', 'QCDF_MILP', dinsC, ignC):.0f}"

try:
    A = json.load(open('aridity.json')); rows = []
    for nm, m in A['models'].items():
        pd_ = A['prospective_dixie'][nm]
        rows.append(f"{nm} & ${m['cv_ll']:,.0f}$ & ${m['cv_auc']:.3f}$ & ${pd_['ll']:,.0f}$ & ${pd_['auc']:.3f}$ & ${pd_['oe']:.2f}$ \\\\".replace(',', '{,}'))
    tok['XV_ARIDROWS'] = "\n".join(rows)
    for k, (g, lo, hi) in A['gains'].items():
        key = re.sub(r'[^A-Za-z0-9]', '', k)[:40]; tok['XV_g_' + key] = f"{g:+.0f} [{lo:+.0f}, {hi:+.0f}]"
    tok['XV_p30zero'] = f"{100*A['p30_zero']:.0f}"; tok['XV_p90zero'] = f"{100*A['p90_zero']:.0f}"
    cr = A['corr']; tok['XV_corr_DM_D'] = f"{cr['DM365']['D']:.2f}"; tok['XV_corr_VPD_E'] = f"{cr['VPD']['E']:.2f}"; tok['XV_corr_DM_E'] = f"{cr['DM365']['E']:.2f}"
    json.dump({k: v for k, v in tok.items() if k.startswith('XV_g_')}, open('aridity_gain_keys.json', 'w'), indent=1)
except FileNotFoundError: pass
json.dump(tok, open('v8_tokens.json', 'w'), indent=1)
fn = '../paper/v4_body_filled.tex'; s = open(fn).read()
for k, v in sorted(tok.items(), key=lambda kv: -len(kv[0])): s = s.replace('@@' + k + '@@', v)
open(fn, 'w').write(s)
print('unfilled', sorted(set(re.findall(r'@@(\w+)@@', s))))
for k, v in tok.items():
    if 'ROWS' not in k: print(k, v)
