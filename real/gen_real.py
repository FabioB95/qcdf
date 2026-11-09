"""Aggregate the Dixie Fire hindcast (real_*.json), write TikZ figures and fill @@X tokens in ../paper/v4_body_filled.tex (run after fill_v4.py)."""
import json, glob, sys, numpy as np, pandas as pd, h3
from scipy.stats import wilcoxon
OUT = '../paper/figures/tikz/'
R = []
for f in sorted(glob.glob('real_[ab].json')): R += json.load(open(f))
R = sorted(R, key=lambda r: r['T0']); nd = len(R)
CAL = json.load(open('calib.json'))
tok = {}
fmt = lambda x, d=1: f"{x:.{d}f}"

rows = []
names = [('constant', 'constant'), ('wind+slope', 'wind + slope (used)'), ('wind+slope+dryness', '+ dryness'),
         ('full (wind+slope+dryness+ERC)', '+ dryness + ERC'), ('wind+slope+relative dryness', '+ relative dryness'),
         ('wind+slope+relative dryness+relative ERC', '+ relative dryness + relative ERC')]
for key, lab in names:
    m = CAL['models'][key]; b = dict(zip(['0'] + list(m['use']), m['beta']))
    g = lambda *ks: next((f"${b[k]:+.2f}$" for k in ks if k in b), '--')
    rows.append(f"{lab} & {g('W')} & {g('S')} & {g('D', 'Q')} & {g('E', 'R')} & ${m['ll_test']:.0f}$ & ${m['auc_test']:.3f}$ \\\\")
tok['XCALROWS'] = "\n".join(rows)
m0 = CAL['models']['wind+slope']
tok['XNTR'] = f"{m0['n_train']:,}".replace(',', '{,}') + f" ({m0['pos_train']} ignitions)"
tok['XNTE'] = f"{m0['n_test']:,}".replace(',', '{,}') + f" ({m0['pos_test']} ignitions)"

G = None
import geopandas as gpd
G = gpd.read_file('grid_r7.gpkg'); G['ign_date'] = pd.to_datetime(G['ign_date'])
idx = {c: k for k, c in enumerate(G.h3)}
NB = [[idx[x] for x in h3.grid_ring(c, 1) if x in idx] for c in G.h3]
def domain(T0, rings=8):
    burned = (G.ign_date <= T0).values
    dist = np.full(len(G), 99); fr = list(np.where(burned)[0]); dist[fr] = 0
    for r in range(1, rings + 1):
        nx = []
        for i in fr:
            for j in NB[i]:
                if dist[j] == 99: dist[j] = r; nx.append(j)
        fr = nx
    return burned, dist
dfrac = []
for r in R:
    T0 = pd.Timestamp(r['T0']); burned, dist = domain(T0)
    un = (dist <= 8) & ~burned
    nb = (G.ign_date > T0) & (G.ign_date <= T0 + pd.Timedelta(days=10))
    dfrac.append(nb[un].mean())
st = [r['static'] for r in R]
P = np.array([s['P_opt'] for s in st])
tok.update(XND=str(nd), XPOPTMEAN=fmt(100 * P.mean()), XPOPTMIN=fmt(100 * P.min()), XPOPTMAX=fmt(100 * P.max()),
           XAMP=fmt(np.mean([s['amp'] for s in st])),
           XQBURN=fmt(100 * sum(s['qcdf_burned_cells'] for s in st) / (3 * nd)),
           XCBURN=fmt(100 * np.mean([s['frac_cand_burned'] for s in st])),
           XDBURN=fmt(100 * np.mean(dfrac)),
           XBETTER=str(sum(s['qcdf_burned_housing'] > s['mean_burned_housing'] + 1e-9 for s in st)),
           XWORSE=str(sum(s['qcdf_burned_housing'] < s['mean_burned_housing'] - 1e-9 for s in st)),
           XRANK=fmt(np.median([s['rank_burned_housing'] for s in st]), 0),
           XDOMMIN=str(min(r['n_domain'] for r in R)), XDOMMAX=str(max(r['n_domain'] for r in R)))
tok['XTIES'] = str(nd - int(tok['XBETTER']) - int(tok['XWORSE']))

FA = {r['T0']: r for f in sorted(glob.glob('fa_[ab].json')) for r in json.load(open(f))}
SPLIT = '2021-08-05'
phases = [('all', lambda d: True), ('early', lambda d: d <= SPLIT), ('late', lambda d: d > SPLIT)]
plab = {'all': 'all dates', 'early': 'active phase (24 Jul--5 Aug)', 'late': 'late phase (6 Aug--11 Sep)'}
rows = []
for ph, f in phases:
    X = [FA[r['T0']] for r in R if f(r['T0'])]
    rows.append(f"{plab[ph]} & {len(X)} & ${np.median([x['auc'] for x in X]):.2f}$ & "
                f"${np.mean([x['pos'] for x in X]):.0f}$ / ${np.mean([x['pred'] for x in X]):.0f}$ & "
                f"${np.mean([x['hous_burn_obs'] for x in X]):.0f}$ / ${np.mean([x['hous_burn_pred'] for x in X]):.0f}$ & "
                f"${100*np.mean([x['cand_burn_obs'] for x in X]):.0f}$ / ${100*np.mean([x['cand_burn_pred'] for x in X]):.0f}$ \\\\")
tok['XFCROWS'] = "\n".join(rows)

for ph, f in phases:
    X = [r['static'] for r in R if f(r['T0'])]
    tok[f'XQB_{ph}'] = fmt(100 * np.mean([x['qcdf_burned_cells'] / 3 for x in X]))
    tok[f'XCB_{ph}'] = fmt(100 * np.mean([x['frac_cand_burned'] for x in X]))
    tok[f'XBET_{ph}'] = str(sum(x['qcdf_burned_housing'] > x['mean_burned_housing'] + 1e-9 for x in X))
    tok[f'XWOR_{ph}'] = str(sum(x['qcdf_burned_housing'] < x['mean_burned_housing'] - 1e-9 for x in X))
    tok[f'XN_{ph}'] = str(len(X))

ks = [('random', 'random plans'), ('static_ranking', 'static ranking'), ('myopic', 'myopic greedy'),
      ('best_sampled', 'best of 2000 sampled plans'), ('seq_anneal', 'seq.\\ anneal (linear)'), ('seq_surrogate', 'seq.\\ surrogate, exact')]
Sall = {k: np.array([100 * (r['seq']['loss']['none'] - r['seq']['loss'][k]) / r['seq']['loss']['none'] for r in R]) for k, _ in ks}
dts = np.array([r['T0'] for r in R])
r2 = np.array([r['seq']['r2_holdout'] for r in R])
tok['XR2'] = fmt(np.median(r2), 3); tok['XR2MIN'] = fmt(r2.min(), 3)
def pstr(p): return f"{p:.3f}" if p >= 0.001 else f"{p:.0e}".replace('e-0', '\\times10^{-') + '}'
rows = []; stat = {}
for k, lab in ks:
    cells = []
    for ph, f in phases:
        m = np.array([f(d) for d in dts]); x = Sall[k][m]
        cells.append(f"${x.mean():.2f}\\pm{x.std(ddof=1)/np.sqrt(m.sum()):.2f}$")
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule")
for k, lab in [('static_ranking', 'static ranking'), ('seq_anneal', 'seq.\\ anneal'), ('seq_surrogate', 'seq.\\ surrogate')]:
    cells = []
    for ph, f in phases:
        m = np.array([f(d) for d in dts]); dd = Sall[k][m] - Sall['myopic'][m]
        b_, t_, w_ = (dd > 0.05).sum(), (abs(dd) <= 0.05).sum(), (dd < -0.05).sum()
        p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
        cells.append(f"{b_}/{t_}/{w_} ($p={pstr(p)}$)")
        stat[(k, ph)] = (dd.mean(), b_, t_, w_, p)
        tok[f'XD_{k}_{ph}'] = fmt(dd.mean(), 2); tok[f'XW_{k}_{ph}'] = f"{b_}/{t_}/{w_}"; tok[f'XP_{k}_{ph}'] = pstr(p)
    rows.append(f"{lab} vs.\\ myopic & " + " & ".join(cells) + " \\\\")
tok['XSEQROWS'] = "\n".join(rows)
for ph, f in phases:
    m = np.array([f(d) for d in dts])
    for k in Sall: tok[f'XS_{k}_{ph}'] = fmt(Sall[k][m].mean(), 2)
    tok[f'XR2_{ph}'] = fmt(np.median(r2[m]), 3)
S = Sall
json.dump(dict(S={k: v.tolist() for k, v in S.items()}, stat={f"{k}|{p}": [float(x) for x in v] for (k, p), v in stat.items()},
               tok=tok, dates=list(dts)), open('real_summary.json', 'w'), indent=1)

coords = " ".join(f"({i},{S[k].mean():.2f}) +- (0,{S[k].std(ddof=1)/np.sqrt(nd):.2f})" for i, (k, _) in enumerate(ks))
lab = ",".join("{" + l + "}" for _, l in ks)
dk = [k for k, _ in ks if k not in ('myopic', 'random')]
dl = ",".join("{" + dict(ks)[k] + "}" for k in dk)
ser = []
for ph, f, col in [('early', phases[1][1], 'qsel'), ('late', phases[2][1], 'qblue')]:
    m = np.array([f(d) for d in dts])
    ser.append(f"\\addplot[fill={col}!55,draw={col},error bars/.cd,y dir=both,y explicit] coordinates {{" + " ".join(
        f"({i},{(S[k][m]-S['myopic'][m]).mean():.2f}) +- (0,{(S[k][m]-S['myopic'][m]).std(ddof=1)/np.sqrt(m.sum()):.2f})" for i, k in enumerate(dk)) + "};")
ymax = max(S[k].mean() for k in S) * 1.35
open(OUT + 'realseq.tex', 'w').write(rf"""% Dixie Fire sequential hindcast, generated by real/gen_real.py ({nd} decision dates)
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.5cm}},height=5.2cm,ymajorgrids,grid style={{gray!20}},
 x tick label style={{rotate=35,anchor=east,font=\scriptsize}},title style={{font=\small}}]
\nextgroupplot[ybar,bar width=9pt,width=0.52\linewidth,ymin=0,ymax={ymax:.1f},title={{(a) housing loss avoided (\%), all dates}},
 xtick={{0,...,{len(ks)-1}}},xticklabels={{{lab}}},nodes near coords,every node near coord/.append style={{font=\tiny,yshift=3pt}},point meta=y]
\addplot[fill=qblue!50,draw=qblue,error bars/.cd,y dir=both,y explicit] coordinates {{{coords}}};
\nextgroupplot[ybar,bar width=6pt,width=0.48\linewidth,title={{(b) difference to myopic (points)}},
 xtick={{0,...,{len(dk)-1}}},xticklabels={{{dl}}},ymin=-2.4,ymax=1.6,legend style={{font=\tiny,at={{(0.03,0.97)}},anchor=north west}}]
{ser[0]}
{ser[1]}
\legend{{active phase,late phase}}
\draw[black] (axis cs:-0.5,0) -- (axis cs:{len(dk)-0.5},0);
\end{{groupplot}}
\end{{tikzpicture}}
""")

MAPDATE = sys.argv[1] if len(sys.argv) > 1 else '2021-08-04'
rec = next(r for r in R if r['T0'] == MAPDATE)
sys.argv = ['x']; import realcase as RC
T0 = pd.Timestamp(MAPDATE); rng = np.random.default_rng(int(T0.strftime('%m%d')))
D = RC.Domain(T0, rng); _, pb = D.run([], D.UA)
var = D.value * pb; var[D.burned0] = -1; cand = D.cells[np.argsort(-var)[:RC.N_STAT]]
sel = {idx[c] for c in rec['static']['qcdf_cells']}
burned, dist = domain(T0)
show = np.where(dist <= 8)[0]
newb = ((G.ign_date > T0) & (G.ign_date <= T0 + pd.Timedelta(days=10))).values
lat0, lon0 = G.lat.values[show].mean(), G.lon.values[show].mean()
proj = lambda la, lo: ((lo - lon0) * np.cos(np.radians(lat0)) * 111.32, (la - lat0) * 110.57)
pts = {i: [proj(a, b) for a, b in h3.cell_to_boundary(G.h3.iloc[i])] for i in show}
allx = [p[0] for v in pts.values() for p in v]; ally = [p[1] for v in pts.values() for p in v]
sc = 11.5 / (max(allx) - min(allx))
L = [f"% Dixie Fire map {MAPDATE}, generated by real/gen_real.py (scale {sc:.3f} cm/km)",
     "\\begin{tikzpicture}[x=1cm,y=1cm]"]
def poly(i, style):
    return f"\\filldraw[{style}] " + " -- ".join(f"({x*sc:.2f},{y*sc:.2f})" for x, y in pts[i]) + " -- cycle;"
for i in show:
    if i in sel: st_ = "fill=qsel,draw=black,line width=0.6pt"
    elif burned[i]: st_ = "fill=gray!45,draw=gray!60,line width=0.1pt"
    elif newb[i]: st_ = "fill=orange!60,draw=orange!80!black,line width=0.1pt"
    else: st_ = "fill=white,draw=gray!35,line width=0.1pt"
    L.append(poly(i, st_))
for i in cand:
    if i not in sel: L.append("\\draw[black,line width=0.7pt] " + " -- ".join(f"({x*sc:.2f},{y*sc:.2f})" for x, y in pts[i]) + " -- cycle;")
H = G.HOUSING20.values
for i in show:
    if H[i] >= 5 and not burned[i]:
        x, y = proj(G.lat.values[i], G.lon.values[i])
        L.append(f"\\fill[qblue,opacity=0.75] ({x*sc:.2f},{y*sc:.2f}) circle ({0.012*np.sqrt(H[i]):.3f});")

x0, y0 = min(allx) * sc, min(ally) * sc - 0.4
L.append(f"\\draw[thick] ({x0:.2f},{y0:.2f}) -- ({x0+20*sc:.2f},{y0:.2f}) node[midway,below,font=\\scriptsize]{{20 km}};")
xa, ya = max(allx) * sc - 0.3, max(ally) * sc - 0.8
L.append(f"\\draw[-latex,thick] ({xa:.2f},{ya:.2f}) -- ({xa:.2f},{ya+0.7:.2f}) node[above,font=\\scriptsize]{{N}};")

lx, ly = max(allx) * sc + 0.4, min(ally) * sc + 3.2
items = [("fill=gray!45,draw=gray!60", "burned by $T_0$"), ("fill=orange!60,draw=orange!80!black", "burned in next 10 days"),
         ("fill=white,draw=black,line width=0.7pt", "static candidate"), ("fill=qsel,draw=black", "QCDF selection ($k=3$)")]
for n_, (s_, t_) in enumerate(items):
    yy = ly - 0.55 * n_
    L.append(f"\\filldraw[{s_}] ({lx:.2f},{yy:.2f}) rectangle ({lx+0.35:.2f},{yy+0.3:.2f}); \\node[right,font=\\scriptsize] at ({lx+0.4:.2f},{yy+0.15:.2f}) {{{t_}}};")
yy = ly - 0.55 * 4
L.append(f"\\fill[qblue,opacity=0.75] ({lx+0.17:.2f},{yy+0.15:.2f}) circle ({0.012*np.sqrt(100):.3f}); \\node[right,font=\\scriptsize] at ({lx+0.4:.2f},{yy+0.15:.2f}) {{100 housing units}};")
L.append("\\end{tikzpicture}")
open(OUT + 'realmap.tex', 'w').write("\n".join(L) + "\n")
tok['XMAPDATE'] = T0.strftime('%-d %B %Y')
print('map', MAPDATE, 'cells shown', len(show), 'cand burned', int(newb[cand].sum()), 'sel burned', int(newb[list(sel)].sum()))

MF = json.load(open('mf_calib.json')); TR = json.load(open('transfer.json'))
M = MF['models']; ci = MF['boot_ci']; f2 = lambda x: f"{x:+.2f}"
tok['XMFROWS'] = f"{MF['n_rows']:,}".replace(',', '{,}'); tok['XMFPOS'] = f"{MF['n_pos']:,}".replace(',', '{,}')
tok['XMF_WS_W'] = f2(M['wind+slope']['beta']['W']); tok['XMF_WS_S'] = f2(M['wind+slope']['beta']['S'])
fl = lambda x: f"{x:,.0f}".replace(',', '{,}').replace('-', '-')
tok['XMF_const_LL'] = fl(M['constant']['cv_ll']); tok['XMF_WS_LL'] = fl(M['wind+slope']['cv_ll'])
tok['XMF_WSD_D'] = f2(M['+ dryness']['beta']['D']); tok['XMF_WSD_DFE'] = f2(M['+ dryness']['beta_fe']['D'])
tok['XCI_WSD_lo'], tok['XCI_WSD_hi'] = f2(ci['WSD'][0]), f2(ci['WSD'][1])
tok['XCI_WSDfe_lo'], tok['XCI_WSDfe_hi'] = f2(ci['WSD_fe'][0]), f2(ci['WSD_fe'][1])
tok['XMF_WSDE_LL'] = fl(M['+ dryness + ERC']['cv_ll']); tok['XMF_WSDE_AUC'] = f"{M['+ dryness + ERC']['cv_auc']:.3f}"
tok['XMF_WSDE_D'] = f2(M['+ dryness + ERC']['beta']['D']); tok['XMF_WSDE_E'] = f2(M['+ dryness + ERC']['beta']['E'])
tok['XCI_DE_D_lo'], tok['XCI_DE_D_hi'] = f2(ci['WSDE_D'][0]), f2(ci['WSDE_D'][1])
tok['XCI_DE_E_lo'], tok['XCI_DE_E_hi'] = f2(ci['WSDE_E'][0]), f2(ci['WSDE_E'][1])
Zt = np.load('mf_transitions.npz'); tok['XDE_corr'] = f"{np.corrcoef(Zt['dj'], Zt['ej'])[0, 1]:.2f}"
PF = MF['per_fire']; tok['XPF_n'] = str(len(PF)); tok['XPF_min'] = f2(min(p['D'] for p in PF)); tok['XPF_max'] = f2(max(p['D'] for p in PF))
tok['XPF_dixie'] = f2(next(p['D'] for p in PF if p['fire'] == 'Dixie'))
rows = []
for key, lab in [('constant', 'constant'), ('wind+slope', 'wind + slope'), ('+ dryness', '+ dryness'), ('+ ERC', '+ ERC'),
                 ('+ dryness + ERC', '+ dryness + ERC'), ('+ relative dryness', '+ relative dryness'), ('+ relative dryness + relative ERC', '+ rel.\\ dryness + rel.\\ ERC')]:
    m = M[key]; b = m['beta']; g = lambda *ks: next((f"${b[k]:+.2f}$" for k in ks if k in b), '--')
    bfe = m['beta_fe']; dfe = next((f" (FE ${bfe[k]:+.2f}$)" for k in ('D', 'Q') if k in bfe), '')
    rows.append(f"{lab} & {g('W')} & {g('S')} & {g('D','Q')}{dfe} & {g('E','R')} & ${m['cv_ll']:,.0f}$ & ${m['cv_auc']:.3f}$ \\\\".replace(',', '{,}'))
tok['XMFROWS_TAB'] = "\n".join(rows)
best = min((v for k, v in TR.items() if k != 'constant'), key=lambda v: -v['test']['ll'])
tok['XTR_best_LL'] = f"{max(v['test']['ll'] for k, v in TR.items() if k != 'constant'):.0f}"
tok['XTR_best_AUC'] = f"{max(v['test']['auc'] for v in TR.values()):.3f}"
tok['XTR_pred'] = f"{TR['wind+slope']['test']['pred']:.0f}"; tok['XTR_obs'] = str(TR['wind+slope']['test']['obs'])

fn = '../paper/v4_body_filled.tex'; s = open(fn).read()
for k, v in sorted(tok.items(), key=lambda kv: -len(kv[0])): s = s.replace('@@' + k + '@@', v)
open(fn, 'w').write(s)
import re; print('unfilled', sorted(set(re.findall(r'@@(\w+)@@', s))))
for k, v in tok.items():
    if k not in ('XCALROWS', 'XSEQROWS', 'XFCROWS'): print(k, v)
print(tok['XSEQROWS']); print(tok['XFCROWS'])
