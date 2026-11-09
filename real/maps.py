"""Decision maps (v11). Dixie, 31 July 2021, prospective model: three panels with what was known at the decision (burned area, predicted 10-day burn probability, housing, hospitals, communities) and the QCDF plans for K = 5, 10, 20; a fourth panel with what happened afterwards (cells burned in the next 10 days, destroyed residences from DINS) and the K = 10 plan. Sky: unidentified sources coloured by predicted probability with the K = 20, 50, 200 plans, and the DR4 identifications in a separate panel."""
import sys, json, numpy as np, pandas as pd, geopandas as gpd, h3, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
_ARGV = list(sys.argv); sys.argv = ['x']
import fullscale as FS
sys.argv = _ARGV
G = FS.G; T0 = '2021-07-31'
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
P = FS.Problem(T0, model='prospective'); _, pb = P.sim.losses([], 2000, 21)
R = [r for r in json.load(open('fsP_act.json')) if r['T0'] == T0][0]
plans = {K: set(R['K'][K]['eval']['QCDF_MILP']['plan']) for K in ('5', '10', '20')}
cells = P.D.cells; hx = G.h3.values[cells]
poly = [np.array([(b, a) for a, b in h3.cell_to_boundary(c)]) for c in hx]
burn0 = P.D.burned0; H = P.D.value
obs = P.obs_burn
dins = pd.read_csv('dins_cells.csv').set_index('h3')
dres = np.array([dins.dins_res_destroyed.get(c, 0) for c in hx])
hc = pd.read_csv('../realdata/hcai/lhfl_2021_06_30.csv', encoding='latin1', low_memory=False)
hc = hc[hc.LATITUDE.notna() & hc.LICENSE_CATEGORY_DESC.isin(['General Acute Care Hospital', 'Skilled Nursing Facility']) & (hc.FACILITY_STATUS_DESC == 'Open')]
xs = np.concatenate(poly)[:, 0]; ys = np.concatenate(poly)[:, 1]
hc = hc[(hc.LONGITUDE > xs.min()) & (hc.LONGITUDE < xs.max()) & (hc.LATITUDE > ys.min()) & (hc.LATITUDE < ys.max())]
places = gpd.read_file('zip://../realdata/tl_2021_06_place.zip', bbox=(xs.min(), ys.min(), xs.max(), ys.max())).to_crs(4326)
places['c'] = places.geometry.representative_point()
cent = np.array([h3.cell_to_latlng(c) for c in hx])
def housing_of(pt):
    d = (cent[:, 0] - pt.y) ** 2 + (cent[:, 1] - pt.x) ** 2; return H[np.argsort(d)[:3]].sum()
places['H'] = [housing_of(p) for p in places.c]
lab = places[places.NAME.isin(['Greenville', 'Chester', 'Quincy', 'Taylorsville', 'Crescent Mills', 'Westwood', 'Prattville', 'Butte Meadows', 'Keddie'])]

fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6))
allb = (G.ign_date <= pd.Timestamp(T0)).values
bpoly = [np.array([(b, a) for a, b in h3.cell_to_boundary(c)]) for c in G.h3.values[allb]]
cmap = plt.get_cmap('YlOrRd')
for ax, K in zip(axs.flat[:3], ('5', '10', '20')):
    col = [('0.25' if b else cmap(min(p / 0.5, 1.0))) for b, p in zip(burn0, pb)]
    ax.add_collection(PolyCollection(poly, facecolors=col, edgecolors='0.85', linewidths=0.2))
    ax.add_collection(PolyCollection(bpoly, facecolors='0.25', edgecolors='0.25', linewidths=0.2))
    sel = [k for k, c in enumerate(hx) if c in plans[K]]
    ax.add_collection(PolyCollection([poly[k] for k in sel], facecolors=(0.12, 0.31, 0.61, 0.55), edgecolors='#0b2b66', linewidths=1.4))
    big = H > 50; ax.scatter(cent[big, 1], cent[big, 0], s=np.sqrt(H[big]) * 0.9, c='none', edgecolors='k', linewidths=0.4)
    ax.scatter(hc.LONGITUDE, hc.LATITUDE, marker='P', s=28, c='#c00000', edgecolors='w', linewidths=0.4, zorder=5)
    for _, r in lab.iterrows(): ax.text(r.c.x, r.c.y + 0.012, r.NAME, fontsize=6, ha='center', va='bottom', color='k', bbox=dict(fc='w', ec='none', alpha=0.6, pad=0.3))
    ax.set_title(f'({"abc"[("5","10","20").index(K)]}) known on 31 July; plan $K={K}$', fontsize=8)
ax = axs.flat[3]
col = [('0.25' if b else ('#f08c2e' if o else '0.95')) for b, o in zip(burn0, obs)]
ax.add_collection(PolyCollection(poly, facecolors=col, edgecolors='0.85', linewidths=0.2))
ax.add_collection(PolyCollection(bpoly, facecolors='0.25', edgecolors='0.25', linewidths=0.2))
sel = [k for k, c in enumerate(hx) if c in plans['10']]
ax.add_collection(PolyCollection([poly[k] for k in sel], facecolors='none', edgecolors='#0b2b66', linewidths=1.6))
d = dres > 0; ax.scatter(cent[d, 1], cent[d, 0], s=np.sqrt(dres[d]) * 3, c='k', marker='x', linewidths=0.8)
for _, r in lab.iterrows(): ax.text(r.c.x, r.c.y + 0.012, r.NAME, fontsize=6, ha='center', va='bottom', bbox=dict(fc='w', ec='none', alpha=0.6, pad=0.3))
ax.set_title('(d) after the decision (not known on 31 July)', fontsize=8)
for ax in axs.flat:
    ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.min(), ys.max()); ax.set_aspect(1 / np.cos(np.radians(40))); ax.tick_params(labelsize=6)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 0.5)); cax = fig.add_axes([0.9, 0.56, 0.015, 0.36]); cb = fig.colorbar(sm, cax=cax)
cb.set_label('predicted 10-day burn probability (prospective model)', fontsize=7)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc='0.25', label='burned by 31 July'), Patch(fc=(0.12, 0.31, 0.61, 0.55), ec='#0b2b66', lw=1.4, label='QCDF plan'),
                    Line2D([], [], ls='', marker='o', mfc='none', mec='k', label='housing (size)'), Line2D([], [], ls='', marker='P', c='#c00000', label='hospital / nursing'),
                    Patch(fc='#f08c2e', label='burned 1-10 Aug (observed)'), Line2D([], [], ls='', marker='x', c='k', label='destroyed residences (DINS)')],
           loc='lower center', ncol=3, fontsize=7, frameon=False)
fig.subplots_adjust(left=0.07, right=0.88, bottom=0.1, top=0.96, wspace=0.18, hspace=0.16)
fig.savefig('../paper/figures/map_dixie.pdf'); print('fire map ok', flush=True)

A = np.load('../astro/astro_map.npz'); import healpy as hp
def moll(ax, ra, dec, **kw):
    x = np.radians(-((ra + 180) % 360 - 180)); ax.scatter(x, np.radians(dec), **kw)
fig = plt.figure(figsize=(7.2, 3.3))
th, ph = hp.pix2ang(64, A['cells'])
for k, (title, mode) in enumerate((('(a) known from DR3: predicted probability and plans', 'pre'), ('(b) identified in DR4 (not known when planning)', 'post'))):
    ax = fig.add_subplot(1, 2, k + 1, projection='mollweide'); ax.grid(lw=0.3, alpha=0.5); ax.set_xticklabels([]); ax.tick_params(labelsize=5)
    vis = A['dec'] < 44
    ax.fill_between(np.linspace(-np.pi, np.pi, 50), np.radians(44), np.pi / 2, color='0.9', zorder=0)
    if mode == 'pre':
        o = np.argsort(A['P']); moll(ax, A['ra'][o], A['dec'][o], c=A['P'][o], s=0.8, cmap='viridis', vmin=0, vmax=0.06)
        for key, c, s in (('plan200', '#f08c2e', 10), ('plan50', '#c00000', 18), ('plan20', 'k', 26)):
            th_, ph_ = hp.pix2ang(64, A[key]); moll(ax, np.degrees(ph_), 90 - np.degrees(th_), s=s, facecolors='none', edgecolors=c, linewidths=0.9)
    else:
        moll(ax, A['ra'], A['dec'], c='0.8', s=1)
        th_, ph_ = hp.pix2ang(64, A['plan200']); moll(ax, np.degrees(ph_), 90 - np.degrees(th_), s=6, facecolors='none', edgecolors='#f08c2e', linewidths=0.6)
        y = A['y'] == 1; moll(ax, A['ra'][y], A['dec'][y], marker='*', s=22, c='#1f4e9c')
    ax.set_title(title, fontsize=7)
fig.legend(handles=[Line2D([], [], ls='', marker='o', mfc='none', mec='k', label='plan $K=20$'), Line2D([], [], ls='', marker='o', mfc='none', mec='#c00000', label='$K=50$'),
                    Line2D([], [], ls='', marker='o', mfc='none', mec='#f08c2e', label='$K=200$'), Line2D([], [], ls='', marker='*', c='#1f4e9c', label='identified in DR4'),
                    Patch(fc='0.9', label='not visible ($\\delta>44^\\circ$)')], loc='lower center', ncol=5, fontsize=6.5, frameon=False)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.92, wspace=0.08)
fig.savefig('../paper/figures/map_sky.pdf'); print('sky map ok', flush=True)
