"""Robustness of the allocation comparison to the spread model: 10 fire-cluster bootstrap refits of the prospective model and two intercept shifts (calibration in the large), active phase, K=10."""
import json, glob, re, numpy as np, pandas as pd
R = []
for f in ('pert_a.json', 'pert_b.json'): R += json.load(open(f))
D = pd.DataFrame(R)
for c in ('QCDF_h_only', 'B4_value_at_risk', 'B2_nearest_front', 'B1_communities'): D['d_' + c] = D.QCDF_MILP - D[c]
g = D.groupby('model')[['QCDF_MILP', 'd_QCDF_h_only', 'd_B4_value_at_risk', 'd_B2_nearest_front', 'd_B1_communities', 'none']].mean()
P = [json.load(open('fsP_act.json'))]; P = P[0]
base = dict(QCDF_MILP=np.mean([r['K']['10']['eval']['QCDF_MILP']['saved_pct'] for r in P]),
            **{'d_' + c: np.mean([r['K']['10']['eval']['QCDF_MILP']['saved_pct'] - r['K']['10']['eval'][c]['saved_pct'] for r in P]) for c in ('QCDF_h_only', 'B4_value_at_risk', 'B2_nearest_front', 'B1_communities')})
boot = g[g.index.str.contains('boot')]
cols = ['QCDF_MILP', 'd_QCDF_h_only', 'd_B4_value_at_risk', 'd_B2_nearest_front', 'd_B1_communities']
f = lambda v, s=True: (f"${v:+.1f}$" if s else f"${v:.1f}$")
rows = ["prospective (main) & " + " & ".join(f(base[c], c != 'QCDF_MILP') for c in cols) + " \\\\",
        f"bootstrap refits, mean ({len(boot)}) & " + " & ".join(f(boot[c].mean(), c != 'QCDF_MILP') for c in cols) + " \\\\",
        "bootstrap refits, range & " + " & ".join(f"${boot[c].min():.1f}$--${boot[c].max():.1f}$" for c in cols) + " \\\\"]
for nm, lab in (('pert/shift_m.json', 'intercept $-0.7$ (fewer ignitions)'), ('pert/shift_p.json', 'intercept $+0.7$ (more ignitions)')):
    rows.append(lab + " & " + " & ".join(f(g.loc[nm, c], c != 'QCDF_MILP') for c in cols) + " \\\\")
tok = dict(XS_ROWS="\n".join(rows), XS_nboot=str(len(boot)), XS_honly_min=f"{boot.d_QCDF_h_only.min():+.1f}", XS_honly_max=f"{boot.d_QCDF_h_only.max():+.1f}",
           XS_var_min=f"{boot.d_B4_value_at_risk.min():+.1f}", XS_var_max=f"{boot.d_B4_value_at_risk.max():+.1f}",
           XS_honly_m=f"{g.loc['pert/shift_m.json', 'd_QCDF_h_only']:+.1f}", XS_honly_p=f"{g.loc['pert/shift_p.json', 'd_QCDF_h_only']:+.1f}",
           XS_neg_honly=str(int((D[D.model.str.contains('boot')].d_QCDF_h_only < -0.05).sum())), XS_n_cases=str(int(D.model.str.contains('boot').sum())))
json.dump(tok, open('pert_tokens.json', 'w'), indent=1)
fn = '../paper/v4_body_filled.tex'; s = open(fn).read()
for k, v in sorted(tok.items(), key=lambda kv: -len(kv[0])): s = s.replace('@@' + k + '@@', v)
open(fn, 'w').write(s)
print({k: v for k, v in tok.items() if k != 'XS_ROWS'}); print(tok['XS_ROWS'])

bh = boot.d_QCDF_h_only; bv = boot.d_B4_value_at_risk
hm, hp = g.loc['pert/shift_m.json', 'd_QCDF_h_only'], g.loc['pert/shift_p.json', 'd_QCDF_h_only']
vm, vp = g.loc['pert/shift_m.json', 'd_B4_value_at_risk'], g.loc['pert/shift_p.json', 'd_B4_value_at_risk']
allpos_h = bh.min() > 0 and hm > 0 and hp > 0
allpos_v = bv.min() > 0 and vm > 0 and vp > 0
v = (r"The value of the couplings " + ("survives every perturbation" if allpos_h else "does not survive every perturbation") +
     f": QCDF minus $J=0$ is between ${bh.min():+.1f}$ and ${bh.max():+.1f}$ points across the bootstrap refits "
     f"and ${hm:+.1f}$ and ${hp:+.1f}$ under the two intercept shifts (main model ${base['d_QCDF_h_only']:+.1f}$). ")
v += (r"The margin over the value-at-risk rule is smaller and " + ("also keeps its sign" if allpos_v else "changes sign under some models") +
      f" (${min(bv.min(), vm, vp):+.1f}$ to ${max(bv.max(), vm, vp):+.1f}$). ")
v += (f"Per date, the couplings lose to $J=0$ in {tok['XS_neg_honly']} of {tok['XS_n_cases']} bootstrap cases, by small amounts. "
      r"The coupling gain is therefore not an artefact of one fitted model, whereas the near-tie with value at risk "
      r"is exactly that: a tie, whose sign depends on the model." if allpos_h else
      r"The size, and in some models the sign, of the coupling gain is therefore model-dependent.")
tok['XS_VERDICT'] = v; tok['XS_allpos_h'] = str(allpos_h); tok['XS_allpos_v'] = str(allpos_v)
json.dump(tok, open('pert_tokens.json', 'w'), indent=1)
s = open(fn).read(); s = s.replace('@@XS_VERDICT@@', v); open(fn, 'w').write(s)
print(v)
