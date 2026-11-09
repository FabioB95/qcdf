"""Tokens and table for the full-scale astronomical study (astro_full_{prospective,oracle}.json)."""
import json, numpy as np
P = json.load(open('astro_full_prospective.json')); O = json.load(open('astro_full_oracle.json'))
c = P['cal']; tok = {}
tok.update(XA_ntrain=str(c['n_train']), XA_postrain=str(c['pos_train']), XA_ntest=str(c['n_test']), XA_postest=str(c['pos_test']),
           XA_auc=f"{c['auc_test']:.2f}", XA_slope=f"{c['slope']:.2f}", XA_oe=f"{c['obs_exp']:.2f}", XA_brier=f"{c['brier']:.4f}",
           XA_auccurv=f"{c['auc_curv_only']:.2f}", XA_aucflux=f"{c['auc_flux_only']:.2f}", XA_aucor=f"{c['auc_oracle_insample']:.2f}",
           XA_ncells=f"{P['n_cells']:,}".replace(',', '{,}'), XA_nJ=f"{P['n_J']:,}".replace(',', '{,}'))
names = [('QCDF_MILP', 'QCDF, exact (MILP)'), ('QCDF_h_only', 'QCDF, $J=0$'), ('greedy_exact', 'greedy on $F$'),
         ('most_curved', 'most curved spectra'), ('brightest', 'brightest sources'), ('most_sources', 'most sources'), ('random', 'random')]
Ks = ['20', '50', '200']
rows = []
for k, lab in names:
    cells = []
    for K in Ks:
        e = P['K'][K]['eval'][k]; cells += [f"{e['F_model']:.2f}", f"{e['hits']:.1f}", f"{e['n_hit_any']}"]
    eo = O['K']['200']['eval'][k]; cells += [f"{eo['F_model']:.2f}", f"{eo['n_hit_any']}"]
    rows.append(lab + ' & ' + ' & '.join(cells) + r' \\')
tok['XA_ROWS'] = '\n'.join(rows)
g = lambda m, K, k: m['K'][K]['eval'][k]
cg = [100 * (g(P, K, 'QCDF_MILP')['F_model'] / g(P, K, 'QCDF_h_only')['F_model'] - 1) for K in ('20', '50', '100', '200')]
tok['XA_cgain_min'] = f"{min(cg):.0f}"; tok['XA_cgain_max'] = f"{max(cg):.0f}"
tok['XA_q200'] = str(g(P, '200', 'QCDF_MILP')['n_hit_any']); tok['XA_c200'] = str(g(P, '200', 'most_curved')['n_hit_any'])
tok['XA_b200'] = str(g(P, '200', 'brightest')['n_hit_any']); tok['XA_s200'] = str(g(P, '200', 'most_sources')['n_hit_any'])
tok['XA_r200'] = str(g(P, '200', 'random')['n_hit_any']); tok['XA_h200'] = str(g(P, '200', 'QCDF_h_only')['n_hit_any'])
tok['XA_o200'] = str(g(O, '200', 'QCDF_MILP')['n_hit_any'])
ci = g(P, '200', 'brightest')['hits_diff_ci']; tok['XA_ci_b'] = f"[{ci[0]:.1f},\\,{ci[1]:.1f}]"
ci = g(P, '200', 'most_curved')['hits_diff_ci']; tok['XA_ci_c'] = f"[{ci[0]:.1f},\\,{ci[1]:.1f}]"
ci = g(P, '200', 'QCDF_h_only')['hits_diff_ci']; tok['XA_ci_h'] = f"[{ci[0]:.1f},\\,{ci[1]:.1f}]"
json.dump(tok, open('astro_tokens.json', 'w'), indent=1)
fn = '../paper/v4_body_filled.tex'; s = open(fn).read()
for k_, v in sorted(tok.items(), key=lambda kv: -len(kv[0])): s = s.replace('@@' + k_ + '@@', v)
open(fn, 'w').write(s)
print({k: v for k, v in tok.items() if k != 'XA_ROWS'}); print(tok['XA_ROWS'])
