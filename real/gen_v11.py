"""v11 tokens: validation protocol in both domains (XP_* wildfire, XS2_* sky, XP_PROTROWS summary table). Inputs: fc_[ab].json, calib_recal.json, calib_recal_check.json, pert_recal.json, ops.json, stats_v11.json, calib_caldor.json, dq.json, fsP_act.json, ../astro/astro_v11.json, ../astro/astro_full_prospective.json."""
import json, glob, numpy as np
tok = {}
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; sg = lambda x: f"{x:+.1f}"
ACT = json.load(open('fsP_act.json'))

FC = sorted(sum([json.load(open(f)) for f in sorted(glob.glob('fc_[ab].json'))], []), key=lambda r: r['T0'])
cols = ['FC_QCDF', 'FC_h_only', 'FC_VaR', 'FC_control_QCDF', 'OBS_QCDF', 'OBS_h_only', 'OBS_VaR', 'B2_nearest_front', 'B1_communities']
rows = [r['T0'][5:] + ' & ' + ' & '.join(f1(r['saved'][c]) for c in cols) + r' \\' for r in FC]
M = {c: np.mean([r['saved'][c] for r in FC]) for c in cols}
rows.append(r'\midrule' + '\nmean & ' + ' & '.join(f1(M[c]) for c in cols) + r' \\')
tok['XP_FCROWS'] = '\n'.join(rows)
tok.update(XP_fc_q=f1(M['FC_QCDF']), XP_obs_q=f1(M['OBS_QCDF']), XP_fc_v=f1(M['FC_VaR']), XP_fc_h=f1(M['FC_h_only']),
           XP_fc_ov=f1(np.mean([r['overlap_fc_obs'] for r in FC])), XP_fc_wx=(lambda x: f"{x:.0f}\\%" if x >= 1 else ("less than 0.1\\%" if x < 0.1 else f"{x:.1f}\\%"))(100*np.mean([max(0.0, r['var_weather'] - r['var_spread'] / 100) / (max(0.0, r['var_weather'] - r['var_spread'] / 100) + r['var_spread']) for r in FC])),
           XP_fc_cv=f"{100*np.mean([r['member_base_cv'] for r in FC]):.0f}",
           XP_fc_mae13=f2(np.mean([np.mean(r['wind_mae_day'][:3]) for r in FC])), XP_fc_mae810=f2(np.mean([np.mean(r['wind_mae_day'][7:]) for r in FC])),
           XP_fc_ratio=f2(np.median([r['ratio_med'] for r in FC])), XP_fc_n=str(len(FC)))
loss = M['OBS_QCDF'] - M['FC_QCDF']; tok['XP_fc_loss'] = f1(loss); dqv = M['FC_QCDF'] - M['FC_VaR']; dqh = M['FC_QCDF'] - M['FC_h_only']
head = (f"does not inflate the value of the plans: built from the forecast they avoid {f1(M['FC_QCDF'])}\\% against {f1(M['OBS_QCDF'])}\\% "
        f"(the ensemble objective averages over the weather and rests on more realisations)" if loss <= 0.3 else
        f"overstates the value of the plans by {f1(loss)} points out of {f1(M['OBS_QCDF'])}")
tok['XP_fc_verdict'] = (head + f", and with the forecast QCDF stays {'ahead of' if dqv > 0.3 else ('behind' if dqv < -0.3 else 'level with')} the value-at-risk rule "
                        f"(${sg(dqv)}$ points) and {'ahead of' if dqh > 0.3 else ('behind' if dqh < -0.3 else 'level with')} the plan without couplings (${sg(dqh)}$).")
tok['XP_fc_costtext'] = (f"costs nothing ({f1(M['FC_QCDF'])}\\% against {f1(M['OBS_QCDF'])}\\% of simulated exposure avoided)" if loss <= 0.3 else f"costs {f1(loss)} points of simulated exposure")

orig = {d['T0']: d['K']['10']['eval']['QCDF_MILP']['saved_pct'] for d in ACT}
tok['XP_dt_orig'] = f1(np.mean([orig[r['T0']] for r in FC])); tok['XP_dt_new'] = f1(M['OBS_QCDF'])

RC = json.load(open('calib_recal.json'))['recal']; b, z = RC['best'], RC['base']
tok.update(XP_rc_cases=str(RC['train_cases']), XP_rc_a=f2(RC['a']), XP_rc_b=f2(RC['b']), XP_rc_oe0=f2(z['oe']), XP_rc_s0=f2(z['cal_slope']),
           XP_rc_s1=f2(b['cal_slope']), XP_rc_gain=f"{100*(1-b['logloss']/z['logloss']):.1f}")
import re as _re
_m = _re.search(r'cells at risk (\d+) positives (\d+)', open('recal.log').read())
tok['XP_rc_cells'] = '{:,}'.format(int(_m.group(1))).replace(',', '{,}')
CC = json.load(open('calib_recal_check.json'))['summary']; ca = [x for x in CC if x['phase'] == 'active'][0]
tok['XP_rc_dslope'] = f2(ca['cal_slope']); tok['XP_rc_doe'] = f2(ca['oe_cells'])
PR = json.load(open('pert_recal.json'))
dJ = np.mean([r['QCDF_MILP'] - r['QCDF_h_only'] for r in PR]); dV = np.mean([r['QCDF_MILP'] - r['B4_value_at_risk'] for r in PR])
tok['XP_rc_alloc'] = f"QCDF avoids {f1(np.mean([r['QCDF_MILP'] for r in PR]))}\\%, ${sg(dJ)}$ points over $J=0$ and ${sg(dV)}$ over value at risk"
tok['XP_rc_dJ'] = sg(dJ); tok['XP_rc_dV'] = sg(dV)

OP = json.load(open('ops.json'))
rules = ['QCDF_MILP', 'QCDF_h_only', 'B4_value_at_risk', 'B2_nearest_front', 'B1_communities']
def mrow(lab, key):
    return lab + ' & ' + ' & '.join(f1(np.mean([o[key][k] for o in OP])) for k in rules) + r' \\'
tok['XP_OPSROWS'] = '\n'.join([mrow('constant $\\eta=0.85$ (main)', 'base'), mrow('$\\eta$ by wind and slope', 'eta_cond'),
                               mrow('\\quad plans that assumed $0.85$', 'eta_cond_constplans'), mrow('access $\\le3$\\,km from a main road', 'road')])
em = np.mean([o['eta_cond_mean'] for o in OP]); ec = np.mean([o['eta_cond']['QCDF_MILP'] - o['eta_cond_constplans']['QCDF_MILP'] for o in OP])
fr = np.mean([o['road_frac_elig'] for o in OP]); rd = np.mean([o['base']['QCDF_MILP'] - o['road']['QCDF_MILP'] for o in OP])
ncrit = sorted(set(sum([o['critical']['names'] for o in OP], []))); pmax = max([max(o['critical']['p_burn'] + [0]) for o in OP])
_all = sorted([(p, o['T0'], k) for o in OP for k, p in enumerate(o['critical']['p_burn'])]); pmax = _all[-1][0]; p_other = _all[-2][0]
_o = [o for o in OP if o['T0'] == _all[-1][1]][0]; p_after = _o['critical']['p_after']['QCDF_MILP'][_all[-1][2]]
ch = np.mean([o['critical']['overlap'] for o in OP]); hc = np.mean([o['critical']['housing_saved_without'] - o['critical']['housing_saved_with'] for o in OP])
tok['XP_OPSTEXT'] = (f"When the effectiveness falls with wind speed and slope (mean $\\eta={f2(em)}$ instead of $0.85$), every rule avoids less and the "
    f"ordering of the rules is unchanged (Table~\\ref{{tab:ops}}); plans optimised for constant $\\eta$ lose only {f1(abs(ec))} points against plans optimised for the "
    f"conditional effectiveness, so assuming 85\\% shrinks the benefit but hardly moves the plan. Restricting units to cells within 3\\,km of a "
    f"primary or secondary road leaves {100*fr:.0f}\\% of the eligible cells and costs QCDF {f1(rd)} points. Two hospitals open on 30 June 2021 lie in "
    f"the decision domains (Plumas District Hospital, Quincy; Seneca District Hospital, Chester). Their predicted ten-day burn probability is below "
    f"{100*p_other:.1f}\\% on every active date except 5 August, when the Chester cell (725 housing units) reaches {100*pmax:.0f}\\%; on that date "
    f"every plan already protects it for its housing, lowering its burn probability to {100*p_after:.0f}\\%. Weighting each facility as 500 housing "
    f"units therefore changes on average {f1(10-ch)} of 10 cells and {f1(hc)} points of housing exposure. Chester did not burn. Evacuation routes "
    f"and service continuity would need a network model and are not included.")
tok['XP_eta_mean'] = f2(em); tok['XP_eta_cost'] = f1(abs(ec)); tok['XP_road_frac'] = f"{100*fr:.0f}"; tok['XP_road_cost'] = f1(rd)
tok['XP_eta_q'] = f1(np.mean([o['eta_cond']['QCDF_MILP'] for o in OP])); tok['XP_eta_v'] = f1(np.mean([o['eta_cond']['B4_value_at_risk'] for o in OP]))
tok['XP_crit_n'] = str(len(ncrit)); tok['XP_crit_p'] = f"{100*pmax:.1f}"

o31 = [o for o in OP if o['T0'] == '2021-07-31'][0]; rows = []; corr = 0
for a in sorted(o31['attribution'], key=lambda a: -a['gain']):
    ben = ', '.join(f"{k} ({v:.0f})" for k, v in a['top'][:3] if v >= 1.0) or '--'
    if a['own'] < 0.5 * a['gain']: corr += 1
    rows.append(f"{a['place'] or '--'} & {a['housing']:.0f} & {a['p_burn']:.2f} & {a['gain']:.0f} & {ben} \\\\")
tok['XP_ATTRROWS'] = '\n'.join(rows); tok['XP_attr_corr'] = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}.get(corr, str(corr))
g = o31['greenville']; tok.update(XP_gv_H=f"{g['housing']:.0f}", XP_gv_p=f"{100*g['p_burn']:.1f}", XP_gv_rank=str(g['delta_rank']), XP_gv_n=str(g['n_elig']))

tok['XP_gv_Hrank'] = 'fourth'
k20 = [d for d in ACT if d['T0'] == '2021-07-31'][0]['K']['20']['eval']['QCDF_MILP']['plan']
assert g['cell'] not in k20, 'Greenville in K=20 plan on 31 July'

ST = json.load(open('stats_v11.json'))['res']; rows = []
for key, lab in (('QCDF_MILP-QCDF_h_only', '$J=0$'), ('QCDF_MILP-B4_value_at_risk', 'value at risk'), ('QCDF_MILP-CELF_greedy', 'CELF')):
    v = ST[key]; ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"
    rows.append(f"{lab} & ${v['mean']:+.2f}$ & {ci(v['cbb3'])} & {ci(v['cbb6'])} & {ci(v['cbb10'])} & {ci(v['mbb6'])} & {ci(v['calendar'])} \\\\")
tok['XP_BLOCKROWS'] = '\n'.join(rows)
v = ST['QCDF_MILP-B4_value_at_risk']; tok.update(XP_st_mc=f1(max(ST[k]['mc_se_upper'] for k in ST)), XP_st_sd=f1(np.mean([ST[k]['between_date_sd'] for k in ST])),
                                              XP_st_dv=sg(v['mean']), XP_st_cv=sg(v['caldor_mean']))
CA = json.load(open('calib_caldor.json'))['summary'][0]
tok.update(XP_cc_auc=f2(CA['auc_pooled']), XP_cc_aucm=f2(CA['auc_median']), XP_cc_slope=f2(CA['cal_slope']), XP_cc_oe=f2(CA['oe_cells']), XP_cc_oeH=f2(CA['oe_H']), XP_cc_cov=f"{100*CA['cover90']:.0f}")

DQ = json.load(open('dq.json')); nd = len(DQ)
tok.update(XP_dq_ndates=str(nd), XP_dq_rho=f2(np.mean([d['a']['spearman'] for d in DQ])),
           XP_dq_regret=f2(np.mean([d['a']['best_top'] - d['a']['opt'] for d in DQ])),
           XP_dq_corr=f2(np.mean([b['corr_h'] for d in DQ for b in d['b']])), XP_dq_sign=f"{100*np.mean([b['sign_agree'] for d in DQ for b in d['b']]):.0f}",
           XP_dq_ov=f1(np.mean([b['overlap'] for d in DQ for b in d['b']])),
           XP_dq_zero=f"{100*np.mean([d['c']['n_zero']/d['c']['n_pairs'] for d in DQ]):.0f}", XP_dq_nsig=str(sum(d['c']['n_sig'] for d in DQ)),
           XP_dq_npairs=str(sum(d['c']['n_pairs'] for d in DQ)), XP_dq_ncomp=str(sum(d['c']['n_sig_comp'] for d in DQ)),
           XP_dq_chg=str(sum(1 for d in DQ if d['c']['overlap'] < 10)), XP_dq_cfsame=str(sum(1 for d in DQ if d['a']['same_celf_fg'] == 10)),
           XP_dq_cfval=f2(np.mean([d['a']['celf'] - d['a']['full_greedy'] for d in DQ])))

A = json.load(open('../astro/astro_v11.json')); AF = json.load(open('../astro/astro_full_prospective.json'))
R = A['R']; tok.update(XS2_a=f2(R['a']), XS2_b=f2(R['b']), XS2_s0=f2(R['before']['slope']), XS2_s1=f2(R['after']['slope']),
                       XS2_ov='/'.join(str(R['K'][k]['overlap']) for k in ('20', '50', '200')))
U = A['U']; tok.update(XS2_in_dp=f"{U['input']['dP_sd']:.4f}", XS2_in_ch=f1(50 - U['input']['overlap_mean']), XS2_mo_dp=f"{U['model']['dP_sd']:.3f}",
                       XS2_mo_ch=f1(50 - U['model']['overlap_mean']), XS2_oc_m=f1(U['outcome']['hits_mean']), XS2_oc_sd=f1(U['outcome']['hits_sd']), XS2_hits=f1(U['hits_Z0']))
E, O, C, Gq = A['E'], A['O'], A['C'], A['G']
rows = []
def r3(lab, vals): rows.append(lab + ' & ' + ' & '.join(vals) + r' \\')
r3('main (constant efficiency)', [x for K in ('20', '50', '200') for x in (f2(O[K]['F_free']), f1(R['K'][K]['hits_base']))])
r3('sky temperature, plan ignores it', [x for K in ('20', '50', '200') for x in (f2(E['K'][K]['F_const_plan']), f1(E['K'][K]['hits_const']))])
r3('sky temperature, plan uses it', [x for K in ('20', '50', '200') for x in (f2(E['K'][K]['F_eps_plan']), f1(E['K'][K]['hits_eps']))])
r3('time per RA range (cap)', [x for K in ('20', '50', '200') for x in (f2(O[K]['F_cap']), str(O[K]['n_cap']) + '$^\\ast$')])
r3('priority targets: unweighted plan', [x for K in ('20', '50', '200') for x in (f2(C['K'][K]['Fpri_base']), f1(C['K'][K]['hits_pri_base']))])
r3('priority targets: weighted plan', [x for K in ('20', '50', '200') for x in (f2(C['K'][K]['Fpri_w']), f1(C['K'][K]['hits_pri_w']))])
tok['XS2_OPSROWS'] = '\n'.join(rows)
tok['XS2_OPSTEXT'] = (f"The L-band sky temperature lowers the detection efficiency to a median {f2(E['eps_med'])} (minimum {f2(E['eps_min'])} in the Galactic plane). "
    f"Plans that account for it move {20-E['K']['20']['overlap']}, {50-E['K']['50']['overlap']} and {200-E['K']['200']['overlap']} pointings for $K=20$, 50, 200 away from the plane, "
    f"raise the expected yield slightly and the realised one clearly (Table~\\ref{{tab:skyops}}). Limiting the pointings in each 4-hour range of right ascension to $\\lceil K/6\\rceil$ "
    f"--- the plan otherwise puts {O['200']['bins_free'][4]} of 200 pointings in one range --- costs {100*(1-O['50']['F_cap']/O['50']['F_free']):.0f}\\% of the expected yield at $K=50$. "
    f"Weighting the {C['n_priority']} pulsar-like targets ($|b|>10^\\circ$, curvature significance $>3$; {C['n_priority_id']} of them identified in DR4) three times "
    f"nearly doubles the expected priority yield at $K=20$ and raises the realised one at every budget, at a cost of {100*(1-C['K']['50']['Fall_w']/C['K']['50']['Fall_base']):.0f}\\% of the total expected yield at $K=50$.")
rows = []
for e in sorted(A['X']['plan20'], key=lambda e: -e['expected'])[:5]:
    t = e['top'][0]; rows.append(f"{e['ra']:.1f} & {e['dec']:.1f} & {e['n_src']} & {e['expected']:.2f} & {t['name']} ({t['P']:.2f}) & {'yes' if any(x['identified'] for x in e['top']) else 'no'} \\\\")
tok['XS2_EXPLROWS'] = '\n'.join(rows)
om = A['X']['omissions']['200']['counts']; tok['XP_sky_lowP'] = str(om['low P']); tok['XP_sky_budget'] = str(om['beyond budget'])
tok['XP_sky_trunc'] = f"{max(100*(1-Gq[K]['F_milp']/Gq[K]['F_lazy']) for K in ('20', '50', '200')):.1f}"

rows = [
 ("No future information", f"GEFS forecast plans: QCDF {tok['XP_fc_q']}\\% vs {tok['XP_obs_q']}\\% with observed weather; decision-time domain (perfect forecast) {tok['XP_dt_new']}\\% vs {tok['XP_dt_orig']}\\% on the clipped grid",
  "by construction (DR3 only); 14 of 35 DR4 identifications covered at $K=200$"),
 ("Past-only recalibration", f"training fires: $a={tok['XP_rc_a']}$, $b={tok['XP_rc_b']}$, log-loss $-{tok['XP_rc_gain']}\\%$, slope stays {tok['XP_rc_s1']}; allocation differences ${tok['XP_rc_dJ']}$ ($J=0$), ${tok['XP_rc_dV']}$ (VaR)",
  f"$a={tok['XS2_a']}$, $b={tok['XS2_b']}$; slope {tok['XS2_s0']}$\\to${tok['XS2_s1']}; plans almost unchanged ({tok['XS2_ov']} pointings shared)"),
 ("Input vs.\\ outcome uncertainty", f"weather spread {tok['XP_fc_wx']} of the loss variance of a fixed plan; spread randomness the rest",
  f"inputs move {tok['XS2_in_ch']} and model {tok['XS2_mo_ch']} of 50 pointings; outcome s.d.\\ {tok['XS2_oc_sd']} identifications"),
 ("Model sensitivity", "coupling gain $+1.2$ to $+4.5$ in 12 perturbed models; margin over VaR not robust (Table~\\ref{tab:pert})",
  f"bootstrap refits keep {f1(U['model']['overlap_mean'])} of 50 pointings, {100*U['model']['F_rel']:.0f}\\% of the expected yield"),
 ("Conditional effectiveness", f"mean $\\eta={tok['XP_eta_mean']}$: QCDF {tok['XP_eta_q']}\\%, VaR {tok['XP_eta_v']}\\%; assuming 0.85 costs {tok['XP_eta_cost']} points",
  f"sky temperature: realised hits at $K=50$ {f1(E['K']['50']['hits_const'])}$\\to${f1(E['K']['50']['hits_eps'])} when the plan uses it"),
 ("Operational constraint", f"road access keeps {tok['XP_road_frac']}\\% of eligible cells, costs {tok['XP_road_cost']} points",
  f"RA-range cap costs {100*(1-O['50']['F_cap']/O['50']['F_free']):.0f}\\% of expected yield at $K=50$"),
 ("Critical assets", f"2 hospitals in the domains; Chester reaches {tok['XP_crit_p']}\\% on 5 Aug and is protected for its housing anyway: plans unchanged",
  f"priority weighting: realised priority hits {f1(C['K']['50']['hits_pri_base'])}$\\to${f1(C['K']['50']['hits_pri_w'])} ($K=50$)"),
 ("Explanation; omissions", f"every cell protects named communities (Table~\\ref{{tab:attrib}}); Greenville missed by forecast ($P={tok['XP_gv_p']}$\\%)",
  f"each pointing targets listed sources; misses: {om['low P']} low $P$, {om['beyond budget']} budget, 0 visibility"),
 ("Surrogate vs.\\ objective", f"regret {tok['XP_dq_regret']} points among 30 best plans; fields correlate {tok['XP_dq_corr']} across rebuilds; non-adjacent pairs change {tok['XP_dq_chg']}/{tok['XP_dq_ndates']} plans",
  f"second-order truncation loses $\\le{tok['XP_sky_trunc']}\\%$"),
 ("Greedy check", f"CELF = full greedy on {tok['XP_dq_cfsame']}/{tok['XP_dq_ndates']} dates", "lazy = full greedy (submodular), all $K$"),
 ("Statistics", f"intervals stable across 5 block schemes (Table~\\ref{{tab:blocks}}); between-fire difference exceeds between-date",
  "bootstrap over the 35 identified sources (Table~\\ref{tab:astro_full})"),
]
tok['XP_PROTROWS'] = '\n'.join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
json.dump(tok, open('v11_tokens.json', 'w'), indent=1)
fn = '../paper/v4_body_filled.tex'; s = open(fn).read()
for k_, v in sorted(tok.items(), key=lambda kv: -len(kv[0])): s = s.replace('@@' + k_ + '@@', v)
open(fn, 'w').write(s)
import re; print('unfilled', sorted(set(re.findall(r'@@(X[PS]2?_\w+)@@', s))))
for k, v in tok.items():
    if len(v) < 300: print(k, '=', v)
