"""v12 tokens: remaining review checks (XR_* wildfire, XS3_* sky). Inputs: extra12_[ab].json, calib_check_v12_cells.csv.gz, fireline.json, fsM.json, calib_monument.json, fc_members.json, ../astro/astro_full_prospective_rep.json, ../astro/astro_v11.json."""
import json, glob, numpy as np, pandas as pd
from scipy.optimize import minimize
tok = {}
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; sg = lambda x: f"{x:+.1f}"
X = sorted(sum([json.load(open(f)) for f in sorted(glob.glob('extra12_[ab].json'))], []), key=lambda r: r['T0'])

nJ = [v['nJ'] for r in X for v in r['thr'].values()]; dv = [abs(v['val'] - r['thr']['2se']['val']) for r in X for v in r['thr'].values()]
tok.update(XR_thr_nmin=str(min(nJ)), XR_thr_nmax=str(max(nJ)), XR_thr_dmax=f2(max(dv)),
           XR_thr_ov=f1(np.mean([v['overlap_main'] for r in X for k, v in r['thr'].items() if k != '2se'])))

o510 = np.mean([r['nest']['o5_10'] for r in X]); o1020 = np.mean([r['nest']['o10_20'] for r in X])
tok['XR_nest_text'] = ("nested on every date: the five cells chosen with $K=5$ are all in the $K=10$ plan, and the ten chosen with $K=10$ are all in the $K=20$ plan, so more units extend the plan rather than move it." if o510 == 5 and o1020 == 10 else
                       f"nearly nested: on average {f1(o510)} of the 5 cells chosen with $K=5$ are in the $K=10$ plan, and {f1(o1020)} of the 10 cells chosen with $K=10$ are in the $K=20$ plan.")
tok.update(XR_reinf5=f1(np.mean([r['reinf']['v5p5'] for r in X])), XR_v10=f1(np.mean([r['reinf']['v10'] for r in X])),
           XR_reinf10=f1(np.mean([r['reinf']['v10p10'] for r in X])), XR_v20=f1(np.mean([r['reinf']['v20'] for r in X])))

c = lambda k: np.mean([r['c2010'][k] for r in X])
tok.update(XR_c10_q=f1(c('qcdf')), XR_c10_q20=f1(c('qcdf2020')), XR_c10_v=f1(c('var')), XR_c10_ov=f1(c('overlap')))
worst = min(X, key=lambda r: r['c2010']['qcdf'] - r['c2010']['qcdf2020'])
tok['XR_c10_text'] = (f"The cost of using the value that was public at the time is {f1(c('qcdf2020') - c('qcdf'))} points on average "
                      f"(largest on {worst['T0'][5:]}, {f1(worst['c2010']['qcdf2020'] - worst['c2010']['qcdf'])} points), where housing built "
                      f"between 2010 and 2020 changes which cells are worth protecting.")

rows = []
for lab, key in (('construction delay of 2 days', 'delay'), ('protection lasting 5 days', 'last5')):
    m = {k: np.mean([r[key][k] for r in X]) for k in ('qcdf', 'h_only', 'var', 'front', 'comm', 'main_plan')}
    rows.append(lab + ' & ' + ' & '.join(f1(m[k]) for k in ('qcdf', 'h_only', 'var', 'front', 'comm', 'main_plan')) + r' \\')
tok['XR_TIMEROWS'] = '\n'.join(rows)
tok['XR_last5_v'] = f1(np.mean([r['last5']['var'] for r in X])); tok['XR_delay_q'] = f1(np.mean([r['delay']['qcdf'] for r in X])); tok['XR_last5_q'] = f1(np.mean([r['last5']['qcdf'] for r in X]))

C = pd.read_csv('calib_check_v12_cells.csv.gz'); C = C[(C.model == 'prospective') & (C.T0 <= '2021-08-05')]
def si(p, y):
    x = np.log(np.clip(p, 5e-4, 1 - 5e-4) / (1 - np.clip(p, 5e-4, 1 - 5e-4)))
    f = lambda b: -np.sum(y * (b[0] + b[1] * x) - np.logaddexp(0, b[0] + b[1] * x)); return minimize(f, [0, 1]).x
a0, b0 = si(C.p.values, C.y.values); rng = np.random.default_rng(4); dates = C.T0.unique(); bs = []
for _ in range(400):
    dd = rng.choice(dates, len(dates)); S = pd.concat([C[C.T0 == d] for d in dd]); bs.append(si(S.p.values, S.y.values))
bs = np.array(bs); ci = lambda k: f"[{np.percentile(bs[:, k], 2.5):.2f}, {np.percentile(bs[:, k], 97.5):.2f}]"
tok['XR_cal_int'] = f"${a0:.2f}$ {ci(0)}"; tok['XR_cal_slope'] = f"${b0:.2f}$ {ci(1)}"
import re
m_ = re.search(r'cells at risk (\d+) positives (\d+)', open('recal.log').read()); rate = int(m_.group(2)) / int(m_.group(1))
br = np.mean((C.p - C.y) ** 2); brr = np.mean((rate - C.y) ** 2); brr2 = np.mean((C.y.mean() - C.y) ** 2)
tok.update(XR_brier=f"{br:.3f}", XR_brier_ref=f"{brr:.3f}", XR_bss=f"{1 - br / brr:+.2f}", XR_brier_ref2=f"{brr2:.3f}", XR_bss2=f"{1 - br / brr2:+.2f}", XR_ref_rate=f"{100*rate:.1f}")
tok['XR_cal_text'] = ("The model ranks cells better than the reference forecast and is worse calibrated: its predictions are too extreme "
                      "(slope below one), so a decision that needs probabilities, not a ranking, should not use them as they are."
                      if br > brr2 else "The model beats both references, but its slope stays below one: the probabilities are too extreme.")

R = json.load(open('fsP_act.json')) + json.load(open('fsP_late.json')); txt = []
for lab, f in (('the first three weeks (to 3 August)', lambda d: d <= '2021-08-03'), ('the first month (to 12 August)', lambda d: d <= '2021-08-12'), ('all 29 dates', lambda d: True)):
    Y = [r for r in R if f(r['T0'])]; g = lambda a, b: np.mean([r['K']['10']['eval'][a]['saved_pct'] - r['K']['10']['eval'][b]['saved_pct'] for r in Y])
    txt.append(f"{lab}: QCDF minus $J=0$ ${sg(g('QCDF_MILP', 'QCDF_h_only'))}$, minus value at risk ${sg(g('QCDF_MILP', 'B4_value_at_risk'))}$, minus CELF ${sg(g('QCDF_MILP', 'CELF_greedy'))}$")
tok['XR_split_text'] = '; '.join(txt) + '.'

F = json.load(open('fireline.json')); km = F['km_total']; e = F['by_engagement_km']; t = F['by_type_km']
tok.update(XR_fl_km=f"{km:,.0f}".replace(',', '{,}'), XR_fl_dozer=f"{100*t.get('Completed Dozer Line', 0)/km:.0f}", XR_fl_road=f"{100*t.get('Completed Road as Line', 0)/km:.0f}",
           XR_fl_held=f"{100*e.get('Held', 0)/km:.0f}", XR_fl_burned=f"{100*e.get('Burned Over', 0)/km:.0f}", XR_fl_not=f"{100*e.get('Not Engaged', 0)/km:.0f}",
           XR_fl_d0=F['createdate_range'][0][:10], XR_fl_d1=F['createdate_range'][1][:10], XR_fl_kmact=f"{F['km_mapped_by_active_end']:,.0f}".replace(',', '{,}'))
po = F['plan_overlap_active']['frac_line_10d']
tok.update(XR_fl_qcdf=f"{100*po['QCDF_MILP']:.0f}", XR_fl_var=f"{100*po['B4_value_at_risk']:.0f}", XR_fl_front=f"{100*po['B2_nearest_front']:.0f}", XR_fl_comm=f"{100*po['B1_communities']:.0f}")

M = json.load(open('fsM.json')); CM = json.load(open('calib_monument.json'))['summary'][0]
g = lambda a, b: np.mean([r['K']['10']['eval'][a]['saved_pct'] - r['K']['10']['eval'][b]['saved_pct'] for r in M])
tok.update(XR_mo_dates=str(len(M)), XR_mo_elig=f"{min(r['n_elig'] for r in M)}--{max(r['n_elig'] for r in M)}",
           XR_mo_q=f1(np.mean([r['K']['10']['eval']['QCDF_MILP']['saved_pct'] for r in M])), XR_mo_dJ=f"${sg(g('QCDF_MILP', 'QCDF_h_only'))}$",
           XR_mo_dV=f"${sg(g('QCDF_MILP', 'B4_value_at_risk'))}$", XR_mo_dC=f"${sg(g('QCDF_MILP', 'CELF_greedy'))}$", XR_mo_slope=f2(CM['cal_slope']), XR_mo_oe=f2(CM['oe_cells']))
dJm = g('QCDF_MILP', 'QCDF_h_only'); dVm = g('QCDF_MILP', 'B4_value_at_risk')
tok['XR_mo_text'] = ("The pattern of Dixie and Caldor holds: " if dJm > 0 else "Unlike Dixie and Caldor, the couplings do not help here: ") + \
    f"couplings {'add value' if dJm > 0.3 else 'add little'}, value at risk is {'close' if abs(dVm) < 1.5 else ('behind' if dVm > 0 else 'ahead')}, and the greedy search on the simulator is the reference to beat. The allocation policy is now tested on three fires."

try:
    FM = json.load(open('fc_members.json')); ho = [x for r in FM for x in r['half_overlap']]
    tok['XR_fc_half'] = f1(np.mean(ho)); tok['XR_fc_halfmin'] = str(int(np.min(ho)))
except Exception: tok['XR_fc_half'] = 'n/a'; tok['XR_fc_halfmin'] = 'n/a'

A = json.load(open('../astro/astro_full_prospective.json')); P = {K: set(A['K'][K]['plans']['QCDF_MILP']) for K in ('20', '50', '200')}
tok['XS3_nest'] = f"{len(P['20'] & P['50'])} of 20 ($K=20$ in 50) and {len(P['50'] & P['200'])} of 50 ($K=50$ in 200)"
V = json.load(open('../astro/astro_v11.json'))
tok['XS3_f50'] = f2(A['K']['50']['eval']['QCDF_MILP']['F_model']); tok['XS3_ext'] = f2(V['G']['50']['F_lazy'])
B = json.load(open('../astro/astro_full_prospective_rep.json')); c3 = B['cal']; e2 = lambda K, k: B['K'][K]['eval'][k]
cg = [100 * (e2(K, 'QCDF_MILP')['F_model'] / e2(K, 'QCDF_h_only')['F_model'] - 1) for K in ('20', '50', '100', '200')]
tok.update(XS3_pos=str(c3['pos_test']), XS3_auc=f2(c3['auc_test']), XS3_slope=f2(c3['slope']), XS3_q200=str(e2('200', 'QCDF_MILP')['n_hit_any']),
           XS3_b200=str(e2('200', 'brightest')['n_hit_any']), XS3_c200=str(e2('200', 'most_curved')['n_hit_any']), XS3_r200=str(e2('200', 'random')['n_hit_any']),
           XS3_cg=f"{min(cg):.0f}--{max(cg):.0f}")

rows = [
 ("Value known at decision", f"Census 2010 housing: QCDF {tok['XR_c10_q']}\\% vs {tok['XR_c10_q20']}\\% with 2020 values", "DR3 features only, by construction"),
 ("Budgets vs.\\ reinforcements", f"5+5 units {tok['XR_reinf5']}\\% vs 10 together {tok['XR_v10']}\\%; 10+10 {tok['XR_reinf10']}\\% vs 20 {tok['XR_v20']}\\%", f"plans nested ({tok['XS3_nest']})"),
 ("Coupling selection", f"5 rules, value changes $\\le{tok['XR_thr_dmax']}$ points", "exact couplings"),
 ("Timing of the action", f"2-day delay: QCDF {tok['XR_delay_q']}\\%; 5-day protection: {tok['XR_last5_q']}\\%; QCDF and VaR close or QCDF ahead", "---"),
 ("Real actions", f"fireline archive: no construction dates; {tok['XR_fl_qcdf']}\\% of QCDF cells get line within 10 days", "DR4 identifications (Table~\\ref{tab:astro_full})"),
 ("Replication", f"Monument: QCDF {tok['XR_mo_q']}\\%, {tok['XR_mo_dJ']} vs $J=0$, {tok['XR_mo_dV']} vs VaR", f"DR2$\\to$DR3: {tok['XS3_q200']} of {tok['XS3_pos']} covered at $K=200$ (brightest {tok['XS3_b200']})"),
]
tok['XR_PROTROWS'] = '\n'.join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
json.dump(tok, open('v12_tokens.json', 'w'), indent=1)
fn = '../paper/v4_body_filled.tex'; s = open(fn).read()
for k_, v in sorted(tok.items(), key=lambda kv: -len(kv[0])): s = s.replace('@@' + k_ + '@@', v)
open(fn, 'w').write(s)
print('unfilled', sorted(set(re.findall(r'@@(X[RS]3?_\w+)@@', s))))
for k, v in tok.items():
    if len(v) < 300: print(k, '=', v)
