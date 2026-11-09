"""Out-of-fire transfer: models fitted on the 27 GOFER fires other than Dixie, evaluated on the NIFC Dixie transitions of calibrate.py (all 6,624 and the 4,292 test transitions after 23 Aug), against the Dixie-only NIFC models of calib.json."""
import json, numpy as np, pandas as pd
src = open('calibrate.py').read(); exec(src[:src.index("print('transitions'")])
for k in set(r['k'] for r in rows):
    rr = [r for r in rows if r['k'] == k]; m = np.mean([-r['fm'] for r in rr]); me = np.mean([r['erc'] for r in rr])
    for r in rr: r['dsp'] = -r['fm'] - m; r['esp'] = r['erc'] - me
fsrc = open('multifire_fit.py').read(); exec(fsrc[:fsrc.index('MODELS = [')])
dixie = int(np.where(summ.fire.values == 'Dixie')[0][0]); others = np.setdiff1d(np.arange(nf), [dixie])
def pb(o, r, use):
    nm = o['norm']; b = o['beta']
    eta = np.full(len(r['feats']), o['b0'])
    for c in use:
        if c == 'W': x = r['feats'][:, 0]
        elif c == 'S': x = r['feats'][:, 1]
        elif c == 'D': x = np.full(len(r['feats']), -r['fm'])
        elif c == 'E': x = np.full(len(r['feats']), r['erc'])
        elif c == 'Q': x = np.full(len(r['feats']), r['dsp'])
        elif c == 'R': x = np.full(len(r['feats']), r['esp'])
        eta = eta + b[c] * (x - nm[c][0]) / nm[c][1]
    q = 1 / (1 + np.exp(-eta)); return 1 - np.prod((1 - q) ** r['Delta'])
def auc(p, yv):
    s = pd.Series(p).rank().values; n1 = yv.sum(); n0 = len(yv) - n1
    return float((s[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
test = [r for r in rows if r['d1'] > pd.Timestamp('2021-08-23')]
out = {}
for name, use in [('constant', ''), ('wind+slope', 'WS'), ('+ dryness', 'WSD'), ('+ ERC', 'WSE'), ('+ dryness + ERC', 'WSDE'), ('+ relative dryness', 'WSQ')]:
    o = fit_eval(use, others)
    res = {}
    for lab, data in (('all', rows), ('test', test)):
        p = np.clip(np.array([pb(o, r, use) for r in data]), 1e-9, 1 - 1e-9); yv = np.array([r['y'] for r in data])
        res[lab] = dict(ll=float(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p))), auc=auc(p, yv), pred=float(p.sum()), obs=int(yv.sum()))
    out[name] = dict(use=use, beta=o['beta'], b0=o['b0'], norm=o['norm'], **res)
    print(name, {k: round(v, 3) for k, v in o['beta'].items()}, 'all', {k: round(v, 3) for k, v in res['all'].items()}, 'test', {k: round(v, 3) for k, v in res['test'].items()}, flush=True)
json.dump(out, open('transfer.json', 'w'), indent=1)
