"""Prospective spread model: fitted only on the GOFER fires of 2019-2020 (no Dixie, no information after any 2021 decision date). Also the rolling-origin Dixie calibration: at each decision date T0, MAP fit on the NIFC Dixie transitions ending at or before T0 with a Gaussian prior centred on the prospective model. Writes calib_prospective.json and calib_rolling.json (same layout as calib.json)."""
import json, numpy as np, pandas as pd, pickle, os
from scipy.optimize import minimize
fsrc = open('multifire_fit.py').read(); exec(fsrc[:fsrc.index('MODELS = [')])
yr = summ.year.values
train = np.where(yr <= 2020)[0]
out = {'models': {}, 'DIST': 2.4, 'fires': [summ.fire.iloc[i] for i in train]}
for name, use in [('wind+slope', 'WS'), ('constant', '')]:
    o = fit_eval(use, train)
    out['models'][name] = dict(use=use, beta=[o['b0']] + [o['beta'][c] for c in use], n=o['n'])
    if use: out['norm'] = {k: v for k, v in o['norm'].items()}
print('prospective', out['models'], out['norm'], len(train), 'fires')
json.dump(out, open('calib_prospective.json', 'w'), indent=1)

if not os.path.exists('nifc_rows.pkl'):
    src = open('calibrate.py').read(); exec(src[:src.index("print('transitions'")])
    pickle.dump([dict(d1=r['d1'], Delta=r['Delta'], y=r['y'], feats=r['feats']) for r in rows], open('nifc_rows.pkl', 'wb'))
rows = pickle.load(open('nifc_rows.pkl', 'rb'))
nm = out['norm']; bp = np.array(out['models']['wind+slope']['beta'])
def X(r): return np.c_[np.ones(len(r['feats'])), (r['feats'][:, 0] - nm['W'][0]) / nm['W'][1], (r['feats'][:, 1] - nm['S'][0]) / nm['S'][1]]
def nlp(b, data, sd):
    ll = 0.0
    for r in data:
        q = 1 / (1 + np.exp(-X(r) @ b)); p = np.clip(1 - np.prod((1 - q) ** r['Delta']), 1e-9, 1 - 1e-9)
        ll += r['y'] * np.log(p) + (1 - r['y']) * np.log(1 - p)
    return -ll + np.sum((b - bp) ** 2) / (2 * sd ** 2)
dates = sorted(set(json.load(open('real_a.json'))[i]['T0'] for i in range(15)) | set(r['T0'] for r in json.load(open('real_b.json'))))
roll = {}
for d in dates:
    data = [r for r in rows if r['d1'] <= pd.Timestamp(d)]
    b = minimize(nlp, bp, args=(data, 0.5), method='L-BFGS-B').x if data else bp
    roll[d] = dict(beta=b.tolist(), n=len(data), pos=int(sum(r['y'] for r in data)))
    print('rolling', d, len(data), np.round(b, 3), flush=True)
json.dump(dict(norm=nm, DIST=2.4, prior=bp.tolist(), prior_sd=0.5, dates=roll), open('calib_rolling.json', 'w'), indent=1)
