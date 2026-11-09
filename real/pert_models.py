"""Perturbed prospective spread models: (a) 10 fire-cluster bootstrap refits on the 2019-2020 GOFER fires; (b) calibration-in-the-large variants: intercept shifted so that the expected number of burned cells is x0.5 / x2."""
import json, numpy as np, os
fsrc = open('multifire_fit.py').read(); exec(fsrc[:fsrc.index('MODELS = [')])
os.makedirs('pert', exist_ok=True)
train = np.where(summ.year.values <= 2020)[0]; rng = np.random.default_rng(11); files = []
for b in range(10):
    sf = rng.choice(train, len(train), replace=True); u, c = np.unique(sf, return_counts=True)
    o = fit_eval('WS', u, mult=dict(zip(u, c)))
    json.dump(dict(models={'wind+slope': dict(use='WS', beta=[o['b0'], o['beta']['W'], o['beta']['S']])}, norm=o['norm']), open(f'pert/boot{b}.json', 'w'))
    files.append(f'pert/boot{b}.json'); print(b, round(o['b0'], 3), {k: round(v, 3) for k, v in o['beta'].items()})
C = json.load(open('calib_prospective.json'))
for nm, sh in (('shift_m', -0.7), ('shift_p', +0.7)):
    C2 = json.loads(json.dumps(C)); C2['models']['wind+slope']['beta'][0] += sh
    json.dump(C2, open(f'pert/{nm}.json', 'w')); files.append(f'pert/{nm}.json')
open('pert/files.txt', 'w').write(','.join(files))
