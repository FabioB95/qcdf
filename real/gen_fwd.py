"""Tokens for the forward (non-anchored) closed-loop study (fwd_a.json, fwd_b.json)."""
import json, numpy as np
R = []
for f in ('fwd_a.json', 'fwd_b.json'): R += json.load(open(f))['runs']
ks = ['OL_QCDF', 'OL_VaR', 'CL_QCDF', 'CL_h_only', 'CL_VaR']
none = np.array([r['none'] for r in R]); X = {k: np.array([r[k] for r in R]) for k in ks}
pct = {k: 100 * (none.sum() - X[k].sum()) / none.sum() for k in ks}
rng = np.random.default_rng(3); n = len(R); B = 10000
def ci(a, b):
    idx = rng.integers(n, size=(B, n)); N = none[idx].sum(1)
    d = 100 * ((X[b][idx].sum(1)) - (X[a][idx].sum(1))) / N
    return np.percentile(d, [2.5, 97.5])
fmt = lambda c: f"[{c[0]:+.1f},\\,{c[1]:+.1f}]"
tok = dict(XW_n=str(n), XW_none=f"{none.mean():.0f}", **{f"XW_{k}": f"{pct[k]:.1f}" for k in ks})
tok['XW_ratio'] = f"{pct['CL_QCDF'] / pct['OL_QCDF']:.2f}"
tok['XW_gainpts'] = f"{pct['CL_QCDF'] - pct['OL_QCDF']:.0f}"
tok['XW_ci_loop'] = fmt(ci('CL_QCDF', 'OL_QCDF')); tok['XW_ci_var'] = fmt(ci('CL_QCDF', 'CL_VaR')); tok['XW_ci_h'] = fmt(ci('CL_QCDF', 'CL_h_only'))
tok['XW_ci_olvar'] = fmt(ci('OL_QCDF', 'OL_VaR'))
tok['XW_d_var'] = f"{pct['CL_QCDF'] - pct['CL_VaR']:+.1f}"; tok['XW_d_h'] = f"{pct['CL_QCDF'] - pct['CL_h_only']:+.1f}"
tok['XW_d_olvar'] = f"{pct['OL_QCDF'] - pct['OL_VaR']:+.1f}"
json.dump(tok, open('fwd_tokens.json', 'w'), indent=1)
fn = '../paper/v4_body_filled.tex'; s = open(fn).read()
for k_, v in sorted(tok.items(), key=lambda kv: -len(kv[0])): s = s.replace('@@' + k_ + '@@', v)
open(fn, 'w').write(s)
print(tok)
