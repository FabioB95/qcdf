"""Fetch archived GEFS v12 10 m wind (0.5 deg, pgrb2ap5) from the NOAA open-data bucket on AWS."""
import sys, os, re, io, numpy as np, pandas as pd, requests, pygrib, tempfile
from concurrent.futures import ThreadPoolExecutor
B = 'https://noaa-gefs-pds.s3.amazonaws.com'
DATES = ['2021-07-24', '2021-07-25', '2021-07-27', '2021-07-28', '2021-07-29', '2021-07-30', '2021-07-31', '2021-08-01', '2021-08-04', '2021-08-05']
MEMB = ['gec00'] + [f'gep{k:02d}' for k in range(1, 31)]
LEADS = list(range(6, 241, 6))
BOX = dict(lat1=38.5, lat2=42.0, lon1=236.5, lon2=241.5)
S = requests.Session()

def fetch(day, cyc, mem, lead):
    url = f"{B}/gefs.{day:%Y%m%d}/{cyc:02d}/atmos/pgrb2ap5/{mem}.t{cyc:02d}z.pgrb2a.0p50.f{lead:03d}"
    for a in range(6):
        try:
            idx = S.get(url + '.idx', timeout=60); idx.raise_for_status(); L = idx.text.strip().split('\n')
            k = [i for i, l in enumerate(L) if ':UGRD:10 m above ground:' in l][0]
            st = int(L[k].split(':')[1]); en = int(L[k + 2].split(':')[1]) - 1
            r = S.get(url, headers={'Range': f'bytes={st}-{en}'}, timeout=60); r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.grb') as f:
                f.write(r.content); f.flush(); g = pygrib.open(f.name); out = {}
                for m in g:
                    d, la, lo = m.data(**BOX); out[m.shortName] = d.astype(np.float32)
                g.close()
            return out['10u'], out['10v'], la, lo
        except Exception as ex:
            err = ex
    raise RuntimeError(f'{url}: {err}')

if __name__ == '__main__':
    with ThreadPoolExecutor(12) as ex:
        jobs = [(d, m, l) for d in DATES for m in MEMB for l in LEADS]
        res = list(ex.map(lambda j: fetch(pd.Timestamp(j[0]), 0, j[1], j[2]), jobs))
        la, lo = res[0][2], res[0][3]
        U = np.array([r[0] for r in res]).reshape(len(DATES), len(MEMB), len(LEADS), *la.shape)
        V = np.array([r[1] for r in res]).reshape(len(DATES), len(MEMB), len(LEADS), *la.shape)
        np.savez('gefs_fc.npz', u=U, v=V, lat=la, lon=lo, dates=np.array(DATES), members=np.array(MEMB), leads=np.array(LEADS))
        print('forecasts', U.shape, flush=True)
        times = pd.date_range('2021-07-10', '2021-08-05 18:00', freq='6h')
        res = list(ex.map(lambda t: fetch(t.normalize(), t.hour, 'gec00', 0), times))
        np.savez('gefs_an.npz', u=np.array([r[0] for r in res]), v=np.array([r[1] for r in res]), lat=la, lon=lo, times=times.astype(str).values)
        print('analyses', len(times), flush=True)
