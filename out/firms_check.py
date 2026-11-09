"""firms_check.py — Post-hoc check of the Dixie Fire QCDF decision vs NASA FIRMS ================================================================================ Static-allocator validation (paper, Sec. Domain Independence / Dixie Fire): does the triplet chosen at T0 (13 July 2021) coincide with where the fire actually burned afterwards?"""
import argparse
import itertools

import pandas as pd

try:
    import h3
    _to_cell = getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3")
except ImportError:
    raise SystemExit("pip install h3")

T0 = "2021-07-13"
QCDF_PREFIXES = ["852814b7", "852814b3", "8528334b"]
ALL_PREFIXES = ["852814b7", "852814a3", "8528334f", "852814b3", "8528334b", "85281487"]
RES = 5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("firms_csv")
    ap.add_argument("--cells", help="CSV with column 'h3' (full indices of the 6 cells)")
    ap.add_argument("--t0", default=T0)
    ap.add_argument("--t_end", default="2021-09-30")
    a = ap.parse_args()

    df = pd.read_csv(a.firms_csv)
    df = df[(df.acq_date > a.t0) & (df.acq_date <= a.t_end)].copy()
    df["h3"] = [_to_cell(la, lo, RES) for la, lo in zip(df.latitude, df.longitude)]

    if a.cells:
        cells = pd.read_csv(a.cells)["h3"].tolist()
    else:
        uniq = df.h3.unique()
        cells = []
        for p in ALL_PREFIXES:
            m = [c for c in uniq if c.startswith(p)]
            cells.append(m[0] if m else p)
    stats = (df[df.h3.isin(cells)].groupby("h3")
             .agg(n_det=("frp", "size"), frp_sum=("frp", "sum"))
             .reindex(cells).fillna(0))
    stats.index = [c[:8] for c in stats.index]
    print(f"Post-T0 activity per pilot cell ({a.t0} < date <= {a.t_end}):")
    print(stats.to_string(), "\n")

    rows = []
    for trip in itertools.combinations(stats.index, 3):
        rows.append(dict(triplet=trip,
                         n_det=stats.loc[list(trip), "n_det"].sum(),
                         frp=stats.loc[list(trip), "frp_sum"].sum()))
    tr = pd.DataFrame(rows)
    for col in ["n_det", "frp"]:
        tr[f"rank_{col}"] = tr[col].rank(ascending=False, method="min").astype(int)
    q = tr[tr.triplet.apply(lambda t: set(t) == set(QCDF_PREFIXES))].iloc[0]
    tot_n, tot_f = stats.n_det.sum(), stats.frp_sum.sum()
    print(f"QCDF triplet {QCDF_PREFIXES}")
    print(f"  detections: {int(q.n_det)}/{int(tot_n)} = {q.n_det / max(tot_n, 1):.1%}"
          f"   rank {q.rank_n_det}/20")
    print(f"  FRP [MW]:   {q.frp:.0f}/{tot_f:.0f} = {q.frp / max(tot_f, 1e-9):.1%}"
          f"   rank {q.rank_frp}/20")
    print("  (random triplet expectation: 50% of activity, mean rank 10.5)")
    stats.to_csv("firms_cells.csv"); tr.to_csv("firms_triplets.csv", index=False)
    print("saved firms_cells.csv, firms_triplets.csv")

if __name__ == "__main__":
    main()
