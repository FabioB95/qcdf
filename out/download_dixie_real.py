"""download_dixie_real.py — Real data for the Dixie Fire (2021) QCDF case study ================================================================================ Run on your own computer (needs internet): pip install requests python download_dixie_real.py Everything is saved in ./dixie_real/ -> zip that folder and send it to Claude."""
import datetime as dt
import json
import os
import sys
import time

import requests

MAP_KEY = "INSERISCI_LA_TUA_MAP_KEY"
OUT = "dixie_real"
BBOX = (-122.2, 39.3, -120.4, 40.9)
COUNTIES = ["007", "035", "063", "103"]
os.makedirs(OUT, exist_ok=True)

def get(url, path, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=300, **kw)
            r.raise_for_status()
            open(path, "wb").write(r.content)
            print(f"  ok  {path}  ({len(r.content)/1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"  retry {attempt+1}: {e}")
            time.sleep(5 * (attempt + 1))
    print(f"  FAILED {url}")
    return False

print("[1/5] Perimetri NIFC (Zenodo)")
get("https://zenodo.org/records/15492982/files/DixieFire_2021_perimeters.zip?download=1",
    f"{OUT}/DixieFire_2021_perimeters.zip")

print("[2/5] VIIRS S-NPP active fires (FIRMS, standard processing)")
if MAP_KEY.startswith("INSERISCI"):
    print("  !! metti la MAP_KEY in cima al file e rilancia (gli altri download proseguono)")
else:
    area = ",".join(str(x) for x in BBOX)
    d, end, rows, header = dt.date(2021, 7, 13), dt.date(2021, 9, 30), [], None
    while d <= end:
        n = min(5, (end - d).days + 1)
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_SP/{area}/{n}/{d}"
        try:
            txt = requests.get(url, timeout=120).text.strip().splitlines()
            if txt and txt[0].startswith("latitude"):
                header = header or txt[0]; rows += txt[1:]
            else:
                print(f"  {d}: {txt[:1]}")
        except Exception as e:
            print(f"  {d}: {e}")
        d += dt.timedelta(days=n); time.sleep(1)
    if header:
        open(f"{OUT}/viirs_snpp_sp_dixie_2021.csv", "w").write(header + "\n" + "\n".join(rows) + "\n")
        print(f"  ok  {len(rows)} detections")

print("[3/5] Census 2020: tract boundaries + population")
get("https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_06_tract.zip", f"{OUT}/tl_2020_06_tract.zip")
api = ("https://api.census.gov/data/2020/dec/pl?get=P1_001N,NAME&for=tract:*"
       f"&in=state:06&in=county:{','.join(COUNTIES)}")
get(api, f"{OUT}/census2020_P1_tracts.json")

print("[4/5] OSM: hospitals, schools, fire stations, nursing homes")
s, w, n_, e = BBOX[1], BBOX[0], BBOX[3], BBOX[2]
q_poi = f"""[out:json][timeout:180];
(
  nwr["amenity"="hospital"]({s},{w},{n_},{e});
  nwr["amenity"="school"]({s},{w},{n_},{e});
  nwr["amenity"="fire_station"]({s},{w},{n_},{e});
  nwr["social_facility"="nursing_home"]({s},{w},{n_},{e});
);
out center tags;"""
q_roads = f"""[out:json][timeout:300];
way["highway"~"^(motorway|trunk|primary|secondary)$"]({s},{w},{n_},{e});
out geom tags;"""
for name, q in [("osm_pois", q_poi), ("osm_roads", q_roads)]:
    for ep in ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]:
        try:
            r = requests.post(ep, data={"data": q}, timeout=400); r.raise_for_status()
            json.loads(r.text)
            open(f"{OUT}/{name}.json", "w").write(r.text)
            print(f"  ok  {name}.json ({len(r.text)/1e6:.1f} MB)"); break
        except Exception as ex:
            print(f"  {ep}: {ex}")

print("[5/5] Riepilogo")
for f in sorted(os.listdir(OUT)):
    print(f"  {f:40s} {os.path.getsize(os.path.join(OUT, f))/1e6:8.2f} MB")
print(f"\nFatto. Comprimi la cartella '{OUT}' e mandala.")
