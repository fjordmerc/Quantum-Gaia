#!/usr/bin/env python3
"""
fetch_gaia.py – resilient Gaia download
Usage: python fetch_gaia.py  /path/out.csv
"""

import time, sys, pathlib, pandas as pd
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from requests.exceptions import HTTPError

OUT = pathlib.Path(sys.argv[1]).expanduser()
OUT.parent.mkdir(parents=True, exist_ok=True)

# query parameters
CEN = SkyCoord(ra=56.75*u.deg, dec=24.12*u.deg, frame='icrs')
RADIUS = 2.0*u.deg
LIMIT = 10000

def do_query():
    q = f"""
        SELECT TOP {LIMIT} source_id, ra, dec
        FROM gaiaedr3.gaia_source
        WHERE CONTAINS(
             POINT('ICRS', ra, dec),
             CIRCLE('ICRS', {CEN.ra.degree},{CEN.dec.degree},{RADIUS.to(u.deg).value})
        )=1
    """
    return Gaia.launch_job_async(q).get_results()      # async but streaming

# retry logic
for n in range(3):
    try:
        tbl = do_query()
        break
    except HTTPError as e:
        print(f"Server error, retry {n+1}/3 …")
        if n==2:
            raise
        time.sleep(5)

tbl.to_pandas().to_csv(OUT, index=False)
print(f"✓ saved {len(tbl)} rows to {OUT}")
