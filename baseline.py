#!/usr/bin/env python3
"""
baseline.py – brute-force angular-distance filter with progress bar.

Usage:
    python baseline.py  ../data/pleiades.csv  10
"""
import sys, time, pathlib                     # stdlib utilities
import numpy as np                            # vector math
import pandas as pd                           # CSV ingest
from tqdm import tqdm                         # live progress bar
from sys import stderr                        # direct flush channel

# ---------- CLI ----------
if len(sys.argv) != 3:                        # arg check
    sys.exit("Usage: baseline.py <csv> <radius_arcsec>")
csv_file  = pathlib.Path(sys.argv[1]).expanduser()
radius_as = float(sys.argv[2])                # neighbour cut-off

# ---------- load RA, Dec ----------
ra, dec = pd.read_csv(csv_file,
                      usecols=['ra', 'dec']).values.T          # two columns
n_stars = len(ra)
rad_lim = np.deg2rad(radius_as / 3600.0)                       # arcsec→rad

# ---------- brute-force loop ----------
pair_cnt = 0
t0 = time.perf_counter()
for i in tqdm(range(n_stars), desc="Classical"):               # ETA update
    dra  = np.deg2rad(ra[i] - ra[i+1:])
    ddec = np.deg2rad(dec[i] - dec[i+1:])
    sin2 = (np.sin(ddec/2)**2 +
            np.cos(np.deg2rad(dec[i])) *
            np.cos(np.deg2rad(dec[i+1:])) *
            np.sin(dra/2)**2)                                  # haversine
    ang  = 2 * np.arcsin(np.sqrt(sin2))
    pair_cnt += np.count_nonzero(ang < rad_lim)
secs = time.perf_counter() - t0

# ---------- save result ----------
out = pathlib.Path('~/env_QGA/results/baseline.csv').expanduser()
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'N':[n_stars],
              'pairs':[pair_cnt],
              'secs':[secs]}).to_csv(out, index=False)         # CSV export
print(f"N={n_stars}, close_pairs={pair_cnt}, time={secs:.2f}s")
