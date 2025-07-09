#!/usr/bin/env python3
"""
grover_emulation.py – approximate Algorithm 3 (Combarro 2024).

Usage:
    python grover_emulation.py  ../data/pleiades.csv  10
"""
import sys, time, pathlib, random            # stdlib helpers
import numpy as np                           # math engine
import pandas as pd                          # CSV ingest
from tqdm import tqdm                        # progress bar
from qiskit_aer import Aer                   # simulator provider
from qiskit import QuantumCircuit            # minimal circuit
from sys import stderr                       # direct flush channel

print("Counting μ …", file=stderr, flush=True)                # early heartbeat

# ---------- CLI ----------
if len(sys.argv) != 3:
    sys.exit("Usage: grover_emulation.py <csv> <radius_arcsec>")
csv_file  = pathlib.Path(sys.argv[1]).expanduser()
radius_as = float(sys.argv[2])

# ---------- load coordinates ----------
ra, dec = pd.read_csv(csv_file,
                      usecols=['ra', 'dec']).values.T          # two columns
n_stars = len(ra)
rad_lim = np.deg2rad(radius_as / 3600.0)                       # arcsec→rad

# ---------- helper: are two stars close? ----------
def close_pair(i: int, j: int) -> bool:
    dra  = np.deg2rad(ra[i] - ra[j])
    ddec = np.deg2rad(dec[i] - dec[j])
    sin2 = (np.sin(ddec/2)**2 +
            np.cos(np.deg2rad(dec[i])) *
            np.cos(np.deg2rad(dec[j])) *
            np.sin(dra/2)**2)
    return 2 * np.arcsin(np.sqrt(sin2)) < rad_lim              # boolean

# ---------- count true marks μ ----------
mu = 0
for idx in range(n_stars):
    if idx % 500 == 0:                                         # heartbeat every 500 rows
        print(f"μ counter at row {idx}", file=stderr, flush=True)
    for jdx in range(idx + 1, n_stars):
        if close_pair(idx, jdx):
            mu += 1
if mu == 0:
    sys.exit("No neighbours at this radius")

# ---------- Grover constants ----------
search_space = n_stars * n_stars                               # ν
upper_bound  = min(27 * n_stars, search_space)                 # B
sqrt_space   = int(np.sqrt(search_space))                      # √ν
failure_prob = 0.01                                            # w
rep_lim = int(np.ceil(np.log(1 - (1 - failure_prob)**(1/upper_bound))
                      / np.log(3/4)))                          # R

# ---------- emulation loop ----------
left_mu, calls = mu, 0
bar = tqdm(total=mu, desc="Grover found")                      # live bar
t0 = time.perf_counter()
while left_mu:
    success = False
    for _ in range(rep_lim):
        j_len = random.randint(0, sqrt_space - 1)              # Grover length
        calls += 1
        hit_p = np.sin((2*j_len + 1) *
                       np.arcsin(np.sqrt(left_mu / search_space)))**2
        if random.random() < hit_p:
            left_mu -= 1
            bar.update(1)
            success = True
            break
    if not success:
        break
bar.close()
secs = time.perf_counter() - t0

# ---------- save result ----------
outdir = pathlib.Path('~/env_QGA/results').expanduser()
outdir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'N':[n_stars],
              'oracle_calls':[calls],
              'grover_sec':[secs]}
            ).to_csv(outdir/'grover.csv', index=False)         # CSV export
print(f"✓ finished – calls:{calls}  time:{secs:.2f}s")

# ---------- backend sanity check ----------
Aer.get_backend("statevector_simulator").run(QuantumCircuit(1))  # smoke test
