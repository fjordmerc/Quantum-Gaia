#!/usr/bin/env python3
"""
Validate distance-threshold oracle on N = 2,4,8,16,32 toy points.
Outputs fidelity per N and a bar-chart PDF.
"""
import sys, pathlib, itertools, json, numpy as np, pandas as pd
from projectq import MainEngine
from projectq.ops import All, Measure
from oracle.distance_oracle import DistanceOracle2D
from matplotlib import pyplot as plt

# Add project root to Python path
sys.path.append(str(pathlib.Path('~/env_QGA').expanduser()))

SIZES = [2, 4, 8, 16, 32]                                         # requested cases
THRESH = 3                                                        # lattice radius

def brute_truth(coords):
    pairs = {(i, j): (abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) < THRESH ** 2
             for (i, (x1, y1)), (j, (x2, y2))
             in itertools.combinations(enumerate(coords), 2)}
    return pairs

records = []
for n in SIZES:
    coords = [(i, 0) for i in range(n)]                           # 1-D lattice
    truth = brute_truth(coords)
    eng = MainEngine()
    oracle = DistanceOracle2D(bits_per_coord=5, thresh_sq=THRESH**2)  # 5-bit precision
    ok = 0
    for (i, j), mark in truth.items():
        result = oracle.run_test(eng, i, j)                       # helper returns bool
        ok += (result == mark)
    fid = ok / len(truth)
    records.append((n, fid))
    eng.flush()
df = pd.DataFrame(records, columns=["N", "fidelity"])
out = pathlib.Path("~/env_QGA/results/oracle_truth").expanduser()
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "oracle_fidelity.csv", index=False)

# bar-chart
plt.bar(df.N, df.fidelity)
plt.ylim(0, 1.05)
plt.ylabel("Logical fidelity")
plt.xlabel("Catalogue size N")
plt.title("Oracle circuit validation")
plt.savefig(out / "fig_oracle_validation.pdf")
