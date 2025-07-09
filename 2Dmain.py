"""
General-width oracle demo.
argv: xi xj yi yj thresh_sq  (all decimal)
"""

import sys
from projectq import MainEngine
from qsec.Breadboard import Breadboard
from qsec.qmath import subtract, square, add_no_carry
from qsec.qint  import compare_less

if len(sys.argv) != 6:
    sys.exit("args: xi xj yi yj thresh_sq")

xi,xj,yi,yj,th = map(int, sys.argv[1:])
bits = max(xi,xj,yi,yj,th).bit_length()
eng  = MainEngine()
bb   = Breadboard(eng, bits)

# load constants
for i,val in enumerate([xi,xj,yi,yj]):
    reg = [bb.xi,bb.xj,bb.yi,bb.yj][i]
    for k in range(bits):
        if (val>>k)&1: from projectq.ops import X; X|reg[k]

# dx = xi - xj ; dy = yi - yj
subtract(bb.xi, bb.xj, bb.cin)
subtract(bb.yi, bb.yj, bb.cin)

# squares
dx_sq = eng.allocate_qureg(2*bits)
dy_sq = eng.allocate_qureg(2*bits)
square(bb.xj, dx_sq)
square(bb.yj, dy_sq)

# sum = dx_sq + dy_sq
add_no_carry(dx_sq, dy_sq, bb.z)

# threshold register
thr = eng.allocate_qureg(len(dy_sq))
for k in range(len(thr)):
    if (th>>k)&1: from projectq.ops import X; X|thr[k]

# compare
compare_less(dy_sq, thr, bb.cin, bb.flag)

eng.flush()
print("flag =", int(bb.flag))
