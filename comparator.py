from projectq.ops import Measure
from qsec.qmath import compare_less

def oracle_comparator(bb, thresh_bits):
    """
    Compare (dx²+dy²) vs threshold².
    Input register layout in Breadboard instance:
        sum  stored in bb.b[0:len_thresh]
    """
    n = len(thresh_bits)
    anc = bb.a[0]            # reuse any free ancilla
    compare_less(bb.b[:n], thresh_bits, anc, bb.flag)
    Measure | bb.flag
