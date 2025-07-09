from projectq.ops import X, CNOT, Toffoli

def compare_less(reg_a, reg_b, anc, out):
    """
    Half-comparator (Li & Fan design, parameterised).
    Sets |out⟩ = 1  iff  A < B  for n-bit integers in two's complement.
    anc : one ancilla qubit.
    """
    n = len(reg_a)
    X | reg_a[-1]
    for k in range(n-1):
        CNOT | (reg_b[k], reg_a[k])
    CNOT | (reg_b[-1], out)
    CNOT | (reg_a[-1], reg_b[-1])
    for k in reversed(range(n)):
        Toffoli | (reg_b[k], reg_a[k], anc)
        CNOT   | (reg_a[k], reg_b[k])
    CNOT | (anc, out)
    # uncompute
    for k in range(n):
        CNOT   | (reg_a[k], reg_b[k])
        Toffoli| (reg_b[k], reg_a[k], anc)
    CNOT | (reg_a[-1], reg_b[-1])
    CNOT | (reg_b[-1], out)
    for k in range(n-1):
        CNOT | (reg_b[k], reg_a[k])
    X | reg_a[-1]
