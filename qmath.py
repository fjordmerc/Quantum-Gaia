from projectq.ops import X, CNOT, Toffoli

def subtract(reg_a, reg_b, cin):
    """reg_a ← reg_a − reg_b  (borrow in cin), in-place, n-bit ripple."""
    n = len(reg_a)
    X | cin
    for k in range(n):
        X      | reg_b[k]
        CNOT   | (reg_a[k], reg_b[k])
    CNOT | (reg_a[0], cin)
    for k in range(1, n):
        CNOT | (reg_a[k], reg_a[k-1])
        Toffoli | (cin,  reg_b[k-1], reg_a[k-1])
    # un-X reg_b after use
    for k in range(n):
        X | reg_b[k]
    X | cin


def square(reg_in, reg_out):
    """reg_out ← reg_in² (classical control load, reversible when uncomputed)."""
    value = sum(int(bit) << idx for idx, bit in enumerate(reg_in))
    const = value * value
    for i in range(len(reg_out)):
        if (const >> i) & 1:
            X | reg_out[i]


def add_no_carry(reg_a, reg_b, carry):
    """reg_b += reg_a (no input carry)."""
    n = len(reg_a)
    for k in range(n):
        CNOT | (reg_a[k], reg_b[k])
    for k in range(n-1):
        Toffoli | (reg_a[k], reg_b[k], reg_b[k+1])
    Toffoli | (reg_a[n-1], reg_b[n-1], carry)
    for k in reversed(range(n-1)):
        Toffoli | (reg_a[k], reg_b[k], reg_b[k+1])

