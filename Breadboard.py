from projectq.ops import Allocate

class Breadboard:
    """Dynamic breadboard: 2-D coordinates, n bits per coordinate."""

    def __init__(self, eng, n):
        self.n = n

        # coordinates of particle i  (lists: LSB-first)
        self.xi = eng.allocate_qureg(n)
        self.yi = eng.allocate_qureg(n)

        # coordinates of particle j
        self.xj = eng.allocate_qureg(n)
        self.yj = eng.allocate_qureg(n)

        # ancilla pools
        self.a  = eng.allocate_qureg(3*n)   # for x-branch
        self.b  = eng.allocate_qureg(3*n)   # for y-branch
        self.cin = eng.allocate_qubit()     # shared borrow / carry
        self.z   = eng.allocate_qubit()     # global carry out
        self.flag = eng.allocate_qubit()    # oracle output
