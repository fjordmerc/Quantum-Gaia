# qsec
A quantum circuit for Squared Euclidean Distance comparison

This code implements a proposed design of a quantum circuit for the computation of the Squared Euclidean distance between two data points and the comparation of this distance with an upper bound. In this example, the design is adapted to work with points of 2 coordinates X and Y of 3 bits each one (2D points of N=3 digits), but the design can be easily adapted to any number of dimensions and digits. For more information about the methodology, please refer to the paper XXX (the paper is currently under evaluation in the IEEE Transactions on Computers journal).

The circuit has been built using ProjectQ, an open-source software framework for quantum computing. More information about ProjectQ is available in https://projectq.ch/.

The repository contains 4 .py files:
-2Dmain.py: The main file. Run this file to test the example. It accepts 4 input arguments: Xi, Xj, Yi, Yj, and the upper bound. The coordinates are limited to 3 (binary) digits, and the upper bound to 6.
-Breadboard.py: It includes the declaration of the necessary qubits.
-qint.py: This file contains the comparator circuit.
-qmath.py: Includes the subtractors, adders and squaring circuits.
-comparator.py: The generic design of the comparator. This file is not used in 2DCompleto.py, it is ready for the evaluation of the proposed comparator only.

The description of each circuit is included in the referenced paper.

The adder circuit was published in https://dl.acm.org/doi/abs/10.1145/2491682
The subtractor circuit is an adaptation of the previous adder.
The squaring circuit was published in https://dl.acm.org/doi/10.1007/s00034-017-0631-5
The comparator is proposed in our paper.
