#!/usr/bin/env python3

import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm

# Parameters
S = 150  # maximum stock price
K = 50  # strike
SIGMA = 0.2  # volatility
R = 0.04  # interest rate
T = 1  # maturity time
NQ = 8  # number of qubits


# Definitions of operators

# Position
x_max = 2 * math.log(S)
delta_x = 4 * math.log(S) / (2**NQ - 1)
x = np.ndarray(2**NQ, dtype=np.float64)
x[0] = -2 * math.log(S)
for j in range(1, 2**NQ):
    x[j] = -2 * math.log(S) + j * delta_x
sv = np.exp(x + x_max / 2)

# Momentum
p_hat = np.diag(np.ones(2**NQ - 1, dtype=np.complex128), 1) + np.diag(
    -np.ones(2**NQ - 1, dtype=np.complex128), -1
)
p_hat[0, -1] = -1  # Periodic Boundary condition
p_hat[-1, 0] = 1  # Periodic Boundary condition
print(p_hat)  # should print the matrix in eq. 10
p_hat *= -1j / (2 * delta_x)

# Initial condition
c0 = np.zeros(2**NQ // 2)
for j in range(c0.size):
    if math.exp(math.log(1 / S) + j * delta_x) < K:
        c0[j] = K - math.exp(math.log(1 / S) + j * delta_x)
c0 = np.concatenate((c0, np.flip(c0)))

# Evolution operators
h_bsa = expm(-1j * T * (SIGMA**2 / 2 - R) * p_hat)
o_hat = expm(-T * (SIGMA**2 / 2 * p_hat @ p_hat + R * np.eye(2**NQ)))
print(o_hat)

# Evolved state
# Phif = np.dot(h_bsa, np.dot(o_hat, c0))
# Phif = Phif[: Phif.size // 2]
# sv = sv[: Phif.size]
# x = x[: Phif.size]
# c0 = c0[: Phif.size]

# Analytical solution

# (h2,) = plt.plot(sv, Phif, "--")
# plt.axis((0, 135, 0, 40))
# plt.setp(h2, linewidth=2)
# plt.xlabel(r"Log Stock Value ($x = \log(S)$)", fontsize=17)
# plt.ylabel("Initial Option Price", fontsize=17)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.legend(
#     ["4 Qubits", "5 Qubits", "6 Qubits", "7 Qubits", "8 Qubits", "Analytical"],
#     loc="best",
#     fontsize=12,
# )

# # Show the plot
# # plt.grid(True)  # Add gridlines
# plt.tight_layout()  # Adjust spacing for better readability
# plt.show()
