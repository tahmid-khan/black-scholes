"""
Python rewrite of the MATLAB script `Convergence_of_the_solutuion_with_number_of_qubits.m`
"""

import math

import numpy as np
from matplotlib import pyplot as plt
from mibian import BS
from scipy.linalg import expm

# Initial Parameters
s = 150  # Maximum stock price
K = 50  # Strike
sigma = 0.2  # Volatility
r = 0.04  # Interest rate
T = 1  # Maturity time

for Nx in (4, 5, 6, 7, 8, 10):  # Nx is the number of qubits
    ncols = 2**Nx

    # Definitions of operators

    # Position
    xmax = 2 * math.log(s)
    deltax = 4 * math.log(s) / (ncols - 1)
    x = np.ndarray(ncols, dtype=np.float64)
    x[0] = -2 * math.log(s)
    for j in range(1, ncols):
        x[j] = -2 * math.log(s) + j * deltax
    sv = np.exp(x + xmax / 2)

    # Momentum
    P = np.diag(np.ones(ncols - 1, dtype=np.complex128), 1) + np.diag(
        -np.ones(ncols - 1, dtype=np.complex128), -1
    )
    P[0, -1] = -1  # Periodic Boundary condition
    P[-1, 0] = 1  # Periodic Boundary condition
    P *= -1j / (2 * deltax)

    # Initial condition
    C0 = np.zeros(ncols // 2)
    for j in range(C0.size):
        if math.exp(math.log(1 / s) + j * deltax) < K:
            C0[j] = K - math.exp(math.log(1 / s) + j * deltax)
    C0 = np.concatenate((C0, np.flip(C0)))

    # Evolution operators
    Unit = expm(-1j * T * (sigma**2 / 2 - r) * P)
    O = expm(-T * (sigma**2 / 2 * P @ P + r * np.eye(ncols)))

    # Evolved state
    Phif = np.dot(Unit, np.dot(O, C0))
    Phif = Phif[: Phif.size // 2]
    sv = sv[: Phif.size]
    x = x[: Phif.size]
    C0 = C0[: Phif.size]

    # Analytical solution

    if Nx == 10:
        Put = [BS([price, K, r, T], volatility=sigma).putPrice for price in sv]
        (h1,) = plt.plot(sv, Put, "-")
        plt.axis((0, 135, 0, 40))
        plt.setp(h1, linewidth=2)
        plt.xlabel(r"Stock Value ($S$)", fontsize=17)
        plt.ylabel("Option Price", fontsize=17)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    else:
        (h2,) = plt.plot(sv, Phif, "--")
        plt.axis((0, 135, 0, 40))
        plt.setp(h2, linewidth=2)
        plt.xlabel(r"Log Stock Value ($x = \log(S)$)", fontsize=17)
        plt.ylabel("Initial Option Price", fontsize=17)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

plt.legend(
    ["4 Qubits", "5 Qubits", "6 Qubits", "7 Qubits", "8 Qubits", "Analytical"],
    loc="best",
    fontsize=12,
)

# Show the plot
# plt.grid(True)  # Add gridlines
plt.tight_layout()  # Adjust spacing for better readability
plt.show()
