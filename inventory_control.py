import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# cost of ordering (gamma) and storing (beta) one unit of goods:
gamma, beta  = 3, 1

# customer demand function:
def demand(t):
    return 15 + np.sin(2 * np.pi * t / 10) * 5

# time horizon:
T = 100
I = [0, T]

# maximal ordering rate (usually \infty, but to compute the hamiltonian, we have to bound it
max_ord = 1000
possible_alpha_values = np.linspace(0, max_ord, max_ord*100)

# cost function:
def g(x, alpha):
    return gamma * alpha + beta * x

# right side of the ODE:
def f(x, alpha, t):
    return alpha - demand(t)

# Hamiltonian of the problem
def hamiltonian(x, p, t):
    possible_outputs = [gamma * a + beta * x + p * (a - demand(t)) for a in possible_alpha_values]
    return np.max(possible_outputs)