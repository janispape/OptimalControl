import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# death rate of the workers (mu) and the queens (nu
mu, nu = 1, 2

# birth rate of the workers (b) and the queens (c)
b, c = 2, 1

def f(x, alpha):
    w, q = x
    return [-mu * w + b * alpha * w, - nu * q + c * (1-alpha) * w]

def g(x,alpha):
    return alpha

# the function \mathcal{H} on pg. 6 of the Gruene article
def hamiltonian(x, p ,alpha):
    return g(x,alpha) + p * f(x, alpha)

# the right side of the PDE governing the costate p
def hamiltonian_derivative(x, p ,alpha):
    h, v, m = x
    p1, p2, p3 = p
    return [0, p1, - p1 * alpha / m*m]