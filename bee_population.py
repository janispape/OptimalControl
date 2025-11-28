import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# death and work rates respectively
death_workers = 1.0
death_queens = 2.0
birth_workers = 2.0
birth_queens = 1.0

mu = death_workers
nu = death_queens
b = birth_workers
c = birth_queens

# time horizon
T = 365

# productivity function - this gives us the productivity of the workers in producing economic output
# which then can be used to produce and maintain offspring.
# If we have a time-step being equal to a day, it might be sensible to choose s(t) = 1 + cos((2*x - 365)*pi/365).
# This way we model the times of the year with the summer being the most fruitful.
def prod(t):
    return 1 + np.cos((2 * t - 365) * np.pi / 365)

# rhs function for the ODE governing x
def f(t, x, u, s = prod):
    w, q = x
    return np.array([
        -mu * w + b * s(t) * u(t) * w,
        -nu * q + c * s(t) * (1 - u(t)) * w
    ])

# rhs function for the ODE governing the costate p
# in our case this is the partial derivative of the non-maximized Hamiltonian in x
def phi(t, p, u):
    p1, p2 = p
    return np.array([
        (mu - b * prod(t) * u(t)) * p1,
        nu * p2
    ])

def shoot(x0, p0_init, u_init, eps = 1e-5, max_iter = 100):

    p0_curr = p0_init
    u_curr = u_init
    p0_last = p0_curr + 2 * eps
    iteration_count = 0

    while(np.abs(p0_last - p0_curr) > eps and iteration_count < max_iter):
        sol_x_curr = solve_ivp(
            f,
            t_span=(0.0, T),
            y0 = x0,
            args = (u_curr,),
            dense_output=True
        )

        sol_p_curr = solve_ivp(
            phi,
            t_span=(0.0, T),
            y0 = p0_curr,
            args = (u_curr,),
        )

        iteration_count += 1