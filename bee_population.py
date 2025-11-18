import numpy as np
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