import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# cost of ordering (gamma) and storing (beta) one unit of goods:
gamma, beta  = 3, 1

# customer demand function:
def demand(t):
    return 15 + np.sin(2 * np.pi * t / 10) * 5

# time horizon:
T = 100
I = [0, T]

# Hamiltonian of the problem
def hamiltonian(t, x, p):
    return - beta * x + p * demand(t) - (gamma + p) * max_ord * (gamma + p < 0)

# maximal ordering rate (usually \infty, but to compute the hamiltonian, we have to bound it)
max_ord = 100

# Grid
I = 400
N = 6000

x_max = 1000
dx = x_max / I
dt = T / N

x = np.linspace(0, x_max, I+1)

# Storage with terminal condition V(T,x) = \beta * x * dt saved in V_curr
V = np.zeros((N+1, I+1))
V_curr = beta * x
V_next = np.zeros((I+1))

# initial column t = T for whole function V
V[-1,:] = V_curr
Vx = np.zeros((I+1))

# time-stepping backwards:
for n in range(N, 0, -1):

    t = n * dt

    # 1. compute V_x from current time layer V_curr, using backward difference
    Vx = np.zeros_like(V_curr)
    Vx[1:] = (V_curr[1:] - V_curr[:-1]) / dx
    Vx[0] = Vx[1]   # simple boundary (you can improve if needed)

    # 2. compute H(t, x, Vx)
    H = hamiltonian(t, x, Vx)

    # 3. update backwards in time
    V_prev = V_curr + dt * H

    # 4. store if desired
    V[n-1,:] = V_prev

    # update current layer
    V_curr = V_prev


fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(V.shape[1])

y_min = np.min(V)
y_max = np.max(V)
padding = 0.1 * (y_max - y_min)
ax.set_ylim(y_min - padding, y_max + padding)

# create an empty line object
(line,) = ax.plot(x, V[0, :], lw=2)

def update(frame):
    line.set_ydata(V[frame, :])
    return (line,)

anim = animation.FuncAnimation(
    fig, update, frames=range(0, V.shape[0], 10), interval=50, blit=True
)

anim.save("solution.mp4", fps=30, dpi=150)
