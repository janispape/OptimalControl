import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Problem data
# -----------------------------

# cost of ordering (gamma) and storing (beta) one unit of goods:
gamma, beta  = 2.0, 3.0

# customer demand function:
def demand(t):
    return 15.0 + np.sin(2.0 * np.pi * t / 10.0) * 5.0

# time horizon:
T = 100.0

# maximal ordering rate (upper bound on control)
max_ord = 50.0   # must be >= max_t demand(t)


# -----------------------------
# Discretisation
# -----------------------------

num_x = 400      # number of spatial intervals
num_t = 6000     # number of time steps

x_max = 1000.0
dx = x_max / num_x
dt = T / num_t

x_grid = np.linspace(0.0, x_max, num_x + 1)
t_grid = np.linspace(0.0, T, num_t + 1)

# time horizon:
T = 100
I = [0, T]

# -----------------------------
# Hamiltonian (interior, no state constraint)
# -----------------------------
def hamiltonian_interior(t, x, p):
    """
    H(t,x,p) for interior points x > 0:
    minimize over u in [0, max_ord]:
        gamma u + beta x + p (u - d(t))
    = beta x - p d(t) + min_{u} (gamma + p) u
    """
    d = demand(t)
    return beta * x - p * d + np.minimum(0.0, (gamma + p) * max_ord)

# -----------------------------
# HJB solve backward in time
# -----------------------------

# V[n, i] ~ V(t_n, x_i)
V = np.zeros((num_t + 1, num_x + 1))

# Terminal condition: pure running cost ⇒ V(T, x) = 0
V_curr = np.zeros_like(x_grid)
V[-1, :] = V_curr.copy()

# Backward time stepping (explicit Euler in time)
for n in range(num_t, 0, -1):
    t = t_grid[n]

    # compute V_x at current time layer using one-sided derivative at x=0
    Vx = np.empty_like(V_curr)
    Vx[0] = (V_curr[1] - V_curr[0]) / dx          # forward diff at boundary
    Vx[1:] = (V_curr[1:] - V_curr[:-1]) / dx      # backward diff elsewhere

    # interior Hamiltonian
    H = hamiltonian_interior(t, x_grid, Vx)

    # boundary cell x = 0: enforce state constraint (u >= d(t))
    p0 = Vx[0]
    d0 = demand(t)

    # feasible set at x=0: u ∈ [d0, max_ord]
    # minimize (gamma + p0) u - p0 d0
    if gamma + p0 >= 0.0:
        # minimizer u = d0
        H[0] = gamma * d0  # because (gamma+p0)*d0 - p0*d0 = gamma*d0
    else:
        # minimizer u = max_ord
        H[0] = (gamma + p0) * max_ord - p0 * d0

    # explicit Euler step backward in time:
    V_prev = V_curr + dt * H

    # store and update
    V[n - 1, :] = V_prev
    V_curr = V_prev

# -----------------------------
# Compute V_x on whole grid (for feedback law)
# -----------------------------
Vx_all = np.empty_like(V)
# one-sided derivative in x-direction for each time slice
Vx_all[:, 0] = (V[:, 1] - V[:, 0]) / dx
Vx_all[:, 1:] = (V[:, 1:] - V[:, :-1]) / dx

# -----------------------------
# Feedback control law u*(t,x)
# -----------------------------
# interior control (ignoring state constraint): bang-bang
u_star = np.where(gamma + Vx_all < 0.0, max_ord, 0.0)

# enforce boundary state constraint at x = 0: u >= demand(t)
# (vectorized over time)
u_star[:, 0] = np.maximum(u_star[:, 0], demand(t_grid))

# -----------------------------
# Forward simulation with feedback
# -----------------------------

x0 = 400.0  # initial inventory
x_traj = np.zeros(num_t + 1)
u_traj = np.zeros(num_t + 1)

x_traj[0] = x0

for n in range(num_t):
    t = t_grid[n]
    xn = x_traj[n]

    # clamp state to grid for safety
    if xn <= 0.0:
        xn = 0.0
    if xn >= x_max:
        xn = x_max - 1e-8

    # find neighboring grid indices in x and interpolate control
    j = xn / dx
    j0 = int(np.floor(j))
    j1 = min(j0 + 1, num_x)
    theta = j - j0

    # linear interpolation in space:
    u_n = (1.0 - theta) * u_star[n, j0] + theta * u_star[n, j1]
    u_traj[n] = u_n

    # system dynamics: ẋ = u − demand(t)
    x_next = x_traj[n] + dt * (u_n - demand(t))

    # small numerical projection to enforce x >= 0
    x_traj[n + 1] = max(0.0, x_next)

# last control value (for plotting convenience)
u_traj[-1] = u_traj[-2]

# -----------------------------
# Plot results
# -----------------------------

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_grid, x_traj)
plt.xlabel("Time t")
plt.ylabel("Inventory x(t)")
plt.title("Optimal state trajectory")

plt.subplot(1, 2, 2)
plt.plot(t_grid, u_traj)
plt.xlabel("Time t")
plt.ylabel("Order rate u(t)")
plt.title("Applied optimal control")

plt.tight_layout()
plt.show()