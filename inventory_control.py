import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------- constants ----------------------------------
gamma = 2.0      # cost of ordering one unit
beta = 3.0       # cost of storing one unit
max_ord = 50.0   # maximal ordering rate
T = 100.0        # time horizon

def demand(t):
    """Customer demand rate function."""
    return 15.0 + np.sin(2.0 * np.pi * t / 10.0) * 5.0


# -------------------------------- discretisation -------------------------------
num_x = 400      # number of spatial intervals
num_t = 6000     # number of time steps

x_max = 1000.0
dx = x_max / num_x
dt = T / num_t

x_grid = np.linspace(0.0, x_max, num_x + 1)
t_grid = np.linspace(0.0, T, num_t + 1)


# ----------------------------- Hamiltonian interior -----------------------------
def hamiltonian_interior(t, x, p):
    """
    H(t,x,p) for interior points x > 0:
    minimize over u in [0, max_ord]:
        gamma u + beta x + p (u - d(t))
    = beta x - p d(t) + min_{u} (gamma + p) u
    """
    d = demand(t)
    return beta * x - p * d + np.minimum(0.0, (gamma + p) * max_ord)

# ----------------------------- HJB backward solver ------------------------------
def solve_hjb():
    """
    Solve the HJB equation backward in time using explicit Euler.
    Returns:
        V        : value function array, shape (NUM_T+1, NUM_X+1)
        Vx       : spatial derivative of V
    """

    # V[n, i] ~ V(t_n, x_i)
    V = np.zeros((num_t + 1, num_x + 1))

    # terminal condition: pure running cost ⇒ V(T, x) = 0
    V_curr = np.zeros_like(x_grid)
    V[-1, :] = V_curr.copy()

    for n in range(num_t, 0, -1):
        t = t_grid[n]

        # compute V_x using one-sided derivative at x=0
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

    # compute V_x on all time slices
    Vx_all = np.empty_like(V)
    Vx_all[:, 0] = (V[:, 1] - V[:, 0]) / dx
    Vx_all[:, 1:] = (V[:, 1:] - V[:, :-1]) / dx

    return V, Vx_all

# ------------------------------ feedback control --------------------------------
def compute_feedback_control(Vx_all):
    """
    Compute feedback law u*(t, x) from V_x.
    Returns:
        u_star : control array of shape (NUM_T+1, NUM_X+1)
    """
    # interior: bang-bang control
    u_star = np.where(gamma + Vx_all < 0.0, max_ord, 0.0)

    # boundary constraint u >= demand(t)
    u_star[:, 0] = np.maximum(u_star[:, 0], demand(t_grid))

    return u_star

# -------------------------- forward controlled dynamics -------------------------
def simulate_forward(u_star, x0=400.0):
    """
    Forward simulation of ẋ = u(t,x) − demand(t)
    using the feedback law u_star.
    """
    x_traj = np.zeros(num_t + 1)
    u_traj = np.zeros(num_t + 1)

    x_traj[0] = x0

    for n in range(num_t):
        t = t_grid[n]
        xn = x_traj[n]

        # clamp state inside numerical grid
        xn = min(max(xn, 0.0), x_max - 1e-8)

        # linear interpolation in x
        j = xn / dx
        j0 = int(np.floor(j))
        j1 = min(j0 + 1, num_x)
        theta = j - j0

        u_n = (1.0 - theta) * u_star[n, j0] + theta * u_star[n, j1]
        u_traj[n] = u_n

        # inventory dynamics
        x_next = x_traj[n] + dt * (u_n - demand(t))
        x_traj[n + 1] = max(0.0, x_next)

    # last control value (for plotting convenience)
    u_traj[-1] = u_traj[-2]

    return x_traj, u_traj


# ------------------------------------- main -------------------------------------
def main():
    # solve HJB backward
    V, Vx_all = solve_hjb()

    # compute feedback law
    u_star = compute_feedback_control(Vx_all)

    # forward simulation from inventory x(0) = 400
    x_traj, u_traj = simulate_forward(u_star, x0=400.0)

    # ------------------------------- plotting --------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Inventory trajectory
    axs[0].plot(t_grid, x_traj, label="Inventory x(t)", color='tab:blue')
    axs[0].set_ylabel("Inventory")
    axs[0].grid(True)

    # Control trajectory
    axs[1].plot(t_grid, u_traj, label="Order rate u(t)", color='tab:red')
    axs[1].set_xlabel("Time (t)")
    axs[1].set_ylabel("u(t)")
    axs[1].grid(True)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)  # make room for suptitle
    fig.suptitle("Optimal Inventory Control Trajectory", fontsize=14, fontweight='bold', y=0.98)

    plt.show()


if __name__ == "__main__":
    main()