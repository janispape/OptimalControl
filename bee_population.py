import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------- constants ----------------------------------
# mortality and birth rates
death_workers = 0.4
death_queens = 0.2
birth_workers = 0.4
birth_queens = 1.2

mu = death_workers
nu = death_queens
b = birth_workers
c = birth_queens

# time horizon
T = 10.0

# productivity function - this gives us the productivity of the workers in producing economic output
# which then can be used to produce and maintain offspring.
def prod(t):
    """Constant productivity function."""
    return 2.0


# --------------------------- system dynamics: state ----------------------------
def f(t, x, u, s = prod):
    """RHS of the state ODE."""
    w, q = x
    return np.array([
        -mu * w + b * s(t) * u(t) * w,
        -nu * q + c * s(t) * (1 - u(t)) * w
    ])


# -------------------------- system dynamics: costate ---------------------------
# in our case this is the partial derivative of the non-maximized Hamiltonian in x
def phi(t, p, u):
    """RHS of the costate ODE."""
    p1, p2 = p
    return np.array([
        (mu - b * prod(t) * u(t)) * p1 + c * prod(t) * (u(t) - 1.0) * p2,
        nu * p2
    ])


# --------------------------- Hamiltonian evaluation ----------------------------
def hamiltonian(t, x, p, u, s = prod):
    """Hamiltonian for fixed u."""
    if callable(u):
        u_func = u
    else:
        u_val = float(u)
        u_func = lambda _t: u_val
    return np.dot(f(t, x, u_func, s), p)


# ----------------------------------- shooting ----------------------------------
def shoot(x0, p0_init, u_init, eps = 1e-5, eta = 1e-5, max_iter = 100):
    """
    Solve the two-point boundary value problem by Newton–shooting.
    Returns: (optimal control function u(t), state trajectory function x(t)).
    """
    p0_curr = p0_init
    u_curr = u_init
    p0_last = p0_curr + 2 * eps
    iteration_count = 0

    while(np.linalg.norm(p0_last - p0_curr) > eps and iteration_count < max_iter):

        # Solve state and costate trajectories for current guess
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
            dense_output=True
        )

        xT = sol_x_curr.sol(T)
        pT = sol_p_curr.sol(T)

        # Boundary condition: p(T) = ∂F/∂x where F(x(T)) = q(T)
        Fx = np.array([0.0, 1.0])
        G = pT - Fx  # shape (2,)

        # Finite-difference Jacobian DG
        DG = np.zeros((2, 2))

        # In our case F_x is constant, so DG is just p(T) differentiated by p0.
        # The first column is then the directional derivative when p0 is pertubed in the (1,0) direction,
        # while the second column is the analogon in the (0,1) direction.
        # We use the eta (default 1e-5) to get a suitable step size.

        h1 = eta * max(1.0, abs(p0_curr[0]))
        h2 = eta * max(1.0, abs(p0_curr[1]))

        # first pertubation
        p0_pert = p0_curr.copy()
        p0_pert[0] += h1
        sol_p1 = solve_ivp(phi, t_span=(0.0, T), y0=p0_pert, args=(u_curr,))
        DG[:, 0] = (sol_p1.y[:, -1] - pT) / h1

        # second pertubation
        p0_pert = p0_curr.copy()
        p0_pert[1] += h2
        sol_p2 = solve_ivp(phi, t_span=(0.0, T), y0=p0_pert, args=(u_curr,))
        DG[:, 1] = (sol_p2.y[:, -1] - pT) / h2

        # Newton update
        s = np.linalg.solve(DG, -G)
        p0_last = p0_curr
        p0_curr = p0_curr + s

        # improved control update
        def updated_control(t):
            """Maximizing control for given t using Hamiltonian."""
            u_space = np.linspace(0.0, 1.0, 100)
            values = [hamiltonian(t, sol_x_curr.sol(t), sol_p_curr.sol(t), u_val) for u_val in u_space]
            return u_space[np.argmax(values)]

        u_curr = updated_control

        iteration_count += 1

    sol_x_final = solve_ivp(
        f,
        t_span=(0.0, T),
        y0=x0,
        args=(u_curr,),
        dense_output=True
    )

    return u_curr, sol_x_final.sol


# ------------------------------------- main ------------------------------------
def main():
    p0_init = np.array([0.1, 0.1])
    x0 = np.array([50.0, 50.0])

    def u_init(t):
        return 0.5

    control, trajectory = shoot(x0, p0_init, u_init)

    interval = np.linspace(0, T, 400)
    control_disc = np.vectorize(control)(interval)
    w_disc, q_disc = trajectory(interval)

    # ------------------------------- plotting --------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(interval, w_disc, label="Workers", color='tab:blue')
    axs[0].set_ylabel("Workers")
    axs[0].grid(True)

    axs[1].plot(interval, q_disc, label="Queens", color='tab:green')
    axs[1].set_ylabel("Queens")
    axs[1].grid(True)

    axs[2].step(interval, control_disc, where='post', label="Control u(t)", color='tab:red')
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("u(t)")
    axs[2].grid(True)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle("Optimal Bee Colony Management", fontsize=14, fontweight='bold', y=0.98)
    fig.text(
        0.5, 0.94,
        f"final queens: q(T) = {q_disc[-1]:.2f}",
        ha='center', fontsize=10, style='italic'
    )

    plt.show()


if __name__ == "__main__":
    main()