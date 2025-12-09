import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------- dynamics -----------------------------------
def alpha(t, q, v):
    """Closed-form optimal control for the double integrator bang-bang solution."""
    s = v + np.sign(q) * np.sqrt(2.0 * np.abs(q))
    return -np.where(s > 0, 1, -1)

def f(t, x):
    """Dynamics of the double integrator: q' = v, v' = u."""
    q, v = x
    return [v, alpha(t, q, v)]


# ------------------------------------ main -------------------------------------
def main():

    # initial condition
    y0 = np.array([-15.0, 0.0])
    T = 100
    N = 1200

    t_grid = np.linspace(0, T, N)

    # integrate system
    sol = solve_ivp(
        f, [0,T], y0,
        method='Radau',
        rtol=1e-6, atol=1e-8,
        max_step=0.1,
        t_eval=t_grid,
        dense_output=False
    )

    q_traj = sol.y[0]
    v_traj = sol.y[1]

    # evaluate control on trajectory
    alpha_eval = np.array([alpha(t, q, v) for t, q, v in zip(sol.t, q_traj, v_traj)])


    # ------------------------------- time plots -------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(sol.t, q_traj, label="q(t)", color='tab:blue')
    axs[0].set_ylabel("q(t)")
    axs[0].grid(True)

    axs[1].plot(sol.t, v_traj, label="v(t)", color='tab:green')
    axs[1].set_ylabel("v(t)")
    axs[1].grid(True)

    axs[2].step(sol.t, alpha_eval, where="post", label="u(t)", color='tab:red')
    axs[2].set_xlabel("Time (t)")
    axs[2].set_ylabel("u(t)")
    axs[2].grid(True)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle("Bang–Bang Control for the Double Integrator", fontsize=14, fontweight='bold', y=0.98)


    plt.figure(figsize=(8,6))
    plt.plot(q_traj, v_traj, '-k', lw=1.5, label="Trajectory")
    plt.scatter(q_traj[0], v_traj[0], color='tab:green', s=80, zorder=3, label="Start")
    plt.scatter(q_traj[-1], v_traj[-1], color='tab:red', s=80, zorder=3, label="End")

    # switching curves: v = ± sqrt(2|q|)
    q_pos = np.linspace(0, max(1.0, q_traj.max(), 20), 400)
    q_neg = np.linspace(min(-1.0, q_traj.min(), -20), 0, 400)

    plt.plot(q_pos,  np.sqrt(2 * q_pos), '--', color='tab:orange', label="v = +√(2q)")
    plt.plot(q_pos, -np.sqrt(2 * q_pos), '--', color='tab:orange', label="v = -√(2q)")
    plt.plot(q_neg,  np.sqrt(2 * np.abs(q_neg)), '--', color='tab:purple', label="v = +√(2|q|)")
    plt.plot(q_neg, -np.sqrt(2 * np.abs(q_neg)), '--', color='tab:purple', label="v = -√(2|q|)")

    plt.xlabel("q")
    plt.ylabel("v")
    plt.title("Phase Space: (q, v)", fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()