import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def alpha(t, q, v):
    if np.hypot(q, v) < 1e-12:
        return 0.0
    s = v + np.sign(q) * np.sqrt(2.0 * np.abs(q))
    if s > 0:
        return -1.0
    elif s < 0:
        return 1.0
    else:
        return 0.0

def f(t, x):
    q, v = x
    return [v, alpha(t, q, v)]

# Anfangszustand
y0 = np.array([-15.0, -20.0])
T = [0, 100]

# Integration
sol = solve_ivp(
    f, T, y0,
    method='Radau',
    rtol=1e-6, atol=1e-8,
    max_step=0.1,
    dense_output=False
)

# Alpha für die ausgegebenen Zeitpunkte auswerten
alpha_eval = [alpha(t, q, v) for t, q, v in zip(sol.t, sol.y[0], sol.y[1])]

# Plot
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label="q(t)")
plt.plot(sol.t, sol.y[1], label="v(t)")
plt.legend(); plt.grid(True)
plt.xlabel("time")
plt.show()

# --- Phasenporträt ---
q_traj = sol.y[0]
v_traj = sol.y[1]

plt.figure(figsize=(7,6))
plt.plot(q_traj, v_traj, '-k', lw=1.5, label="Trajektorie")
plt.scatter([q_traj[0]], [v_traj[0]], color='green', label="Start")
plt.scatter([q_traj[-1]], [v_traj[-1]], color='red', label="Ende")

# Umschaltparabeln v = ±√(2|q|)
q_pos = np.linspace(0, max(1.0, q_traj.max(), 20), 400)
q_neg = np.linspace(min(-1.0, q_traj.min(), -20), 0, 400)

plt.plot(q_pos,  np.sqrt(2*q_pos), '--', color='blue', label="v = +√(2q)")
plt.plot(q_pos, -np.sqrt(2*q_pos), '--', color='blue', label="v = -√(2q)")
plt.plot(q_neg,  np.sqrt(2*np.abs(q_neg)), '--', color='orange', label="v = +√(2|q|)")
plt.plot(q_neg, -np.sqrt(2*np.abs(q_neg)), '--', color='orange', label="v = -√(2|q|)")

plt.xlabel("q")
plt.ylabel("v")
plt.legend()
plt.grid(True)
plt.show()