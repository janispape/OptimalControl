import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ---------------------------------- constants ----------------------------------
# physical / spacecraft constants
g = 1.6 # lunar gravity
g0 = 9.81 # earth gravity
Isp = 500.0 # specific impulse
k = 1.0 / (g0*Isp)
m_dry = 2000.0 # mass of the empty spacecraft
T_max = 6000.0 # maximal thrust

# grid on normalised time tau ∈ [0,1]
N = 100 # nodes size
tau = np.linspace(0.0, 1.0, N)
dtau = np.diff(tau)

# initial state - we start at 200 m above the lunar surface at 0 velocity and the spacecrafts
# mass is 67% spacecraft and 33% fuel
x0 = np.array([200.0, 0.0, 3000.0])

# decision vector layout:
# h[0..N-1], v[0..N-1], m[0..N-1], u[0..N-2], Tf (scalar)
n_h = n_v = n_m = N
n_u = N - 1
offset_h = 0
offset_v = offset_h + n_h
offset_m = offset_v + n_v
offset_u = offset_m + n_m
offset_Tf = offset_u + n_u
nz = offset_Tf + 1


# ----------------------------------- helpers -----------------------------------
def unpack(z):
    """The state vector we work with is flattened, this unpacks it"""
    h = z[offset_h:offset_h+n_h]
    v = z[offset_v:offset_v+n_v]
    m = z[offset_m:offset_m+n_m]
    u = z[offset_u:offset_u+n_u]      # one per interval
    Tf = z[offset_Tf]
    return h, v, m, u, Tf

def f(h, v, m, u):
    """ returns [h_dot, v_dot, m_dot] """
    thrust = T_max * u
    return np.stack([v, -g + thrust/m, -k * thrust], axis=-1)


# ---------------------------------- objective ----------------------------------
def minimizing_function(z):
    """We maximise the final mass of the spacecraft"""
    _, _, m, _, _ = unpack(z)
    return -m[-1]


# ---------------------------- constraints + bounds -----------------------------
def dynamics_constraint(z):
    """Trapezoidal transcription defects for (h,v,m)"""
    h, v, m, u, Tf = unpack(z)
    # piecewise-constant u on intervals; use u_k at both ends
    f_k  = f(h[:-1], v[:-1], m[:-1], u)       # shape (N-1, 3)
    f_k1 = f(h[1:],  v[1:],  m[1:],  u)       # shape (N-1, 3)
    x_k  = np.stack([h[:-1], v[:-1], m[:-1]], axis=1)
    x_k1 = np.stack([h[1:],  v[1:],  m[1:]], axis=1)
    trap = x_k1 - x_k - 0.5 * (Tf * dtau)[:, None] * (f_k + f_k1)
    return trap.ravel()

def initial_state_eq(z):
    h, v, m, _, _ = unpack(z)
    return np.array([h[0] - x0[0], v[0] - x0[1], m[0] - x0[2]])

v_tol = 0.25
h_tol = 1e-3

def terminal_ineq(z):
    h, v, _, _, _ = unpack(z)
    return np.array([
        h_tol - h[-1],     # h_final \leq h_tol
        v_tol - v[-1],   # v_final \leq v_tol
        v[-1] + v_tol    # - v_tol \leq v_final
    ])

constraints = [
    {'type' : 'eq', 'fun': lambda values: dynamics_constraint(values)},
    {'type' : 'eq',   'fun': initial_state_eq},
    {'type' : 'ineq', 'fun': terminal_ineq},
]

bounds_h = [(-h_tol, None)] * N     # h >= -h_tol
bounds_v = [(None, None)] * N       # v free
bounds_m = [(m_dry, None)] * N      # m >= m_dry
bounds_u = [(0, 1)] * (N-1)         # 0 <= u <= 1
bounds_Tf = [(5.0, 300.0)]          # possible time horizons

# concatenate all
bounds = bounds_h + bounds_v + bounds_m + bounds_u + bounds_Tf


# -------------------------------- initial guess --------------------------------
z0 = np.zeros(nz)

# linear fall
z0[offset_h:offset_h+N] = np.linspace(x0[0], 0, N)

# for v descending to halftime, ascending in the second
z0[offset_v:offset_v+N//2] =  np.linspace(0.0, -10.0, N//2)
z0[offset_v+N//2:offset_v+N] = np.linspace(-10.0, 0.0, N - N//2)

# half of the fuel is used
z0[offset_m:offset_m+N] = np.linspace(x0[2], (x0[2] + m_dry) / 2, N)

# initial guess for u: 0 in the first half, 1 in the second
u0 = np.zeros(N-1)
u0[(N-1)//2:] = 1.0
z0[offset_u:offset_u+(N-1)] = u0

z0[offset_Tf] = 60.0


# ------------------------------------ solve ------------------------------------
res = minimize(
    fun=minimizing_function,
    x0=z0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-10, 'eps': 1e-8, 'disp': True}
)

print("Optimization success:", res.success)
print("Message:", res.message)

h, v, m, u, Tf = unpack(res.x)
t = Tf * tau


# ------------------------------------ plots ------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

axs[0].plot(t, h, label="Height (m)")
axs[0].set_ylabel("Height (m)")
axs[0].grid(True)

axs[1].plot(t, v, label="Velocity (m/s)", color='tab:orange')
axs[1].set_ylabel("Velocity (m/s)")
axs[1].grid(True)

axs[2].plot(t, m, label="Mass (kg)", color='tab:green')
axs[2].set_ylabel("Mass (kg)")
axs[2].grid(True)


axs[3].plot(t, np.append(u, u[-1]), label="Throttle (0–1)", color='tab:red')
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("Throttle")
axs[3].grid(True)

fuel_filling = (m[-1]-m_dry)/(m[0]-m_dry)
plt.tight_layout()
fig.subplots_adjust(top=0.90)  # make room for suptitle
fig.suptitle("Optimal Lunar Landing Trajectory", fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.94, f"(optimal duration: T={t[-1]:.2f}s, "
                    f"final fuel: {fuel_filling * 100:.2f}%)", ha='center', fontsize=10, style='italic')

plt.show()