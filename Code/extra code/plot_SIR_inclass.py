#plot_SIR_inclass.py
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# - File location -
HERE = Path(__file__).resolve().parent

# - IMPORTANT: the CSV must be in the same folder and contain: day,S,I,R -
sir_csv = HERE / "in_class_SIR_data.csv"
if not sir_csv.exists():    
    raise FileNotFoundError(        
        f"SIR CSV not found: {sir_csv}\n"        
        "Create in_class_SIR_data.csv in the SAME folder as this .py file.\n"        
        "It must contain columns: day,S,I,R"    
    )

# --
# 1) Load and validate the data
# --
df = pd.read_csv(sir_csv)

expected = {"day", "S", "I", "R"}
if not expected.issubset(df.columns):    
    raise KeyError(        
        f"Expected columns {sorted(expected)}, but found: {list(df.columns)}"
    )

days_obs = df["day"].to_numpy()
S_obs = df["S"].to_numpy(dtype=float)
I_obs = df["I"].to_numpy(dtype=float)
R_obs = df["R"].to_numpy(dtype=float)

# If your dataset is strictly SIR, you won't have E in the CSV.
# We'll infer E0 = 0 by default (you can change).
# Total population:
N_obs = (S_obs + I_obs + R_obs).max()  # robust if there is noise
# (If you know a more accurate N, set it below in the "Your best parameters" block.)
# --
# 2) Euler method for SEIR with step dt (in days)
# --
def euler_seir(beta, sigma, gamma, S0, E0, I0, R0, N, n_days, dt=1.0):
    """
    Integrate the SEIR model using explicit Euler.
    dS/dt = -beta * S * I / N
    dE/dt =  beta * S * I / N - sigma * E
    dI/dt =  sigma * E - gamma * I
    dR/dt =  gamma * I
    Returns arrays t, S, E, I, R with length n_steps+1.
    """
    n_steps = int(math.ceil(n_days / dt))
    t = np.arange(0, n_steps + 1) * dt
    S = np.zeros(n_steps + 1)
    E = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0
    for k in range(n_steps):
        # Current values
        Sk, Ek, Ik, Rk = S[k], E[k], I[k], R[k]
        # Flows
        inf_flow = beta * Sk * Ik / N
        exp_to_inf = sigma * Ek
        recov = gamma * Ik
        # Euler updates
        S[k + 1] = Sk - dt * inf_flow
        E[k + 1] = Ek + dt * (inf_flow - exp_to_inf)
        I[k + 1] = Ik + dt * (exp_to_inf - recov)
        R[k + 1] = Rk + dt * recov
        # Optional: keep within numeric bounds
        # (helps if dt is large or parameters extreme)
        for arr in (S, E, I, R):
            if arr[k + 1] < 0:
                arr[k + 1] = 0.0
        total = S[k + 1] + E[k + 1] + I[k + 1] + R[k + 1]
        if total != 0:
            scale = N / total
            S[k + 1] *= scale
            E[k + 1] *= scale
            I[k + 1] *= scale
            R[k + 1] *= scale
    return t, S, E, I, R

# --
# 3) Your best parameters (EDIT THESE)
# --
# Contact rate: beta, incubation rate: sigma (1/latent period),
# recovery rate: gamma (1/infectious period)
best_beta = 0.35   # <-- EDIT with your calibrated value
best_sigma = 1/5.0 # e.g., ~5-day latent period
best_gamma = 1/7.0 # e.g., ~7-day infectious period

# Population and initial conditions
N = N_obs          # or replace with a known N
S0 = S_obs[0]
I0 = I_obs[0]
R0 = R_obs[0]
E0 = 0.0           # <-- If you have an estimate, set it (e.g., a few times I0)

# Time settings
dt = 0.25          # smaller dt improves stability/smoothness
days_fit = int(days_obs.max())  # length matching your data
days_forecast = 300             # run long enough to see peak clearly

# --
# 4) Fit window: simulate over data span for SSE
# --
t_fit, S_fit, E_fit, I_fit, R_fit = euler_seir(
    best_beta, best_sigma, best_gamma, S0, E0, I0, R0, N, days_fit, dt=dt)

# Interpolate model to integer days to compare to observations
# (Because dt might be fractional)
def sample_at_integer_days(t, y):
    """Return y sampled at integer times (0, 1, 2, ..., floor(t[-1])) via nearest index."""
    # since dt is uniform, nearest index is round(day/dt)
    n_days = int(np.floor(t[-1]))
    idx = np.clip(np.round(np.arange(n_days + 1) / (t[1] - t[0])).astype(int), 0, len(t) - 1)
    return y[idx]

I_fit_sampled = sample_at_integer_days(t_fit, I_fit)

# Ensure alignment with available observations (in case CSV starts at nonzero)
max_len = min(len(I_fit_sampled), len(I_obs))
I_fit_aligned = I_fit_sampled[:max_len]
I_obs_aligned = I_obs[:max_len]
days_aligned = days_obs[:max_len]
SSE = float(np.sum((I_fit_aligned - I_obs_aligned) ** 2))

# --
# 5) Long run for peak prediction
# --
t_long, S_long, E_long, I_long, R_long = euler_seir(    
    best_beta, best_sigma, best_gamma, S0, E0, I0, R0, N, days_forecast, dt=dt
)

# Locate the peak of I(t) over the long run
peak_idx = int(np.argmax(I_long))
peak_I = float(I_long[peak_idx])
peak_day = float(t_long[peak_idx])  
# may be fractional due to dt

# --
# 6) Plots
# --

plt.figure(figsize=(10, 6))
plt.plot(days_obs, S_obs, "C0o", alpha=0.5, label="Observed S (data)")
plt.plot(days_obs, I_obs, "C1o", alpha=0.7, label="Observed I (data)")
plt.plot(days_obs, R_obs, "C2o", alpha=0.5, label="Observed R (data)")
plt.plot(t_long, S_long, "C0-", label="Model S (SEIR, Euler)")
plt.plot(t_long, I_long, "C1-", label="Model I (SEIR, Euler)")
plt.plot(t_long, R_long, "C2-", label="Model R (SEIR, Euler)")
plt.axvline(peak_day, color="C1", ls="--", alpha=0.6, label=f"Peak I @ day {peak_day:.1f}")
plt.xlabel("Day")
plt.ylabel("Population Count")
plt.title("SEIR (Euler) vs Observed Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# --
# 7) Console summary
# --
print("\n=== SEIR (Euler) Summary ===")
print(f"Parameters: beta={best_beta:.6g}, sigma={best_sigma:.6g}, gamma={best_gamma:.6g}")
print(f"Population N used: {N:,.0f}")
print(f"Fit window days: 0–{days_fit} (dt={dt})")
print(f"SSE on I(t): {SSE:,.3f}")
print(f"Peak Infected I*: {peak_I:,.0f} individuals")
print(f"Peak day (from day 0 in CSV): {peak_day:.2f}")
print("Is that peak magnitude reasonable for your N and context?")
