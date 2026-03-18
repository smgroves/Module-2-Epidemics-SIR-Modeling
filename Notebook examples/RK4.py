import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---------------------------------
# Load Data
# ---------------------------------
data = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#3.csv")

t_data = data["day"].values
I_data = data["active reported daily cases"].values

# ---------------------------------
# Normalize by population
# ---------------------------------
N = 17900

I_frac = I_data / N

I0 = I_frac[0]
S0 = 1 - I0
R0 = 0

y0 = [S0, I0, R0]

# ---------------------------------
# SIR Differential Equations
# ---------------------------------
def sir_model(t, y, beta, gamma):

    S, I, R = y

    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I

    return [dS, dI, dR]

# ---------------------------------
# Solve SIR using RK4 (solve_ivp)
# ---------------------------------
def solve_sir(beta, gamma):

    sol = solve_ivp(
        lambda t, y: sir_model(t, y, beta, gamma),
        [t_data[0], t_data[-1]],
        y0,
        t_eval=t_data,
        method="RK45"
    )

    return sol.y[1]

# ---------------------------------
# SSE Objective Function
# ---------------------------------
def SSE(params):

    beta, gamma = params

    I_model = solve_sir(beta, gamma)

    return np.sum((I_model - I_frac)**2)

# ---------------------------------
# Optimize beta and gamma
# ---------------------------------
initial_guess = [0.6, 0.2]

result = minimize(SSE, initial_guess, bounds=[(0,5),(0,5)])

beta_opt, gamma_opt = result.x

print("Optimized beta:", beta_opt)
print("Optimized gamma:", gamma_opt)
print("SSE:", result.fun)

# ---------------------------------
# Final SIR Simulation
# ---------------------------------
sol = solve_ivp(
    lambda t, y: sir_model(t, y, beta_opt, gamma_opt),
    [t_data[0], t_data[-1]],
    y0,
    t_eval=t_data,
    method="RK45"
)

S, I, R = sol.y

# ---------------------------------
# Plot SIR Fit
# ---------------------------------
plt.figure(figsize=(8,5))

plt.scatter(t_data, I_frac, color="black", label="Data")
plt.plot(t_data, I, label="SIR RK4 Fit")

plt.xlabel("Day")
plt.ylabel("Infectious Fraction")
plt.title("SIR Model Fit using RK4")
plt.legend()

plt.show()

# =================================================
# SEIR MODEL EXTENSION
# =================================================

sigma = 1/14

E0 = 0.01
S0_seir = 1 - I0 - E0

y0_seir = [S0_seir, E0, I0, 0]

# ---------------------------------
# SEIR Equations
# ---------------------------------
def seir_model(t, y, beta, gamma):

    S, E, I, R = y

    dS = -beta * S * I
    dE = beta * S * I - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    return [dS, dE, dI, dR]

# ---------------------------------
# Solve SEIR
# ---------------------------------
sol_seir = solve_ivp(
    lambda t, y: seir_model(t, y, beta_opt, gamma_opt),
    [t_data[0], t_data[-1]],
    y0_seir,
    t_eval=t_data,
    method="RK45"
)

S_seir, E_seir, I_seir, R_seir = sol_seir.y

# ---------------------------------
# Plot SEIR Result
# ---------------------------------
plt.figure(figsize=(8,5))

plt.scatter(t_data, I_frac, color="black", label="Data")
plt.plot(t_data, I_seir, label="SEIR Model")

plt.xlabel("Day")
plt.ylabel("Infectious Fraction")
plt.title("SEIR Model Simulation")
plt.legend()

plt.show()