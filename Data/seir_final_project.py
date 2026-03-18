import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# LOAD DATASETS
# ----------------------------------------------------

data1 = pd.read_csv(
    "Data/mystery_virus_daily_active_counts_RELEASE#1.csv",
    parse_dates=["date"]
)

data2 = pd.read_csv(
    "Data/mystery_virus_daily_active_counts_RELEASE#2.csv",
    parse_dates=["date"]
)

data3 = pd.read_csv(
    "Data/mystery_virus_daily_active_counts_RELEASE#3.csv",
    parse_dates=["date"]
)

t_data1 = data1["day"].values
y_data1 = data1["active reported daily cases"].values

t_data2 = data2["day"].values
y_data2 = data2["active reported daily cases"].values

t_data3 = data3["day"].values
y_data3 = data3["active reported daily cases"].values

# ----------------------------------------------------
# CONSTANTS
# ----------------------------------------------------

N = 17900
R0_daily = 1.1209
r = np.log(R0_daily)

I0_start = y_data2[0]

# ----------------------------------------------------
# SEIR MODEL (Euler Method)
# ----------------------------------------------------

def seir_euler(beta, sigma, gamma, S0, E0, I0, days, N, dt=0.1):

    steps = int(days / dt)

    S, E, I, R = [S0], [E0], [I0], [0.0]

    T = np.linspace(0, days, steps + 1)

    for i in range(steps):

        s_c, e_c, i_c, r_c = S[-1], E[-1], I[-1], R[-1]

        dS = -beta * s_c * i_c / N
        dE = beta * s_c * i_c / N - sigma * e_c
        dI = sigma * e_c - gamma * i_c
        dR = gamma * i_c

        S.append(s_c + dS * dt)
        E.append(e_c + dE * dt)
        I.append(i_c + dI * dt)
        R.append(r_c + dR * dt)

    I_daily = np.interp(np.arange(1, days + 1), T, I)

    return I_daily

# ----------------------------------------------------
# GRID SEARCH PARAMETER FIT
# ----------------------------------------------------

def grid_search_seir(timepoints, N, I0, R0, data,
                     beta_range, sigma_range, gamma_range):

    best_SSE = float("inf")
    best_params = None

    r_val = np.log(R0)

    for b in beta_range:
        for s in sigma_range:
            for g in gamma_range:

                E0_iter = ((r_val + g) / s) * I0
                S0_iter = N - I0 - E0_iter

                I_pred = seir_euler(
                    b, s, g,
                    S0_iter,
                    E0_iter,
                    I0,
                    int(timepoints[-1]),
                    N
                )

                SSE = np.sum((I_pred - data) ** 2)

                if SSE < best_SSE:

                    best_SSE = SSE
                    best_params = (b, s, g, E0_iter)

    return best_params, best_SSE

# ----------------------------------------------------
# RUN PARAMETER FITTING (Dataset #2)
# ----------------------------------------------------

beta_space = np.linspace(0.4, 0.9, 20)
sigma_space = np.linspace(1/18, 1/12, 10)
gamma_space = np.linspace(1/11, 1/7, 10)

best_p, min_sse = grid_search_seir(
    t_data2,
    N,
    I0_start,
    R0_daily,
    y_data2,
    beta_space,
    sigma_space,
    gamma_space
)

b_opt, s_opt, g_opt, e0_opt = best_p
s0_opt = N - I0_start - e0_opt

print("\n--- Best Fit Parameters ---")
print("Beta:", b_opt)
print("Sigma:", s_opt)
print("Gamma:", g_opt)
print("Initial Exposed:", e0_opt)

print("\nTraining SSE:", min_sse)

# ----------------------------------------------------
# VALIDATE MODEL USING DATASET #3
# ----------------------------------------------------

I_validation = seir_euler(
    b_opt,
    s_opt,
    g_opt,
    s0_opt,
    e0_opt,
    I0_start,
    int(t_data3[-1]),
    N
)

# Calculate SSE
SSE_validation = np.sum((I_validation[:len(y_data3)] - y_data3) ** 2)

# Calculate R^2
ss_res = np.sum((y_data3 - I_validation[:len(y_data3)])**2)
ss_tot = np.sum((y_data3 - np.mean(y_data3))**2)

r_squared = 1 - (ss_res / ss_tot)

print("\n--- Validation Results ---")
print("Validation SSE:", SSE_validation)
print("R^2:", r_squared)

# ----------------------------------------------------
# PEAK PREDICTION
# ----------------------------------------------------

t_proj_limit = 120

I_proj = seir_euler(
    b_opt,
    s_opt,
    g_opt,
    s0_opt,
    e0_opt,
    I0_start,
    t_proj_limit,
    N
)

t_proj_days = np.arange(1, len(I_proj) + 1)

peak_idx = np.argmax(I_proj)

peak_day = t_proj_days[peak_idx]
peak_cases = I_proj[peak_idx]

print("\n--- Epidemic Peak Prediction ---")
print("Peak Day:", peak_day)
print("Peak Active Cases:", int(peak_cases))

# ----------------------------------------------------
# VT SIMULATION
# ----------------------------------------------------

VT_population = 38000

S0_vt = VT_population - I0_start - e0_opt

days_pre = 70

I_pre = seir_euler(
    b_opt,
    s_opt,
    g_opt,
    S0_vt,
    e0_opt,
    I0_start,
    days_pre,
    VT_population
)

I70 = I_pre[-1]

E70 = ((np.log(R0_daily) + g_opt) / s_opt) * I70
S70 = VT_population - I70 - E70

# ----------------------------------------------------
# INTERVENTIONS
# ----------------------------------------------------

days_post = 50

I_int1 = seir_euler(b_opt * 0.8, s_opt, g_opt, S70, E70, I70, days_post, VT_population)
I_int2 = seir_euler(b_opt * 0.6, s_opt, g_opt, S70, E70, I70, days_post, VT_population)
I_int3 = seir_euler(b_opt * 0.4, s_opt, g_opt, S70, E70, I70, days_post, VT_population)

I_full_int1 = np.concatenate([I_pre, I_int1])
I_full_int2 = np.concatenate([I_pre, I_int2])
I_full_int3 = np.concatenate([I_pre, I_int3])

# ----------------------------------------------------
# PLOTS
# ----------------------------------------------------

plt.figure(figsize=(10,6))

plt.scatter(t_data3, y_data3, color='black', label="Reported Data", s=15)

plt.plot(t_proj_days, I_proj, color='red', label="SEIR Model")

plt.plot(I_full_int1, label="Intervention 1 (20%)")
plt.plot(I_full_int2, label="Intervention 2 (40%)")
plt.plot(I_full_int3, label="Intervention 3 (60%)")

plt.axvline(x=70, linestyle='--', label="Intervention Start")

plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("SEIR Model and Intervention Scenarios")

plt.legend()
plt.grid(True)

plt.show()



 