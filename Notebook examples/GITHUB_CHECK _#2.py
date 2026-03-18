<<<<<<< Updated upstream
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LOAD DATA
data = pd.read_csv('Data/mystery_virus_daily_active_counts_RELEASE#3.csv', parse_dates=['date'], header=0, index_col=None)

t_data = data['day'].values
y_data = data['active reported daily cases'].values

# CONSTANTS
N = 17900
R0_daily = 1.1209  # Growth factor calculated from data
r = np.log(R0_daily) 
I0_start = y_data[0]

### 2c. Use Euler's method to solve the SEIR model.
def seir_euler(beta, sigma, gamma, S0, E0, I0, days, N, dt=0.1):
    steps = int(days / dt)
    S, E, I, R = [S0], [E0], [I0], [0.0]
    T = np.linspace(0, days, steps + 1)
    
    for i in range(steps):
        s_c, e_c, i_c, r_c = S[-1], E[-1], I[-1], R[-1]
        
        # Differential equations
        dS = -beta * s_c * i_c / N
        dE = (beta * s_c * i_c / N) - (sigma * e_c)
        dI = (sigma * e_c) - (gamma * i_c)
        dR = (gamma * i_c)
        
        # Update values
        S.append(s_c + dS * dt)
        E.append(e_c + dE * dt)
        I.append(i_c + dI * dt)
        R.append(r_c + dR * dt)
        
    # Interpolate back to daily integers for comparison
    I_daily = np.interp(np.arange(1, days + 1), T, I)
    return I_daily

### 2d. Fit the SEIR model to the data by changing beta, gamma, and sigma.
def grid_search_seir(timepoints, N, S0, E0, I0, R0, data, 
                     beta_range, sigma_range, gamma_range):
    best_SSE = float("inf")
    best_params = None
    r_val = np.log(R0)
    
    for b in beta_range:
        for s in sigma_range:
            for g in gamma_range:
                # Calculate E0 based on the growth rate to stay consistent with early data
                E0_iter = ((r_val + g) / s) * I0
                S0_iter = N - I0 - E0_iter
                
                # Run Euler 
                I_pred = seir_euler(b, s, g, S0_iter, E0_iter, I0, int(timepoints[-1]), N)
                
                # Calculate SSE
                SSE = np.sum((I_pred - data)**2)
                
                if SSE < best_SSE:
                    best_SSE = SSE
                    best_params = (b, s, g, E0_iter)
                    
    return best_params, best_SSE

# RUN GRID SEARCH
beta_space = np.linspace(0.4, 0.9, 20)
sigma_space = np.linspace(1/18, 1/12, 10)
gamma_space = np.linspace(1/11, 1/7, 10)

best_p, min_sse = grid_search_seir(t_data, N, 0, 0, I0_start, R0_daily, y_data,
                                   beta_space, sigma_space, gamma_space)

b_opt, s_opt, g_opt, e0_opt = best_p
s0_opt = N - I0_start - e0_opt

### 2e. Plot the model-predicted infections over time compared to the data.
# Extended simulation to 100 days to see the peak
t_proj_limit = 100
I_proj = seir_euler(b_opt, s_opt, g_opt, s0_opt, e0_opt, I0_start, t_proj_limit, N)
t_proj_days = np.arange(1, len(I_proj) + 1)

plt.figure(figsize=(10, 6))
plt.scatter(t_data, y_data, color='black', label='Reported Data', s=15, alpha=0.7)
plt.plot(t_proj_days, I_proj, color='red', label='Optimized SEIR Model', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Active Infectious Cases')
plt.title('SEIR Model Fit and Epidemic Projection')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

### 2e. Predict the day and amount of active cases at the peak of the epidemic spread.
peak_idx = np.argmax(I_proj)
peak_day = t_proj_days[peak_idx]
peak_cases = I_proj[peak_idx]
 
print(f"--- Results ---")
print("Best beta:", b_opt)
print("Best sigma:", s_opt)
print("Best gamma:", g_opt)
print("Estimated E0:", e0_opt)
print(f"Minimum SSE: {min_sse:.2f}")
print(f"Predicted Peak Day: Day {peak_day}")
print(f"Predicted Peak Active Cases: {int(peak_cases)} cases")
=======
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Load data
data = pd.read_csv("mystery_virus_daily_active_counts_RELEASE#3.csv")

t_data = data["day"].values
I_data = data["active reported daily cases"].values

# Parameters
N = 17900
beta = 0.69
gamma = 0.13

# Initial conditions
I0 = I_data[0]
S0 = N - I0
R0 = 0

y0 = [S0, I0, R0]

# SIR model
def sir_model(t, y):
    S, I, R = y

    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I

    return [dS, dI, dR]

# Solve the system
sol = solve_ivp(
    sir_model,
    [t_data[0], t_data[-1]],
    y0,
    t_eval=t_data
)

# Extract model infected values
I_model = sol.y[1]

# Compute SSE
SSE = np.sum((I_data - I_model)**2)

print("SSE =", SSE)
>>>>>>> Stashed changes
