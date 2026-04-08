# importing the necessary libraries to run the code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import numpy as np

data3_df = pd.read_csv(r'C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#3.csv') # Load the new data set for error calculation
day3_data = data3_df['day'].values # Extracting the 'day' column as a NumPy array
obs_i = data3_df['active reported daily cases'].values # Extracting the 'active reported daily cases' column as a NumPy array

N = 17900 # UVA population size [cite: 250]
S0 = N - 5
E0 = 5
I0 = 1
R0_init = 0
h = 1  # 1 day per step

def euler_seir(t_max, N, S0, E0, I0, R0_val, beta, sigma, gamma, intervention=None): # The intervention parameter allows us to specify which intervention to simulate, and the model will adjust beta and/or gamma accordingly at the specified time points.
    timepoints = np.arange(0, t_max + 1, 1) # Create an array of time points from 0 to t_max with a step of 1 day
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0_val # Initialize the first values of S, E, I, R based on the initial conditions

    for n in range(len(timepoints)-1): # Loop through each time point to calculate the next values of S, E, I, R
        t = timepoints[n] # Current time point
        beta_t = beta # Initialize beta_t to the base beta value, which can be modified by interventions
        gamma_t = gamma # Initialize gamma_t to the base gamma value, which can be modified by interventions

        # --- INTERVENTIONS ---
        # The following code modifies beta and gamma based on the specified intervention and the current time point. Each intervention has specific conditions for when it takes effect and how it alters the parameters of the model.
        if intervention == "masking early" and t >= 40: # Masking starts on day 40 and reduces transmission by 40%
            beta_t = beta * 0.6  # 40% reduction in transmission

        if intervention == "quarantine early" and t >= 40: # Testing and quarantine starts on day 40, reducing the infectious period by 2 days for detected cases
            # Reduce infectious period by 2 days: gamma_new = 1/(original infectious period -2)
            gamma_t = 1 / (1/gamma - 2) # Adjust gamma to reflect the reduced infectious period


        # --- SEIR EQUATIONS ---
        dS = -beta_t * S[n] * I[n] / N # Change in susceptible population based on the current beta_t, S, I, and total population N
        dE = beta_t * S[n] * I[n] / N - sigma * E[n] # Change in exposed population based on new infections and progression to infectious
        dI = sigma * E[n] - gamma_t * I[n] # Change in infectious population based on progression from exposed and recovery
        dR = gamma_t * I[n] # Change in recovered population based on recovery from infectious

        S[n+1] = S[n] + dS
        E[n+1] = E[n] + dE
        I[n+1] = I[n] + dI
        R[n+1] = R[n] + dR

    return timepoints, S, E, I, R # Return timepoints along with S, E, I, R for plotting and analysis

#%%
# --- GRID SEARCH CALIBRATION ---
# (Google Gemini, 2026) Google Gemini was used to help construct the grid search calibration process, including the ranges for beta, sigma, and gamma, as well as the calculation of the sum of squared errors (SSE) to evaluate model fit.
beta_range = np.linspace(0.3,0.5,12) # Adjusted range to capture the Day 80-ish peak [cite: 288]
sigma_range = np.linspace(1/15,1/10,4) # Use the incubation period (12-18 days) to inform sigma [cite: 166, 282]
gamma_range = np.linspace(0.08,0.15,4) # Adjusted range to capture the Day 80-ish peak [cite: 288]

best_sse = float('inf') # Initialize best sum of squared errors to infinity
best_params = (0,0,0) # Initialize best parameters to a default value
t_max = max(day3_data) # Time range for calibration based on the length of the new data set for error calculation

for b in beta_range: # Loop through each beta value
    for s in sigma_range: # Loop through each sigma value
        for g in gamma_range: # Loop through each gamma value
            t, S, E, I_model, R = euler_seir(t_max, N, S0, E0, I0, R0_init, b, s, g) # Run the SEIR model with the current set of parameters
            model_matches = I_model[day3_data.astype(int)] # Match model 'I' to the specific days in day3_data
            sse = np.sum((model_matches - obs_i)**2) # Calculate the sum of squared errors (SSE) between model predictions and actual data
            if sse < best_sse: # If the current SSE is better than the best SSE, update best_sse and best_params
                best_sse = sse 
                best_params = (b, s, g)

b, s, g = best_params
print(f"Best parameters (beta, sigma, gamma): ({b:.4f}, {s:.4f}, {g:.4f})")

#%%
# --- BASELINE MODEL ---
# (ChatGPT, 2026) ChatGPT was used to help construct the code to run the baseline SEIR model using the best parameters obtained from the grid search calibration, which will serve as a reference for comparing the effects of various interventions.
t_base, S_b, E_b, I_b, R_b = euler_seir(120, N, S0, E0, I0, R0_init, *best_params)

#%%
# --- INTERVENTION SIMULATIONS ---
# (ChatGPT, 2026) ChatGPT was used to help construct the code to simulate the various interventions using the SEIR model with Euler's method, allowing us to compare the effects of each intervention on the number of active cases over time.
target_intervention = ["masking early"] # List of interventions to simulate
results = {} # Dictionary to store the results of each intervention simulation

for intervention in target_intervention: # Loop through each intervention and run the SEIR model with the specified intervention to get the trajectory of S, E, I, R over time, and store the number of active cases (I) for each intervention in the results dictionary for later comparison.
    t_i, S_i, E_i, I_i, R_i = euler_seir(120, N, S0, E0, I0, R0_init, *best_params, intervention=intervention) # Run the SEIR model with the specified intervention
    results[intervention] = I_i # Store the number of active cases (I) for each intervention in the results dictionary

#%%
# --- VISUALIZATION ---
# (ChatGPT, 2026) ChatGPT was used to help construct the code to visualize the results of the baseline model and each intervention on a single plot, allowing us to compare the trajectories of active cases over time for each scenario, with a vertical line indicating when interventions start.
plt.figure(figsize=(12,6))
plt.plot(t_base, I_b, label="Baseline", color = 'blue', linewidth=3)
# Expanded color list to accommodate the 7 interventions now in the list

plt.plot(t_base, results[intervention], label=intervention.replace("_"," ").title(), color='red')
plt.axvline(40, color='blue', linestyle=':', label="Early intervention start") # Added vertical line for Day 40
plt.xlabel("Days")
plt.ylabel("Active Cases")
plt.title("SEIR Model: Comparison of Interventions (h=1)")
plt.legend()
plt.show() # Show the plot comparing the baseline model and each intervention, with appropriate labels and a title for clarity.
