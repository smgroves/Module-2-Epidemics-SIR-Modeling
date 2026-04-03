#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv('/Users/megansullivan/Desktop/Comp BME/Module-2-epidemics-SIR-modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r, A):
    return A * np.exp(r * t)

day_data = data['day'].values
active_cases_data = data['active reported daily cases'].values

# Fit the exponential growth model to the data. 
popt, pcov = curve_fit(exponential_growth, day_data, active_cases_data, p0=[0.1, 1])

r_fit, A_fit = popt

D = 2 # Assuming an infectious period of 10 days, we can calculate R0 using the formula R0 = 1 + (r * D)

# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.
# Approximate R0 using this fit
r_val = popt[0]
r = 1 +(r_val * D)
print(f"Estimated R0: {r:.2f}")
# Add the fit as a line on top of your scatterplot.

#question 1: What viruses have a similar R0? Use the viruses.html file to find a virus or 2 with a similar R0 and give a 1-2 sentence background of the diseases.
# Influenza (seasonal), influenza (H1N1 2009), Lassa Fever, Marburg, Rabies, and Rhinovirus have similar R0 values.
# Influenza (seasonal) is a contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs. It can cause mild to severe illness and can lead to hospitalization and even death in some cases, particularly in young children, elderly individuals, and those with certain underlying health conditions.
# Influenza (H1N1 2009) is a strain of the influenza virus that caused a global pandemic in 2009. It is a novel strain that emerged from a combination of human, swine, and avian influenza viruses. The H1N1 2009 virus caused widespread illness and resulted in a significant number of hospitalizations and deaths worldwide, particularly among younger individuals and those with underlying health conditions.
# Lassa Fever is an acute viral hemorrhagic illness caused by the Lassa virus, which is transmitted to humans through contact with food or household items contaminated with rodent urine or feces. It is endemic in parts of West Africa and can cause a range of symptoms, from mild to severe, including fever, weakness, and in some cases, bleeding. Severe cases can lead to death if not treated promptly.
# Marburg virus disease is a severe and often fatal illness caused by the Marburg virus,
# Rabies is a viral disease that causes inflammation of the brain in humans and other mammals. It is typically transmitted through the bite of an infected animal, such as a dog or bat. Rabies is almost always fatal once symptoms appear, but it can be prevented through prompt medical treatment after exposure.
# Rhinovirus is a common viral infectious agent that primarily causes the common cold. It is highly
#question 2: How accurate do you think your R0 estimate is?
# I think the R0 estimate is a fairly accurate measure, but since it was measured by a line-fit test, there is certainly room for error. However, the data set was thorough and contained many data points, which should help to improve the accuracy of the estimate. Additionally, the R0 value is an average measure and can vary based on factors such as population density, social behavior, and public health interventions, so it may not capture all the nuances of the virus's transmission dynamics.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SET UP PARAMETERS & DATA ---
N = 17900
S0 = N - 5       # Initial Susceptible [cite: 250]
E0 = 5            # Initial Exposed [cite: 250]
I0 = 1            # Initial Infectious [cite: 250]
R0_comp = 0       # Initial Recovered [cite: 250]
h = 0.1           # Time step (step size) [cite: 242, 246]

# Load your local data
# Updated surveillance includes daily case counts through day 70 
data_df = pd.read_csv('/Users/megansullivan/Desktop/Comp BME/Module-2-epidemics-SIR-modeling/Data/mystery_virus_daily_active_counts_RELEASE#2.csv')
day_data = data_df['day'].values
obs_i = data_df['active reported daily cases'].values

# --- 2. THE EULER METHOD (SEIR) ---
def euler_seir(timepoints, N, S0, E0, I0, R0_init, beta, sigma, gamma):
    """
    Approximates SEIR using Euler's Method: y_{i+1} = y_i + f(t_i, y_i) * h
    Based on Lecture Pseudocode[cite: 245, 247, 248].
    """
    # Initialize S, E, I, and R as arrays [cite: 249, 250]
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))
    
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0_init
    
    for i in range(len(timepoints) - 1):
        # Calculate the four derivatives at current timepoint [cite: 252]
        # Equations: dS/dt, dE/dt, dI/dt, dR/dt 
        dSdt = -beta * S[i] * I[i] / N
        dEdt = (beta * S[i] * I[i] / N) - (sigma * E[i])
        dIdt = (sigma * E[i]) - (gamma * I[i])
        dRdt = gamma * I[i]
        
        # Calculate next values using Euler's formula [cite: 246, 253]
        S[i+1] = S[i] + dSdt * h
        E[i+1] = E[i] + dEdt * h
        I[i+1] = I[i] + dIdt * h
        R[i+1] = R[i] + dRdt * h
        
    return S, E, I, R

# --- 3. GRID SEARCH CALIBRATION ---
# Use the incubation period (12-18 days) to inform sigma [cite: 166, 282]
# sigma = 1 / incubation_period
sigma_vals = np.linspace(1/18, 1/12, 5) # 5 values between 1/18 and 1/12
beta_vals = np.linspace(0.4, 0.8, 10)  # Adjusted range to capture the Day 80-ish peak [cite: 288]
gamma_vals = np.linspace(0.05, 0.25, 5)  # Adjusted range to capture the Day 80-ish peak [cite: 288]

best_sse = float('inf') # Initialize best sum of squared errors to infinity
best_params = {} # Dictionary to store best parameters

# Timepoints for the data length
t_eval = np.arange(0, max(day_data) + 1, h)

for b in beta_vals:
    for s in sigma_vals:
        for g in gamma_vals:
            S, E, I, R = euler_seir(t_eval, N, S0, E0, I0, R0_comp, b, s, g)
            
            # Match model 'I' to the specific days in day_data [cite: 279]
            # Since h=0.1, index = day / h
            model_i_at_data_days = I[(day_data / h).astype(int)]
            sse = np.sum((model_i_at_data_days - obs_i)**2)
            
            if sse < best_sse:
                best_sse = sse
                best_params = {'beta': b, 'sigma': s, 'gamma': g}

# --- 4. PREDICT THE PEAK ---
# Run model many more days until it peaks 
extended_t = np.arange(0, 150, h)
S_f, E_f, I_f, R_f = euler_seir(extended_t, N, S0, E0, I0, R0_comp, **best_params)

peak_val = np.max(I_f)
peak_day = extended_t[np.argmax(I_f)]

print(f"Best Parameters: {best_params}")
print(f"Peak Height: {peak_val:.0f} cases at Day {peak_day:.1f}") # [cite: 288]

# --- 5. VISUALIZATION ---
plt.figure(figsize=(10, 5))
plt.scatter(day_data, obs_i, label='Actual Data', color='red')
plt.plot(extended_t, I_f, label='Model Prediction (I)', color='blue')
plt.axvline(peak_day, linestyle='--', color='gray', label='Predicted Peak')
plt.xlabel("Days")
plt.ylabel("Active Cases")
plt.legend()
plt.show()

data3_df = pd.read_csv('/Users/megansullivan/Desktop/Comp BME/Module-2-epidemics-SIR-modeling/Data/mystery_virus_daily_active_counts_RELEASE#3.csv')
day3_data = data3_df['day'].values
obs_i = data3_df['active reported daily cases'].values

# Actual Peak for error calculation
actual_peak_cases = np.max(obs_i)
actual_peak_day = day3_data[np.argmax(obs_i)]

# --- 2. EULER METHOD (from Lecture Screenshots) ---
def euler_seir(t_max, h, N, S0, E0, I0, R0_val, beta, sigma, gamma):
    timepoints = np.arange(0, t_max + h, h)
    S, E, I, R = np.zeros(len(timepoints)), np.zeros(len(timepoints)), np.zeros(len(timepoints)), np.zeros(len(timepoints))
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0_val
    for n in range(len(timepoints) - 1):
        dS = (-beta * S[n] * I[n] / N)
        dE = (beta * S[n] * I[n] / N) - (sigma * E[n])
        dI = (sigma * E[n]) - (gamma * I[n])
        dR = (gamma * I[n])
        S[n+1] = S[n] + dS * h
        E[n+1] = E[n] + dE * h
        I[n+1] = I[n] + dI * h
        R[n+1] = R[n] + dR * h
    return timepoints, I

# --- 3. GRID SEARCH CALIBRATION ---
# Ranges adjusted to capture the Day 80-ish peak
beta_range = np.linspace(0.3, 0.5, 12)
sigma_range = np.linspace(1/15, 1/10, 4)
gamma_range = np.linspace(0.08, 0.15, 4)

best_sse = float('inf')
best_params = (0,0,0)

for b in beta_range:
    for s in sigma_range:
        for g in gamma_range:
            t, I_model = euler_seir(max(day3_data), h, N, S0, E0, I0, 1.24, b, s, g)
            model_matches = I_model[(day3_data / h).astype(int)]
            sse = np.sum((model_matches - obs_i)**2)
            if sse < best_sse:
                best_sse = sse
                best_params = (b, s, g)

# --- 4. PREDICT & CALCULATE ERROR ---
t_final, I_final = euler_seir(120, h, N, S0, E0, I0, 1.24, *best_params)
model_peak_cases = np.max(I_final)
model_peak_day = t_final[np.argmax(I_final)]

error_y = abs(model_peak_cases - actual_peak_cases)
error_x = abs(model_peak_day - actual_peak_day)

print(f"--- CALIBRATION RESULTS ---")
print(f"Best Beta: {best_params[0]:.4f}, Best Sigma: {best_params[1]:.4f}, Best Gamma: {best_params[2]:.4f}")
print(f"Model Peak: {model_peak_cases:.0f} cases on Day {model_peak_day:.1f}")
print(f"Actual Peak: {actual_peak_cases} cases on Day {actual_peak_day}")
print(f"ERROR IN Y (Cases): {error_y:.1f}")
print(f"ERROR IN X (Days): {error_x:.1f}")

# --- 5. VISUALIZE ---
plt.figure(figsize=(10,6))
plt.scatter(day3_data, obs_i, label='Actual Data', color='red', s=15, alpha=0.5)
plt.plot(t_final, I_final, label='Calibrated SEIR Model', color='blue', linewidth=2)
plt.title("Mystery Virus: Actual Data vs. SEIR Model")
plt.xlabel("Days")
plt.ylabel("Active Cases")
plt.legend()
plt.show()