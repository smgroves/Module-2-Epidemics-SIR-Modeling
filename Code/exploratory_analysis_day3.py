#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv(r'/Users/priya/Documents/Comp_Bio/GitHub/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#2.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.
#define data 
x_values = data['day']
y_values = data['active reported daily cases']
popt, pcov = curve_fit(exponential_growth, x_values, y_values) 

# Approximate R0 using this fit
r = popt[0]
D= 2
R0 = np.exp(r * D)
beta_guess = R0 / D
print(f"Estimated growth rate (r): {r}")
print(f"Estimated R0: {R0}")
# Add the fit as a line on top of your scatterplot.

'''
r = 0.12144
R0 = 1.274
What viruses have a similar R0? Use the viruses.html file to find a virus or 2 with a similar R0 and give a 1-2 sentence background of the diseases.
Influenza - seasonal with an R0 of 1.3
How accurate do you think your R0 estimate is?
This estimate is pretty accurate given the r value is 0.12144, which is close to 0. 
'''
#Euler's Pseudo-code 
'''
INPUTS: beta, sigma, gamma, S0, E0, I0, R0, timepoints, N
Initialize S, E, I, and R as empty arrays or lists
Set first item in each list equal to initial values S0, E0, I0, R0
 '''
def seir_euler(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))
    #add initial value to first index of each list
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    #for each timepoint in timepoints: 
    for t in range(len(timepoints)-1):
        #Calculate the four derivatives at timepoint
        dS = - (beta * S[t] * I[t]) / N
        dE = (beta * S[t] * I[t]) / N - (sigma * E[t])
        dI = (sigma * E[t]) - (gamma * I[t])
        dR = (gamma * I[t])
      #  Calculate S, E, I, and R at timepoint + 1 using Euler’s method
        S[t+1] = S[t] + dS * dt
        E[t+1] = E[t] + dE * dt
        I[t+1] = I[t] + dI * dt
        R[t+1] = R[t] + dR * dt  
    return S, E, I, R
#Grid search optimization 
#INPUTS: timepoints, N, S0, E0, I0, R0, data
N = 100000 #assumption about population size 
S0 = N-41
E0 = 40
I0 = 1
R0_init = 0
#Initialize a range for beta, sigma, and gamma
beta_range = np.linspace(0.1, 1.5, 10) #beta = R0 / D = 1.274 / 2 = 0.637, so we want to search around that value
sigma_range = np.linspace(0.055, 0.09, 5) #based on incubation period of 12-18 days, so sigma = 1 / incubation period = 1/12 to 1/18
gamma_range = np.linspace(0.1, 0.5, 10) #recovery rate = 1/2 = 0.5
#Initialize an empty array of SSE
best_sse = float('inf')
best_params = (0,0,0)
#Make arrays of values given each range for each parameter
for b in beta_range:
    for g in gamma_range:
        for s in sigma_range:
#Use the Euler method function you developed to calculate S, E, I, and R given those parameters
            S, E, I, R = seir_euler(b, s, g, S0, E0, I0, R0_init, x_values, N)
#Calculate the SSE given the model results and the data and append this to the SSE array	
            sse = np.sum((I - y_values)**2)
            if sse < best_sse:
                best_sse = sse
                best_params = (b, s, g)
#Determine parameters corresponding to lowest SSE
#Return best_beta, best_sigma, and best_gamma and corresponding SSE
best_beta, best_sigma, best_gamma = best_params
print(f"Best beta: {best_beta}, Best sigma: {best_sigma}, Best gamma: {best_gamma}, SSE: {best_sse}")
#predicting future 
future_timepoints = np.arange(0, 150, 1)
S_f, E_f, I_f, R_f = seir_euler(best_beta, best_sigma, best_gamma, S0, E0, I0, R0_init, future_timepoints, N)
peak_idx = np.argmax(I_f)
peak_day = future_timepoints[peak_idx]
peak_cases = I_f[peak_idx]

print(f"Predicted peak day: {peak_day}")
print(f"Predicted peak cases: {peak_cases}")
plt.plot(x_values, exponential_growth(x_values, *popt), 'r-', label='Fitted curve')
plt.plot(x_values, y_values, marker='o')
plt.plot(future_timepoints, I_f, 'b-', label='SEIR Prediction')
plt.title('Active Cases of Mystery Virus Over Time') 
plt.xlabel('Day') 
plt.ylabel('Number of Active Cases') 
plt.legend() 
plt.show() 
'''Use the Euler method to run the model out many more days until the data peaks
How high is the peak? Is that a reasonable value?
The peak is around 4182.406 cases, which seems reasonable given the data we have so far.
What day will the peak occur?
The peak will occur around day 85
'''