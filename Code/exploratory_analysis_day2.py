#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv(r'C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#3.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.
params, covariance = curve_fit(exponential_growth, data['day'], data['active reported daily cases'], p0=[0.1]) #fitting the exponential model to our data set
r_fit = params[0] #pulling out the growth rate to allow for further calculations later on 

# Approximate R0 using this fit
D = 2 # defining the infectious period to match our data
R0_estimated = 1 + r_fit * D #approximating R0 using the formula 
fit_curve = exponential_growth(data['day'],r_fit) #generates the predicated case counts for plotting

# Add the fit as a line on top of your scatterplot.
plt.scatter(data['day'], data['active reported daily cases']) # plotting the raw data points as a scatter plot
plt.plot(data['day'], fit_curve, color = 'red', label = 'Exponential Fit') # fitting the exponential curve on top of our scatter plot
plt.text(
    0.05, 0.95,
    f"$R_0$ = {R0_estimated:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top') # adding and positioning the text to show our R0 value 
plt.title("Exponential Growth Fit of Day vs Active Reported Daily Cases") #adding a title to our plot
plt.xlabel("Day") #adding a label to the x-axis
plt.ylabel("Active Reported Daily Cases") #adding a label to the y-axis
plt.show() #showing the plot

# The estimated R0 value is 1.24.

#%% --------------------------------------------------
# Extract data for SEIR model

timepoints = data['day'].values
observed_I = data['active reported daily cases'].values

# Initial conditions
N = 17000

I0 = observed_I[0]
E0 = I0
R0 = 0
S0 = N - I0 - E0 - R0

#%% --------------------------------------------------
# SEIR Euler Method

def SEIR_Euler(beta, sigma, gamma, S0, E0, I0, R0, t_array, N):

    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]

    for t in range(len(t_array) - 1):

        dt = t_array[t+1] - t_array[t]

        S_current = S[-1]
        E_current = E[-1]
        I_current = I[-1]
        R_current = R[-1]

        dS = -beta * S_current * I_current / N
        dE =  (beta * S_current * I_current / N)- (sigma * E_current)
        dI = (sigma * E_current) - (gamma * I_current)
        dR = gamma * I_current

        S.append(S_current + dt * dS)
        E.append(E_current + dt * dE)
        I.append(I_current + dt * dI)
        R.append(R_current + dt * dR)

    return np.array(S), np.array(E), np.array(I), np.array(R)

#%% --------------------------------------------------
# Grid Search for Best Parameters

beta_range = np.linspace(1/18, 1/12, 5)
sigma_range = np.linspace(0.4, 0.8, 10)
gamma_range = np.linspace(0.05, 0.25, 5)

best_SSE = float("inf")

best_beta = None
best_sigma = None
best_gamma = None

for b in beta_range:
    for s in sigma_range:
        for g in gamma_range:

            S, E, I, R = SEIR_Euler(
                b, s, g,
                S0, E0, I0, R0,
                timepoints,
                N
            )

            SSE = np.sum((I - observed_I) ** 2)

            if SSE < best_SSE:
                best_SSE = SSE
                best_beta = b
                best_sigma = s
                best_gamma = g

# Print best parameters
# print("Best beta:", best_beta)
# print("Best sigma:", best_sigma)
# print("Best gamma:", best_gamma)
# print("Best SSE:", best_SSE)
print(f"Optimization Results:\nBeta: {best_beta}\nSigma: {best_sigma}\nGamma: {best_gamma}\nSSE: {best_SSE:.2f}")
#%% --------------------------------------------------
# Run model with best parameters

projection_days = np.arange(0,150,1)

S, E, I, R = SEIR_Euler(
    best_beta,
    best_sigma,
    best_gamma,
    S0, E0, I0, R0,
    projection_days,
    N
)
# Identifying the peak
peak_val = np.max(I)
peak_day = projection_days[np.argmax(I)]

# Plot SEIR model vs data

plt.figure()

plt.scatter(
    data['day'],
    observed_I, color = 'red',
    label="Observed Data"
)

plt.plot(projection_days, I, color='blue', label='Infected (Projected)', linewidth=2)


plt.axvline(peak_day, color='blue', linestyle=':', alpha=0.7)


plt.title("Full SEIR Outbreak Projection")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# masking reduces beta by 40%
# vaccine interventions removes people from susceptible to recovered with campaign
# rollout vaccination removes people from suscpetible to recovered in batches
# testing and quarantine reduces infectious period