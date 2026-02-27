#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv('Data/mystery_virus_daily_active_counts_RELEASE#1.csv',
                   parse_dates=['date'], header=0, index_col=None)
#%%
t = data['day']
I = data['active reported daily cases']
print(data.columns)
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
def exponential_growth(t, a, r):
    return a * np.exp(r * t)

# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
from scipy.optimize import curve_fit
params, covariance = curve_fit(exponential_growth, t, I)

a_est, r_est = params

print("Estimated growth rate r:", r_est)

# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.
# Fit the model to the data
from scipy.optimize import curve_fit

params, covariance = curve_fit(exponential_growth, t, I)

# Extract fitted parameters
a_est, r_est = params

print("Estimated growth rate r:", r_est)
# Approximate R0 using this fit
D = 5
R0 = 1 + r_est * D
print("Approximate R0:", R0)
# Add the fit as a line on top of your scatterplot.
t_smooth = np.linspace(min(t), max(t), 100)
fit_curve = exponential_growth(t_smooth, a_est, r_est)

plt.scatter(t, I, label="Data")
plt.plot(t_smooth, fit_curve, label="Exponential Fit")

plt.xlabel("Day")
plt.ylabel("Active Reported Daily Cases")
plt.legend()
plt.show()

# What viruses have a similar R0? Use the viruses.html file to find a virus or 2 with a similar R0 and give a 1-2 sentence background of the diseases.
# How accurate do you think your R0 estimate is?
