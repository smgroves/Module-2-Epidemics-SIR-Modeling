#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv(r'C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.
params, covariance = curve_fit(exponential_growth, data['day'], data['active reported daily cases'], p0=[0.1])
r_fit = params[0]

# Approximate R0 using this fit
D = 2
R0_estimated = 1 + r_fit * D
fit_curve = exponential_growth(data['day'],r_fit)

# Add the fit as a line on top of your scatterplot.
plt.scatter(data['day'], data['active reported daily cases'])
plt.plot(data['day'], fit_curve, color = 'red', label = 'Exponential Fit')
plt.text(
    0.05, 0.95,
    f"$R_0$ = {R0_estimated:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top')
plt.title("Exponential Growth Fit of Day vs Active Reported Daily Cases")
plt.xlabel("Day")
plt.ylabel("Active Reported Daily Cases")
plt.show()

# The estimated R0 value is 1.24.