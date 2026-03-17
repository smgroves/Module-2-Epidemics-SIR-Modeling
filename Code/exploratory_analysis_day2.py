#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path


#%%
# Load the data
# find the folder where the script is located
HERE = Path(__file__).parent

# build path to the csv
csv_path = HERE / "mystery_virus_daily_active_counts_RELEASE#1.csv"

data = pd.read_csv(csv_path, parse_dates=['date'])


#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data.
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.
# Approximate R0 using this fit 

# this extracts the day and active cases data to be used in curve fitting 
t = data['day']
cases = data['active reported daily cases']

# this is the curve_fit function which takes in the function we want to fit (exponential growth), x_data (t), and y_data (active cases)
est_params, covariance = curve_fit(exponential_growth, t, cases)
#curve_fit returns the estimated parameters (est_params) which is growth rate r0 and the covariance

est_r = est_params[0] #this is the estimated growth rate r0 from the curve fitting
print(f"Estimated growth rate (r): {est_r:.4f} per day")

t_fit = np.linspace(min(t), max(t), 100) # this creates the fitted curve to plot for x_data (days)
cases_fit = exponential_growth(t_fit, est_r) # this finds the y_data (active cases) for the fitted curve

D = 9 #9 days is infectious period 
R0 = 1 + est_r * D
print(f"Estimated R0: {R0:.4f}")
# Add the fit as a line on top of your scatterplot.
plt.scatter(t, cases, label='Data')
plt.plot(t_fit, cases_fit, color='red', label='Exponential Fit')
plt.title('Day vs Active Infections with Exponential Fit')
plt.xlabel('Day')
plt.ylabel('Active Infections')
plt.legend()
plt.show()



# %%
