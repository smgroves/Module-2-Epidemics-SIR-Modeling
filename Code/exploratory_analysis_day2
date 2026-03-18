import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load the data
data = pd.read_csv('Data/mystery_virus_daily_active_counts_RELEASE#2.csv', parse_dates=['date'], header=0, index_col=None)

t_data = data['day'].values
y_data = data['active reported daily cases'].values
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.

#parameters optizmized and parameter covariance
popt, pcov = curve_fit(exponential_growth, t_data, y_data)
r_fit = popt[0]
# Approximate R0 using this fit
R0_daily = np.exp(r_fit)

print(f"Growth Rate (r): {r_fit:.4f}")
print(f"Approximate Daily R0: {R0_daily:.4f}")

# Add the fit as a line on top of your scatterplot.

plt.figure(figsize=(10, 6))
plt.scatter(t_data, y_data, label='Reported Data', color='blue', alpha=0.5)
plt.plot(t_data, exponential_growth(t_data, r_fit), label='Exponential Fit', color='red', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Active Reported Daily Cases')
plt.title('Mystery Virus Growth Analysis')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()