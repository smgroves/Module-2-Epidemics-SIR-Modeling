#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv(r'C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time

plt.scatter(data['day'], data['active reported daily cases'])
plt.title("Day vs Active Reported Daily Cases")
plt.xlabel("Day")
plt.ylabel("Active Reported Daily Cases")
plt.show()
# %%
