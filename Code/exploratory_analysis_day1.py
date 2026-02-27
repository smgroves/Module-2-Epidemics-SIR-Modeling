#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv('Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time

active_col = data.columns[2]
data[active_col] = pd.to_numeric(data[active_col], errors="coerce")

# Day and active infections
days = data["day"]
active_cases = data[active_col]

# Plot the data
plt.figure()
plt.plot(days, active_cases)
plt.xlabel("Day")
plt.ylabel("Active Infections")
plt.title("Day vs Active Infections (Data Release #1)")
plt.show()

# What do you notice about the initial infections? For the first several days, the amount of infections stayed low, curve a little flat. After that, the curve begins to rise faster - exponential-like growth
# How could we measure how quickly its spreading? (cases today - cases yesterday)/casees yesterday - look at the slope of the curve and how long the cases takes to double 
# What information about the virus would be helpful in determining the shape of the outbreak curve? how easily it spreads, how long it takes before the symptoms appear, how long someone can spread it, reecovery time, immmunity rates, and what is being done to prevent it from further spreading

# %%
