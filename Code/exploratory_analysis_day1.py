#%%
import pandas as pd
import matplotlib.pyplot as plt


# Load data
data = pd.read_csv('../Data/mystery_virus_daily_active_counts_RELEASE#1.csv',
                   parse_dates=['date'],
                   header=0,
                   index_col=None)

# Rename long column
data = data.rename(columns={'active reported daily cases': 'active'})

# Plot infections over time
plt.figure()
plt.plot(data['day'], data['active'])

plt.xlabel('Day')
plt.ylabel('Active Infections')
plt.title('Mystery Virus: Active Infections Over Time')

plt.show()

#  QUESTIONS:
# The initial infections start out really slow, but after about three weeks the growth becomes exponential and the curve just takes off.

# We can measure the speed by looking at the growth rate and R0, which shows that each person is infecting more than one other person on average.

# To predict the full curve, it would be helpful to know the infectious period and how many people are asymptomatic or already immune.
# %%
