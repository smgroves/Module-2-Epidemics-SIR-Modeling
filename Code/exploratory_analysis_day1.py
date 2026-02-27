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
# %%
