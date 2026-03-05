#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#%%
# Load the data
# find the folder where the script is located
HERE = Path(__file__).parent

# build path to the csv
csv_path = HERE / "mystery_virus_daily_active_counts_RELEASE#1.csv"

data = pd.read_csv(csv_path, parse_dates=['date'])

#%%
# Make a plot of the active cases over time
plt.scatter(data['day'], data['active reported daily cases'])
plt.title('Day vs Active Infections')
plt.xlabel('Day')
plt.ylabel('Active Infections')
plt.show()

# %%
