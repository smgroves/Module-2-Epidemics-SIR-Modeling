#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv(r'/Users/priya/Documents/Comp_Bio/GitHub/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time
x_values = data['date']
y_values = data['active reported daily cases']
plt.plot(x_values, y_values, marker='o')
plt.title('Active Cases of Mystery Virus Over Time') 
plt.xlabel('Date') 
plt.ylabel('Number of Active Cases') 
plt.show() 
'''What do you notice about the initial infections? Seems to be following an exponential growth pattern 
• How could we measure how quickly its spreading? Find the rate of growth of the curve 
• What information about the virus would be helpful in determining the shape of the
outbreak curve? 
'''