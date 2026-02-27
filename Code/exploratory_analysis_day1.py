#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv(r'C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time

plt.scatter(data['day'], data['active reported daily cases']) # plotting the raw data points as a scatter plot
plt.title("Day vs Active Reported Daily Cases") #adding a title to the plot
plt.xlabel("Day") # adding a label to the x-axis
plt.ylabel("Active Reported Daily Cases") # adding a label to the y-axis
plt.show() # showing the plot

# 1. The first thing I notice about the graphs is that the number of active reported daily cases increases exponentially after staying flat for a decent bit in the first few days. The suggests that the virus is spreading slowly at first and then has a very exponential increase in new cases, as opposed to linear growth.
# 2. Some ways we could measure how quickly the disease is spreading is by calculating the growth rate (the percentage of increase in new cases each day), the amount of time it takes cases to double, the slope of the curve at its steepest, and average number of people one infected person transmits the virus to R₀).
# 3. Some helpful information for determining the shape of the outbreak curve would be R₀, the incubation period, the infectious period, how the population is repsonding to the outbreak or any percautions they might be taking, and the recovery rate of the disease.