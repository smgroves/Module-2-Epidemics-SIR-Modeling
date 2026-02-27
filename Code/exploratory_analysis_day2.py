import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 1. Load and Rename (to prevent KeyErrors later)
data = pd.read_csv('Data/mystery_virus_daily_active_counts_RELEASE#2.csv')
data = data.rename(columns={'active reported daily cases': 'active'})

# 2. Slice early phase
early_data = data.iloc[0:45].copy() # .copy() prevents warnings

# 3. Calculate Log and Regression
# We add a small check to ensure no zero values exist before log
log_I = np.log(early_data['active'].replace(0, np.nan).dropna())
slope, intercept, r_value, p_value, std_err = linregress(early_data['day'], log_I)

r = slope
D = 2
R0 = 1 + r * D


print(f"--- Results ---")
print(f"Growth rate (r): {r:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"Estimated R0: {R0:.4f}")

# 4. Plotting
plt.figure(figsize=(8, 5))
plt.scatter(early_data['day'], early_data['active'], label='Data')
fit_I = np.exp(intercept + r * early_data['day'])
plt.plot(early_data['day'], fit_I, color='red', label='Exponential Fit')
plt.xlabel('Day')
plt.ylabel('Active Infections')
plt.title(f'Early Outbreak Fit (R0 ≈ {R0:.2f})')
plt.legend()
plt.show()