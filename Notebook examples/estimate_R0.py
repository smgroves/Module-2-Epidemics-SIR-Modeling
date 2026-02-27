import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(
    'Data/mystery_virus_daily_active_counts_RELEASE#1.csv',
    parse_dates=['date']
)

# Convert active cases to numeric
active_col = data.columns[2]
data[active_col] = pd.to_numeric(data[active_col], errors="coerce")

# Extract early exponential phase
early_data = data.iloc[0:6]   # first 6 days (adjust if needed)

days = early_data["day"].values
I = early_data[active_col].values

# Remove zeros (cannot log 0)
mask = I > 0
days = days[mask]
I = I[mask]

# Log transform
log_I = np.log(I)

# Fit line
slope, intercept = np.polyfit(days, log_I, 1)
r = slope

print("Estimated growth rate r =", r)

# Define recovery rate gamma
gamma = 1/5   # CHANGE if infectious period given

R0 = 1 + r/gamma
print("Estimated R0 =", R0)

# Plot fit
plt.figure()
plt.plot(days, log_I)
plt.plot(days, slope*days + intercept)
plt.xlabel("Day")
plt.ylabel("ln(I)")
plt.title("Exponential Growth Fit")
plt.show()