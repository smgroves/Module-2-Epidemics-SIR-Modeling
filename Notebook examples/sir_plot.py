import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(
    'Data/mystery_virus_daily_active_counts_RELEASE#1.csv',
    parse_dates=['date'],
    header=0
)

# Convert numeric columns
for col in data.columns:
    if col != 'date':
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Extract compartments
days = data["day"]
S = data["susceptible"]
I = data["active_cases"]
R = data["recovered"]

# Plot all three on same graph
plt.figure()
plt.plot(days, S)
plt.plot(days, I)
plt.plot(days, R)

plt.xlabel("Day")
plt.ylabel("Population Size")
plt.title("SIR Compartments Over Time")
plt.legend(["Susceptible", "Infected", "Recovered"])

plt.show()