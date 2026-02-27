import pandas as pd
import matplotlib.pyplot as plt

# Load in-class SIR data
data = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#1.csv")

# Convert columns to numeric just in case
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Extract compartments
days = data["day"]
S = data["S"]
I = data["I"]
R = data["R"]

# Create single plot
plt.figure()
plt.plot(days, S)
plt.plot(days, I)
plt.plot(days, R)

plt.xlabel("Day")
plt.ylabel("Population Size")
plt.title("SIR Compartments Over Time")
plt.legend(["Susceptible", "Infected", "Recovered"])

plt.show()
