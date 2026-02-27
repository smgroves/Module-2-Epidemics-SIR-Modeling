import os
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent

# ----------------------
# Path to your CSV (same folder as this script)
# ----------------------
csv_filename = "mystery_virus_daily_active_counts_RELEASE#1.csv"
csv_path = HERE / csv_filename

print("Using CSV at:", csv_path.resolve())

# Check if the file exists
if not csv_path.exists():
    raise FileNotFoundError(
        f"Could not find CSV at: {csv_path.resolve()}\n"
        "Make sure the CSV is in the same folder as this script."
    )

df = pd.read_csv(csv_path)



print("CWD:", os.getcwd())
print("\nHere are files/folders in the current directory:\n")
for p in Path(".").iterdir():    
    print("-", p)
print("\nCSV files I can see, recursively:\n")
for p in Path(".").rglob("*.csv"):    
    print("-", p)






import matplotlib.pyplot as plt

# -Load your S, I, R data -

# Replace with the correct filename from your repo

df = pd.read_csv(csv_path)
days = df["day"]
S = df["S"]
I = df["I"]
R = df["R"]

# - Plot -
plt.figure(figsize=(8,5))
plt.plot(days, S, label="Susceptible (S)")
plt.plot(days, I, label="Infected (I)")
plt.plot(days, R, label="Recovered (R)")

plt.xlabel("Day")
plt.ylabel("Population size")
plt.title("SIR Compartments Over Time")
plt.legend()
plt.grid(True)
plt.show()