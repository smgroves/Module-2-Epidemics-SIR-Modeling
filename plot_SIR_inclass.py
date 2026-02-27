#plot_SIR_inclass.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Your files are now all in ONE folder ---
HERE = Path(__file__).resolve().parent

# IMPORTANT: Use the EXACT filename of your SIR CSV here:
sir_csv = HERE / "in_class_SIR_data.csv"   

# If you don't have this file, you must create it.
if not sir_csv.exists():    
    raise FileNotFoundError(        
        f"SIR CSV not found: {sir_csv}\n"        
        "Create in_class_SIR_data.csv in the SAME folder as this .py file.\n"        
        "It must contain columns: day,S,I,R"    
    )

df = pd.read_csv(sir_csv)

# Check columns
expected = {"day", "S", "I", "R"}
if not expected.issubset(df.columns):    
    raise KeyError(        
        f"Expected columns {sorted(expected)}, but found: {list(df.columns)}"    
    )

days = df["day"]
S = df["S"]
I = df["I"]
R = df["R"]

plt.figure(figsize=(8,5))
plt.plot(days, S, label="Susceptible (S)")
plt.plot(days, I, label="Infected (I)")
plt.plot(days, R, label="Recovered (R)")
plt.xlabel("Day")
plt.ylabel("Population Size")
plt.title("SIR Compartments Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()