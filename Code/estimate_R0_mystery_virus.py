#estimate_R0_mystery_virus



#estimate_R0_mystery_virus.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# - Settings  -
<<<<<<< HEAD
csv_path = Path("Data") / "mystery_virus_daily_active_counts_RELEASE.csv"
infectious_period_days = 5.0  
min_window = 7                # minimum length (days) of the exponential window
max_window = 14               # maximum length (days) of the exponential window

#- File-relative path (robust) -
HERE = Path(__file__).resolve().parent
csv_path = HERE / "Data" / "mystery_virus_daily_active_counts_RELEASE.csv"
print("Using CSV at:", csv_path)

if not csv_path.exists():    
    raise FileNotFoundError(f"Could not find CSV at: {csv_path}\n"                            
                            "Check the filename and folder exactly.")
=======
infectious_period_days = 5.0
min_window = 7
max_window = 14
>>>>>>> f0a4222042e96cec350d53b79d14bb591b1e922f


csv_path = Path("mystery_virus_daily_active_counts_RELEASE#1.csv")
print("Using CSV at:", csv_path.resolve())

if not csv_path.exists():
    raise FileNotFoundError(
        f"Could not find CSV at: {csv_path.resolve()}\n"
        "Make sure the CSV is in the same folder as this script/notebook."
    )

# ---- Load data ----
df = pd.read_csv(csv_path)


# Standardize column names (lowercase, replace spaces with underscores)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# We expect: 'day' and 'active_reported_daily_cases'
expected = {"day", "active_reported_daily_cases"}
if not expected.issubset(df.columns):    
    raise KeyError(        
        f"Expected columns {sorted(expected)}, but found {list(df.columns)}.\n"        
        "Update the column mapping below if names differ."    
    )
# Extract day and I(t)
t = df["day"].astype(float).to_numpy()
I = df["active_reported_daily_cases"].astype(float).to_numpy()

# Guard against nonpositive I values (log undefined)
if np.any(I <= 0):    
      
    I = np.where(I <= 0, 1e-6, I)

# - Helper to fit a window and return r, I0, r2
def fit_exp_window(tw, Iw):    
    """    
    Fit log(I) = r * t + log(I0).    
    Returns r, I0, r2 (R-squared on log-scale).    
    """    
    logI = np.log(Iw)    
    # Linear fit: slope r, intercept logI0    
    r, logI0 = np.polyfit(tw, logI, 1)    
    # Compute R^2 on log-scale    
    logI_pred = r * tw + logI0    
    ss_res = np.sum((logI - logI_pred) ** 2)    
    ss_tot = np.sum((logI - np.mean(logI)) ** 2)    
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan    
    I0 = np.exp(logI0)    
    return r, I0, r2

# - Search for the most exponential-looking window
best = None  # (r2, r, I0, t_start, t_end)
n = len(t)

for w in range(min_window, max_window + 1):    
    for i in range(0, n - w + 1):        
        tw = t[i:i+w]        
        Iw = I[i:i+w]        
        # Avoid all-constant segments in log-space        
        if np.allclose(Iw, Iw[0]):            
            continue        
        r, I0, r2 = fit_exp_window(tw, Iw)        
        if np.isnan(r2):            
            continue        
        if (best is None) or (r2 > best[0]):            
            best = (r2, r, I0, int(tw[0]), int(tw[-1]))
        
if best is None:    
    raise RuntimeError("Could not find a suitable exponential window. Check your data.")

best_r2, r, I0, t_start, t_end = best

# - Compute R0 and doubling time
doubling_time = np.log(2) / r if r > 0 else np.inf
R0 = 1.0 + r * infectious_period_days

print(f"Best exponential window: day {t_start} to {t_end} (length {t_end - t_start + 1})")
print(f"Fit on log(I): R^2 = {best_r2:.4f}")
print(f"Growth rate r: {r:.4f} per day")
print(f"Doubling time: {doubling_time:.2f} days")
print(f"Assumed infectious period D: {infectious_period_days} days")
print(f"Estimated R0 ≈ 1 + r*D = {R0:.3f}")

# - Plot: scatter all data + fitted exponential curve
t_fit = np.linspace(t.min(), t.max(), 300)
I_fit = I0 * np.exp(r * t_fit)

plt.figure(figsize=(9, 5.5))
plt.scatter(t, I, s=28, alpha=0.8, label="Observed I(t)")
plt.plot(t_fit, I_fit, 'r', lw=2.0, label=f"Fitted exponential: I(t)=I0·e^(r t)\n(r={r:.3f}/day, R²={best_r2:.3f})")


# Shade the selected window
plt.axvspan(t_start, t_end, color="orange", alpha=0.15, label="Fit window")
plt.xlabel("Day")
plt.ylabel("Active infections")
plt.title("Exponential Fit to Mystery Virus Growth (for R0 estimate)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
