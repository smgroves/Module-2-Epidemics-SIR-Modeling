#relative model error prediction

from pyexpat import model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path


# Load the data
# find the folder where the script is located
HERE = Path(__file__).parent

# build path to the csv
csv_path = HERE / "mystery_virus_daily_active_counts_RELEASE#3.csv"

data = pd.read_csv(csv_path, parse_dates=['date'])


#peaks from the dataset
cases = data['active reported daily cases'].values
days = data['day'].values

peak_index = np.argmax(cases)

actual_peak_value = cases[peak_index]
actual_peak_day = days[peak_index]

print("Actual peak infections:", actual_peak_value)
print("Actual peak day:", actual_peak_day)


#comparing model and dataset peaks
#True error & percent relative error
#Predicted peak infections: 3039.4968290416637
#Predicted peak day: 78

model_peak_value = 3039.4968290416637
model_peak_day = 78
true_error_peak_value = abs(model_peak_value - actual_peak_value)
relative_error_peak_value = true_error_peak_value / actual_peak_value * 100
true_error_peak_day = abs(model_peak_day - actual_peak_day)
relative_error_peak_day = true_error_peak_day / actual_peak_day * 100
print(f"True error in peak value: {true_error_peak_value}")
print(f"Relative error in peak value: {relative_error_peak_value:.2f}%")
print(f"True error in peak day: {true_error_peak_day}")
print(f"Relative error in peak day: {relative_error_peak_day:.2f}%")


