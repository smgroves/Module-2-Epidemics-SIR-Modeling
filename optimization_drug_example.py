# drug efficacy optimization example for BME 2315
# made by Lavie, fall 2025

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt


#%% define drug models

# define toxicity levels for each drug (lambda)
metformin_lambda = 0.5

lisinopril_lambda = 0.8

escitalopram_lambda = 0.3

def metformin(x):   # mild toxicity, moderate efficacy
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - metformin_lambda * toxicity
def lisinopril(x):  # strong efficacy, higher toxicity
    efficacy = np.exp(-0.1*(x-7)**2)
    toxicity = 0.3 * x**2 / 80
    return efficacy - lisinopril_lambda * toxicity
def escitalopram(x):  # weaker efficacy, low toxicity
    efficacy = 0.6 * np.exp(-0.1*(x-4)**2)
    toxicity = 0.1 * x**2 / 120
    return efficacy - escitalopram_lambda * toxicity
def ideal_dose(x): # ideal dose curve is the sum of the three drug efficacies, which we want to maximize
    return (metformin(x) + lisinopril(x) + escitalopram(x))

#%% plot drug efficacies
x = np.linspace(0, 15, 100)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, metformin(x), label='Metformin', color='blue')
plt.plot(x, lisinopril(x), label='Lisinopril', color='orange')
plt.plot(x, escitalopram(x), label='Escitalopram', color='green')
plt.plot(x, ideal_dose(x), label='Ideal Dose Curve', color='red', linestyle='--')
plt.title('Drug Efficacy vs Dosage')
plt.xlabel('Dosage (mg)')
plt.ylabel('Net Effect')
plt.legend()
plt.show()

# %% Find optimal dosages for each drug

# First method: Steepest Ascent using the update rule

# first, need the first derivative (gradient)
def gradient(f, x, h=1e-4):
    """Central difference approximation for f'(x)."""
    return (f(x + h) - f(x - h)) / (2*h)

def steepest_ascent(f, x0, h_step=0.1, tol=1e-6, max_iter=1000):
    x = x0 # update initial guess
    for i in range(max_iter):
        grad = gradient(f, x)
        x_new = x + h_step * grad     
        
        if abs(x_new - x) < tol:      # convergence condition, when solution is 0
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin, opt_effect_metformin = steepest_ascent(metformin, x0=1.0)
print(f"Steepest Ascent Method - Optimal Metformin Dose: {opt_dose_metformin:.2f} mg")
print(f"Steepest Ascent Method - Optimal Metformin Effect: {opt_effect_metformin*100:.2f}%")

# lisinopril
opt_dose_lisinopril, opt_effect_lisinopril = steepest_ascent(lisinopril, x0=1.0)
print(f"Steepest Ascent Method - Optimal Lisinopril Dose: {opt_dose_lisinopril:.2f} mg")
print(f"Steepest Ascent Method - Optimal Lisinopril Effect: {opt_effect_lisinopril*100:.2f}%")

# escitalopram
opt_dose_escitalopram, opt_effect_escitalopram = steepest_ascent(escitalopram, x0=1.0)
print(f"Steepest Ascent Method - Optimal Escitalopram Dose: {opt_dose_escitalopram:.2f} mg")
print(f"Steepest Ascent Method - Optimal Escitalopram Effect: {opt_effect_escitalopram*100:.2f}%")

opt_ideal_dose, opt_effect_ideal_dose = steepest_ascent(ideal_dose, x0=1.0)
print(f"Steepest Ascent Method- Optimal Ideal Dose (combined effect of all three drugs): {opt_ideal_dose:.2f} mg")
print(f"Steepest Ascent Method - Optimal Ideal Dose Effect (combined effect of all three drugs): {opt_effect_ideal_dose*100:.2f}%")

# %% Newton's method

# requires second derivative
def second_derivative(f, x, h=1e-4):
    """Central difference approximation for f''(x)."""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

def newtons_method(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = gradient(f, x)
        hess = second_derivative(f, x)
        
        if hess == 0:  # avoid division by zero
            print("Zero second derivative. No solution found.")
            return x, f(x)
        
        x_new = x - grad / hess
        
        if abs(x_new - x) < tol:
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin_nm, opt_effect_metformin_nm = newtons_method(metformin, x0=1.0)
print(f"Newton's Method - Optimal Metformin Dose: {opt_dose_metformin_nm:.2f} mg")
print(f"Newton's Method - Optimal Metformin Effect: {opt_effect_metformin_nm*100:.2f}%")                

# lisinopril
opt_dose_lisinopril_nm, opt_effect_lisinopril_nm = newtons_method(lisinopril, x0=1.0)
print(f"Newton's Method - Optimal Lisinopril Dose: {opt_dose_lisinopril_nm:.2f} mg")
print(f"Newton's Method - Optimal Lisinopril Effect: {opt_effect_lisinopril_nm*100:.2f}%")

# escitalopram
opt_dose_escitalopram_nm, opt_effect_escitalopram_nm = newtons_method(escitalopram, x0=1.0)
print(f"Newton's Method - Optimal Escitalopram Dose: {opt_dose_escitalopram_nm:.2f} mg")
print(f"Newton's Method - Optimal Escitalopram Effect: {opt_effect_escitalopram_nm*100:.2f}%")

#ideal dose
opt_ideal_dose_nm, opt_effect_ideal_dose_nm = newtons_method(ideal_dose, x0 = 1.0)
print(f"Newton's Method - Optimal Ideal Dose: {opt_effect_ideal_dose_nm: .2f} mg")
print(f"Newton's Method - Optimal Ideal Dose Effect: {opt_effect_ideal_dose_nm*100: .2f}%")

#testing multiple lambda values to determine optimum values

lambda_values = np.linspace(0.1, 1.0, 10)  # try 10 λ values from 0.1 to 1.0, (ChatGPT, 2026)
best_lambda = None # variable to store the best lambda value that gives the optimal dose closest to the original optimal ideal dose
closest_diff = float('inf') # variable to store the closest difference between the optimal dose from steepest ascent and the original optimal ideal dose, initialized to infinity
best_opt_dose = None # variable to store the optimal dose corresponding to the best lambda value

for lam in lambda_values: # loop through each lambda value and update the global variable for metformin_lambda, then optimize the ideal dose curve using steepest ascent and compare the resulting optimal dose to the original optimal ideal dose
    metformin_lambda = lam # update global variable for metformin_lambda to the current lambda value in the loop
    opt_dose, opt_effect = steepest_ascent(ideal_dose, x0=1.0)  # optimize combined effect
    
    diff = abs(opt_dose - opt_ideal_dose)  # compare to original optimal ideal dose
    
    if diff < closest_diff: # if the difference is smaller than the closest difference found so far, update the closest difference and store the current lambda value and optimal dose
        closest_diff = diff
        best_lambda = lam
        best_opt_dose = opt_dose

print(f"Best Metformin λ: {best_lambda:.2f}")
print(f"Resulting Optimal Dose (combined effect): {best_opt_dose:.2f} mg")
