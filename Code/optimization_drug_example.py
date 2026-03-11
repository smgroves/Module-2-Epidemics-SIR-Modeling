# drug efficacy optimization example for BME 2315
# made by Lavie, fall 2025

# %% import libraries
import numpy as np
import matplotlib.pyplot as plt

# %% define drug models

# define toxicity levels for each drug (lambda)
metformin_lambda = 0.5
lisinopril_lambda = 0.8
escitalopram_lambda = 0.3

def metformin(x):   # mild toxicity, moderate efficacy
    efficacy = 0.8 * np.exp(-0.1 * (x - 5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - metformin_lambda * toxicity

def lisinopril(x):  # strong efficacy, higher toxicity
    efficacy = np.exp(-0.1 * (x - 7)**2)
    toxicity = 0.3 * x**2 / 80
    return efficacy - lisinopril_lambda * toxicity

def escitalopram(x):  # weaker efficacy, low toxicity
    efficacy = 0.6 * np.exp(-0.1 * (x - 4)**2)
    toxicity = 0.1 * x**2 / 120
    return efficacy - escitalopram_lambda * toxicity

def total_effect(doses):
    metformin_dose, lisinopril_dose, escitalopram_dose = doses
    return (
        metformin(metformin_dose)
        + lisinopril(lisinopril_dose)
        + escitalopram(escitalopram_dose)
    )

# %% plot drug efficacies
x = np.linspace(0, 15, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, metformin(x), label='Metformin', color='blue')
plt.plot(x, lisinopril(x), label='Lisinopril', color='orange')
plt.plot(x, escitalopram(x), label='Escitalopram', color='green')
plt.plot(x, total_effect([x, x, x]), label='Total Effect', color='red', linestyle='--')

plt.title('Drug Efficacy vs Dosage')
plt.xlabel('Dosage (mg)')
plt.ylabel('Net Effect')
plt.legend()
plt.show()

# %% Optimization methods

def gradient(f, x, h=1e-4):
    """Central difference approximation for f'(x)."""
    return (f(x + h) - f(x - h)) / (2 * h)

def steepest_ascent(f, x0, h_step=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = gradient(f, x)
        x_new = x + h_step * grad

        if abs(x_new - x) < tol:
            break

        x = x_new

    return x, f(x)

# %% Steepest ascent optimization

opt_dose_metformin, opt_effect_metformin = steepest_ascent(metformin, x0=1.0)
opt_dose_lisinopril, opt_effect_lisinopril = steepest_ascent(lisinopril, x0=1.0)
opt_dose_escitalopram, opt_effect_escitalopram = steepest_ascent(escitalopram, x0=1.0)

opt_doses_total = np.array([
    opt_dose_metformin,
    opt_dose_lisinopril,
    opt_dose_escitalopram
])

opt_effect_total = total_effect(opt_doses_total)

print("Steepest Ascent Results")
print(f"Metformin: {opt_dose_metformin:.2f} mg")
print(f"Lisinopril: {opt_dose_lisinopril:.2f} mg")
print(f"Escitalopram: {opt_dose_escitalopram:.2f} mg")
print(f"Total Effect: {opt_effect_total*100:.2f}%")

# %% Newton's method

def second_derivative(f, x, h=1e-4):
    """Central difference approximation for f''(x)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

def newtons_method(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = gradient(f, x)
        hess = second_derivative(f, x)

        if hess == 0:
            break

        x_new = x - grad / hess

        if abs(x_new - x) < tol:
            break

        x = x_new

    return x, f(x)

# %% Newton's method optimization

opt_dose_metformin_nm, _ = newtons_method(metformin, x0=1.0)
opt_dose_lisinopril_nm, _ = newtons_method(lisinopril, x0=1.0)
opt_dose_escitalopram_nm, _ = newtons_method(escitalopram, x0=1.0)

opt_doses_total_nm = np.array([
    opt_dose_metformin_nm,
    opt_dose_lisinopril_nm,
    opt_dose_escitalopram_nm
])

opt_effect_total_nm = total_effect(opt_doses_total_nm)

print("\nNewton's Method Results")
print(f"Metformin: {opt_dose_metformin_nm:.2f} mg")
print(f"Lisinopril: {opt_dose_lisinopril_nm:.2f} mg")
print(f"Escitalopram: {opt_dose_escitalopram_nm:.2f} mg")
print(f"Total Effect: {opt_effect_total_nm*100:.2f}%")

# %% Lambda optimization for Metformin

target_dose = opt_dose_metformin

lambda_values = np.linspace(0.1, 1.5, 50)
dose_differences = []

best_lambda = None
best_dose = None
min_difference = np.inf

for lam in lambda_values:
    metformin_lambda = lam  # update lambda
    
    dose, _ = steepest_ascent(metformin, x0=1.0)
    diff = abs(dose - target_dose)
    dose_differences.append(diff)
    
    if diff < min_difference:
        min_difference = diff
        best_lambda = lam
        best_dose = dose

print("\nLambda Optimization for Metformin")
print(f"Target Optimal Dose: {target_dose:.2f} mg")
print(f"Best Lambda Value: {best_lambda:.2f}")
print(f"Optimal Dose at Best Lambda: {best_dose:.2f} mg")
print(f"Difference from Target: {min_difference:.4f} mg")

# %% Plot lambda vs dose difference

plt.figure(figsize=(8, 5))
plt.plot(lambda_values, dose_differences)
plt.xlabel("Lambda Value")
plt.ylabel("Absolute Difference from Target Dose (mg)")
plt.title("Effect of Lambda on Optimal Metformin Dose")
plt.grid(True)
plt.show()
