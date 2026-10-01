import random
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# --------------------------------------------------
# 1. SIMULATED EXPERIMENT
# --------------------------------------------------

def simulate_spectrum(red, green, blue):
    """
    Pretend that these RGB values are sent to a physical LED
    and measured by a sensor.
    """

    return [red, green, blue]


# --------------------------------------------------
# 2. CALCULATE HOW FAR WE ARE FROM THE TARGET
# --------------------------------------------------

def calculate_error(measured, target):
    total_error = 0

    for i in range(len(target)):
        difference = abs(measured[i] - target[i])
        total_error += difference

    return total_error / len(target)


# This is the spectrum we want to achieve.
target = [0.7, 0.5, 0.3]


# --------------------------------------------------
# 3. OBJECTIVE FUNCTION
# --------------------------------------------------

def objective(red, green, blue):
    """
    The optimizer only needs to call this function.

    Input:
        RGB values

    Output:
        How bad/good those values were
    """

    measured = simulate_spectrum(red, green, blue)

    error = calculate_error(measured, target)

    return error


# --------------------------------------------------
# 4. INITIAL EXPERIMENTS
# --------------------------------------------------

X = []
y = []

# Bayesian optimization needs some observations
# before it can start learning.
for experiment in range(5):

    red = random.random()
    green = random.random()
    blue = random.random()

    error = objective(red, green, blue)

    X.append([red, green, blue])
    y.append(error)


# Convert Python lists into NumPy arrays
X = np.array(X)
y = np.array(y)


print("Initial Inputs:")
print(X)

print("\nInitial Errors:")
print(y)


# --------------------------------------------------
# 5. CREATE THE GAUSSIAN PROCESS MODEL
# --------------------------------------------------

kernel = RBF(length_scale=1.0)

model = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True
)


# --------------------------------------------------
# 6. BAYESIAN OPTIMIZATION LOOP
# --------------------------------------------------

kappa = 2.0

for iteration in range(10):

    # ----------------------------------------------
    # A. Learn from every experiment performed so far
    # ----------------------------------------------

    model.fit(X, y)


    # ----------------------------------------------
    # B. Generate possible experiments
    # ----------------------------------------------

    candidates = np.random.random((1000, 3))


    # ----------------------------------------------
    # C. Ask the Gaussian Process to predict them
    # ----------------------------------------------

    predicted_errors, uncertainties = model.predict(
        candidates,
        return_std=True
    )


    # ----------------------------------------------
    # D. Acquisition function
    #
    # Low predicted error = attractive
    # High uncertainty = also attractive
    # ----------------------------------------------

    acquisition_scores = (
        predicted_errors
        - kappa * uncertainties
    )


    # ----------------------------------------------
    # E. Choose the best candidate
    # ----------------------------------------------

    best_index = np.argmin(acquisition_scores)

    next_rgb = candidates[best_index]


    # ----------------------------------------------
    # F. Actually perform the experiment
    # ----------------------------------------------

    actual_error = objective(*next_rgb)


    # ----------------------------------------------
    # G. Save what we learned
    # ----------------------------------------------

    X = np.vstack([X, next_rgb])

    y = np.append(y, actual_error)


    print(f"\nIteration {iteration + 1}")

    print("RGB:", next_rgb)

    print(
        "Predicted error:",
        predicted_errors[best_index]
    )

    print(
        "Uncertainty:",
        uncertainties[best_index]
    )

    print(
        "Actual error:",
        actual_error
    )


# --------------------------------------------------
# 7. FIND THE BEST EXPERIMENT WE EVER PERFORMED
# --------------------------------------------------

best_index = np.argmin(y)

best_rgb = X[best_index]

best_error = y[best_index]


print("\n---------------------------")
print("FINAL RESULT")
print("---------------------------")

print("Target:", target)
print("Best RGB found:", best_rgb)
print("Best error:", best_error)