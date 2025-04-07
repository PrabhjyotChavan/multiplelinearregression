import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Perform multiple linear regression using gradient descent.")
parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for gradient descent.")
parser.add_argument("--iterations", type=int, required=True, help="Number of iterations for gradient descent.")
args = parser.parse_args()

# Load dataset
df = pd.read_csv("/Users/rohitchavan/Documents/UND COURSES/527 Predictive modeling/Project 1/myDataMLR(Sheet1).csv")

# Extract features and target variable
X = df[['x1', 'x2']].values
y = df['y'].values.reshape(-1, 1)

# Store original mean and standard deviation of y for denormalization
y_mean = np.mean(y)
y_std = np.std(y)

# Normalize features and target variable
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

# Add intercept term (β0)
m, n = X.shape
X = np.c_[np.ones((m, 1)), X]

# Initialize parameters
learning_rate = args.learning_rate
iterations = args.iterations
beta = np.zeros((n + 1, 1))

# Gradient Descent
mse_log = []
for i in range(iterations):
    predictions = X.dot(beta)
    error = predictions - y
    mse = np.mean(error ** 2)
    mse_log.append(mse)

    # Compute gradients and update beta
    gradient = (2 / m) * X.T.dot(error)
    beta -= learning_rate * gradient

# Denormalize predictions to original scale
predictions_original = predictions * y_std + y_mean
y_original = y * y_std + y_mean

# Convert intercept back to original scale
beta_0_original = y_mean - (beta[1] * X_mean[0] + beta[2] * X_mean[1])
beta_original = [beta_0_original] + beta[1:].flatten().tolist()

# Save MSE log file
np.savetxt("MLRTraining[{}][{}]MSE.txt".format(iterations, learning_rate), mse_log, delimiter=",")

# Save Model Parameters
r2 = 1 - (np.sum((predictions - y) ** 2) / np.sum((y - np.mean(y)) ** 2))
model_params = [learning_rate, iterations, mse_log[-1], beta_original, r2]
pd.DataFrame([model_params], columns=["Learning Rate", "Iterations", "Final MSE", "Slopes & Intercept", "R2"]).to_csv("MLRModelParameters.txt", index=False)

# Print Model Parameters
print("Final Parameters:")
print("Learning Rate:", learning_rate)
print("Iterations:", iterations)
print("Final MSE:", mse_log[-1])
print("Slopes & Intercept (Original Scale):", beta_original)
print("R2 Score:", r2)

# Plot Actual vs. Predicted Values (Denormalized)
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_original)), y_original, color='blue', label='Actual Values')
plt.scatter(range(len(predictions_original)), predictions_original, color='red', label='Predicted Values')
for i in range(len(y_original)):
    plt.plot([i, i], [y_original[i], predictions_original[i]], 'k-')
plt.legend()
plt.title("Distribution of Actual and Predicted Values")
plt.savefig("Actual_vs_Predicted.png")
plt.show()

# Plot MSE per iteration
plt.figure(figsize=(8, 6))
plt.plot(range(iterations), mse_log, color='red')
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE vs Iterations - Gradient Descent")
plt.savefig("MSE_vs_Iterations.png")
plt.show()

