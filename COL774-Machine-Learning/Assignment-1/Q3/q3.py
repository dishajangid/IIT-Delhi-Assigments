import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
def load_data():
    # Load the input features and target labels
    X = pd.read_csv('logisticX.csv', header=None).values
    y = pd.read_csv('logisticY.csv', header=None).values.flatten()
    return X, y

# Normalize features
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute the gradient of the log-likelihood
def compute_gradient(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    gradient = (1/m) * X.T.dot(predictions - y)
    return gradient

# Compute the Hessian matrix of the log-likelihood
def compute_hessian(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    S = np.diag(predictions * (1 - predictions))  # Diagonal matrix of sigmoid derivatives
    hessian = (1/m) * X.T.dot(S).dot(X)
    return hessian

# Newton's method for logistic regression
def newton_method(X, y, num_iterations=10, tolerance=1e-5):
    m, n = X.shape
    # Initialize theta (including the intercept term)
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        # Compute the gradient and Hessian
        gradient = compute_gradient(X, y, theta)
        hessian = compute_hessian(X, y, theta)
        
        # Update theta using Newton's method
        theta_new = theta - np.linalg.inv(hessian).dot(gradient)
        
        # Check for convergence
        if np.linalg.norm(theta_new - theta) < tolerance:
            break
        
        theta = theta_new
    
    return theta

# Plot for the training data and decision boundary

def plot_data_and_boundary(X, y, theta):
    plt.figure(figsize=(8, 6))

    # Plot class 0 with circles, class 1 with crosses
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', color='blue', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', color='red', label='Class 1')

    # Decision boundary: find line where sigmoid(theta^T x) = 0.5 => theta^T x = 0
    # For 2D, line equation: theta0 + theta1*x1 + theta2*x2 = 0 -> x2 = -(theta0 + theta1*x1)/theta2
    x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
    y_vals = -(theta[0] + theta[1] * x_vals) / theta[2]

    # Plot decision boundary line
    plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')

    plt.xlabel('Feature 1 (x1)')
    plt.ylabel('Feature 2 (x2)')
    plt.legend()
    plt.title('Training Data and Logistic Regression Decision Boundary')
    plt.show()

def main():
    # Load data
    X, y = load_data()

    # Normalize features
    X_normalized = normalize_features(X)

    # Add intercept term (bias) to X
    X_normalized = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

    # Apply Newton's method to fit the logistic regression model
    theta = newton_method(X_normalized, y)

    # Print the coefficients (theta)
    print(f"Fitted coefficients: {theta}")

    # Plot the data and decision boundary
    plot_data_and_boundary(X_normalized[:, 1:], y, theta)

if __name__ == "__main__":
    main()
