import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_normalize_data():
    # Load the feature data (2D array)
    X = np.loadtxt('../Q4/q4x.dat')
    # Load the target labels as strings ('Alaska', 'Canada')
    y_str = np.loadtxt('../Q4/q4y.dat', dtype=str)
    # Convert labels from strings ('Alaska', 'Canada') to integers (0, 1)
    y = np.where(y_str == 'Alaska', 0, 1)
    # Normalize data (zero mean, unit variance)
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    return X_normalized, y

# Learn GDA Parameters for Shared Covariance Matrix
def learn_gda_shared_covariance(X, y):
    # Separate data by class
    X0 = X[y == 0]  # Class 0 (Alaska)
    X1 = X[y == 1]  # Class 1 (Canada)
    
    # Compute means for each class
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    
    # Compute shared covariance matrix Σ
    m0 = X0.shape[0]
    m1 = X1.shape[0]
    cov0 = np.cov(X0, rowvar=False)
    cov1 = np.cov(X1, rowvar=False)
    
    # Since the covariance matrices are assumed equal
    Sigma = (cov0 * m0 + cov1 * m1) / (m0 + m1)
    
    return mu0, mu1, Sigma

# Plot Training Data Only
def plot_data_only(X, y):
    """Plot only the data points without any decision boundary"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', label='Alaska', color='blue')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', label='Canada', color='red')
    
    plt.xlabel('Fresh Water Diameter (Normalized)')
    plt.ylabel('Marine Water Diameter (Normalized)')
    plt.legend()
    plt.title('GDA: Data Points (Alaska vs. Canada)')
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot Training Data and Linear Decision Boundary
def plot_data_and_boundary(X, y, mu0, mu1, Sigma):
    """Plot the data and linear decision boundary"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', label='Alaska', color='blue')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', label='Canada', color='red')
    
    # Compute the linear decision boundary
    w = np.linalg.inv(Sigma).dot(mu1 - mu0)
    b = -0.5 * (mu1 + mu0).dot(np.linalg.inv(Sigma)).dot(mu1 - mu0)
    
    # Define the boundary as w[0] * x1 + w[1] * x2 + b = 0
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = w[0] * xx + w[1] * yy + b
    
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, linestyles='--', label='Linear Boundary')
    plt.xlabel('Fresh Water Diameter (Normalized)')
    plt.ylabel('Marine Water Diameter (Normalized)')
    plt.legend()
    plt.title('GDA: Linear Decision Boundary')
    plt.grid(True, alpha=0.3)
    plt.show()

# Learn GDA Parameters for Separate Covariance Matrices
def learn_gda_separate_covariance(X, y):
    # Separate data by class
    X0 = X[y == 0]  # Class 0 (Alaska)
    X1 = X[y == 1]  # Class 1 (Canada)
    
    # Compute means for each class
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    
    # Compute covariance matrices for each class
    Sigma0 = np.cov(X0, rowvar=False)
    Sigma1 = np.cov(X1, rowvar=False)
    
    return mu0, mu1, Sigma0, Sigma1

# Plot the Quadratic Decision Boundary
def plot_quadratic_boundary(X, y, mu0, mu1, Sigma0, Sigma1):
    """Plot the data and quadratic decision boundary"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', label='Alaska', color='blue')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', label='Canada', color='red')

    # Define a mesh grid to plot the boundary
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Compute decision boundary (quadratic form)
    Z = np.array([np.dot(np.dot([x, y] - mu0, np.linalg.inv(Sigma0)), [x, y] - mu0) - 
                  np.dot(np.dot([x, y] - mu1, np.linalg.inv(Sigma1)), [x, y] - mu1) 
                  for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, linestyles='-', label='Quadratic Boundary')
    plt.xlabel('Fresh Water Diameter (Normalized)')
    plt.ylabel('Marine Water Diameter (Normalized)')
    plt.legend()
    plt.title('GDA: Quadratic Decision Boundary')
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot Both Linear and Quadratic Decision Boundaries
def plot_both_boundaries(X, y, mu0, mu1, Sigma, Sigma0, Sigma1):
    """Plot data with both linear and quadratic decision boundaries"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', label='Alaska', color='blue')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', label='Canada', color='red')
    
    # Linear decision boundary
    w = np.linalg.inv(Sigma).dot(mu1 - mu0)
    b = -0.5 * (mu1 + mu0).dot(np.linalg.inv(Sigma)).dot(mu1 - mu0)
    
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z_linear = w[0] * xx + w[1] * yy + b
    
    # Quadratic decision boundary
    Z_quadratic = np.array([np.dot(np.dot([x, y] - mu0, np.linalg.inv(Sigma0)), [x, y] - mu0) - 
                            np.dot(np.dot([x, y] - mu1, np.linalg.inv(Sigma1)), [x, y] - mu1) 
                            for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z_quadratic = Z_quadratic.reshape(xx.shape)
    
    # Plot both boundaries
    plt.contour(xx, yy, Z_linear, levels=[0], colors='black', linewidths=2, linestyles='--', label='Linear Boundary')
    plt.contour(xx, yy, Z_quadratic, levels=[0], colors='green', linewidths=2, linestyles='-', label='Quadratic Boundary')
    
    plt.xlabel('Fresh Water Diameter (Normalized)')
    plt.ylabel('Marine Water Diameter (Normalized)')
    plt.legend(['Alaska', 'Canada', 'Linear Boundary', 'Quadratic Boundary'])
    plt.title('GDA: Linear and Quadratic Decision Boundaries')
    plt.grid(True, alpha=0.3)
    plt.show()

# Main function to execute the tasks
def main():
    # Step 1: Load and normalize data
    X, y = load_and_normalize_data()
    print("Data loaded and normalized successfully!")
    
    # Step 2: Plot data points only
    print("\nPlotting data points only...")
    plot_data_only(X, y)
    
    # Step 3: Learn parameters for shared covariance matrix
    mu0, mu1, Sigma = learn_gda_shared_covariance(X, y)
    print(f"\nShared Covariance Parameters:")
    print(f"mu0: {mu0}\nmu1: {mu1}\nSigma: {Sigma}")
    
    # Step 4: Plot data and linear decision boundary
    print("\nPlotting data with linear decision boundary...")
    plot_data_and_boundary(X, y, mu0, mu1, Sigma)
    
    mu0, mu1, Sigma0, Sigma1 = learn_gda_separate_covariance(X, y)
    print(f"\nSeparate Covariance Parameters:")
    print(f"mu0: {mu0}\nmu1: {mu1}\nSigma0: {Sigma0}\nSigma1: {Sigma1}")
    
    print("\nPlotting data with quadratic decision boundary...")
    plot_quadratic_boundary(X, y, mu0, mu1, Sigma0, Sigma1)
    
    print("\nPlotting data with both linear and quadratic decision boundaries...")
    plot_both_boundaries(X, y, mu0, mu1, Sigma, Sigma0, Sigma1)


if __name__ == "__main__":
    main()
