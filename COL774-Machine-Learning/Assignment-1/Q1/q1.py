import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class LinearRegression:
    def __init__(self):
        self.theta = None
        self.cost_history = []
        self.theta_history = []
    
    def load_data(self):
        """Load the dataset"""
        X = pd.read_csv('linearX.csv', header=None).values
        y = pd.read_csv('linearY.csv', header=None).values
        X = np.c_[np.ones(X.shape[0]), X]
        return X, y
    
    def compute_cost(self, X, y, theta):
        """Compute the cost function J(theta)"""
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost
    
    def batch_gradient_descent(self, X, y, learning_rate=0.01, max_iterations=10000, epsilon=1e-8):
        """Implement batch gradient descent for linear regression"""
        m = len(y)
        self.theta = np.zeros((X.shape[1], 1))
        self.cost_history = []
        self.theta_history = []
        
        prev_cost = float('inf')
        
        for iteration in range(max_iterations):
            predictions = X.dot(self.theta)
            error = predictions - y
            cost = self.compute_cost(X, y, self.theta)
            self.cost_history.append(cost)
            self.theta_history.append(self.theta.flatten().copy())
            gradient = (1/m) * X.T.dot(error)
            self.theta = self.theta - learning_rate * gradient
            if abs(prev_cost - cost) < epsilon:
                print(f"Converged at iteration {iteration+1} with cost {cost:.6f}")
                break
            prev_cost = cost
        
        return self.theta, self.cost_history, self.theta_history
    
    def plot_data_and_hypothesis(self, X, y):
        """Plot the data points and fitted line"""
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 1], y, color='blue', alpha=0.6, label='Data points')
        x_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        y_pred = self.theta[0] + self.theta[1] * x_range
        plt.plot(x_range, y_pred, color='red', linewidth=2, label='Fitted Line')
        plt.xlabel('Acidity (x)')
        plt.ylabel('Density (y)')
        plt.title('Linear Regression: Data and Fitted Line')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_3d_cost_surface(self, X, y, animated=False):
        """Plot 3D cost surface with theta₁ starting at 0 up to 60"""
        theta_history = np.array(self.theta_history)
        
        # Set range for θ₀ (intercept) - adjust as needed
        theta0_min = min(theta_history[:, 0].min(), 0) - 10
        theta0_max = max(theta_history[:, 0].max(), 0) + 10
        
        # Set range for θ₁ specifically from 0 to 60
        theta1_min = 0
        theta1_max = 60
        
        # Generate grid with sufficient resolution
        theta0_range = np.linspace(theta0_min, theta0_max, 200)
        theta1_range = np.linspace(theta1_min, theta1_max, 200)
        Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
        
        # Compute cost for each grid point
        Z = np.zeros_like(Theta0)
        for i in range(Theta0.shape[0]):
            for j in range(Theta0.shape[1]):
                theta_temp = np.array([[Theta0[i, j]], [Theta1[i, j]]])
                Z[i, j] = self.compute_cost(X, y, theta_temp)
        
        # Plot surface
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Theta0, Theta1, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
        
        # Plot gradient descent path
        if len(theta_history) > 0:
            costs = [self.compute_cost(X, y, t.reshape(-1,1)) for t in theta_history]
            costs = np.array(costs)
            ax.plot(theta_history[:, 0], theta_history[:, 1], costs, 'r.-', linewidth=3, markersize=6,
                    markeredgecolor='red', label='Gradient Descent Path')
            ax.scatter([theta_history[0, 0]], [theta_history[0, 1]], [costs[0]], color='green', s=100, label='Start', edgecolors='white', linewidth=2)
            ax.scatter([theta_history[-1, 0]], [theta_history[-1, 1]], [costs[-1]], color='red', s=100, label='End', edgecolors='white', linewidth=2)
        
        ax.set_xlabel('θ₀ (Intercept)', fontsize=12, labelpad=10)
        ax.set_ylabel('θ₁ (Slope)', fontsize=12, labelpad=10)
        ax.set_zlabel('Cost Function J(θ)', fontsize=12, labelpad=10)
        ax.set_title(f'3D Cost Surface with Gradient Descent Path - Learning Rate: {learning_rate}\n', fontsize=14, pad=20)
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
        cbar.set_label('Cost J(θ)', fontsize=12)
        
        ax.legend(loc='upper right')
        ax.view_init(elev=30, azim=45)  # Adjust view angle
        
        if animated:
            for angle in range(0, 360, 5):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(0.05)
        
        plt.tight_layout()
        plt.show()
    
    def plot_contour_cost_surface(self, X, y, animated=False):
        """Plot contour of cost function with closer contours near final values, no labels"""
        theta_history = np.array(self.theta_history)
        
        # Dynamic ranges based on theta history with ±10 padding
        theta0_min = min(theta_history[:, 0].min(), 0) - 10
        theta0_max = max(theta_history[:, 0].max(), 0) + 10
        theta1_min = min(theta_history[:, 1].min(), 0) - 10
        theta1_max = max(theta_history[:, 1].max(), 0) + 10
        theta0_range = np.linspace(theta0_min, theta0_max, 300)
        theta1_range = np.linspace(theta1_min, theta1_max, 300)
        
        Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
        
        # Compute cost for each parameter combination
        Z = np.zeros_like(Theta0)
        for i in range(Theta0.shape[0]):
            for j in range(Theta0.shape[1]):
                theta_temp = np.array([[Theta0[i, j]], [Theta1[i, j]]])
                Z[i, j] = self.compute_cost(X, y, theta_temp)
        
        # Non-linear contour levels to emphasize minimum
        min_cost = min(self.cost_history) if self.cost_history else 0
        max_cost = max(self.cost_history) * 2 if self.cost_history else 100
        levels = np.logspace(np.log10(min_cost + 1e-6), np.log10(max_cost), 50)
        
        # Create contour plot without labels
        fig, ax = plt.subplots(figsize=(12, 10))
        contours = ax.contour(Theta0, Theta1, Z, levels=levels, cmap='viridis', alpha=0.7)
        ax.contourf(Theta0, Theta1, Z, levels=levels, cmap='viridis', alpha=0.3)
        
        # Plot gradient descent path
        if len(theta_history) > 0:
            ax.plot(theta_history[:, 0], theta_history[:, 1], 'r.-', 
                    linewidth=3, markersize=6, markeredgecolor='white', 
                    markeredgewidth=1, label='Gradient Descent Path')
            ax.scatter([theta_history[0, 0]], [theta_history[0, 1]], 
                      color='green', s=100, label='Start', edgecolors='white', linewidth=2)
            ax.scatter([theta_history[-1, 0]], [theta_history[-1, 1]], 
                      color='red', s=100, label='End', edgecolors='white', linewidth=2)
        
        ax.set_xlabel('θ₀ (Intercept)', fontsize=12)
        ax.set_ylabel('θ₁ (Slope)', fontsize=12)
        ax.set_title('Contour Plot with Gradient Descent Path\n(Closer Contours Near Minimum)', 
                    fontsize=14, pad=20)
        fig.colorbar(contours, ax=ax, label='Cost J(θ)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if animated:
            step = max(1, len(theta_history) // 20)
            for i in range(0, len(theta_history), step):
                ax.clear()
                ax.contour(Theta0, Theta1, Z, levels=levels, cmap='viridis', alpha=0.7)
                ax.contourf(Theta0, Theta1, Z, levels=levels, cmap='viridis', alpha=0.3)
                ax.plot(theta_history[:i+1, 0], theta_history[:i+1, 1], 'r.-', 
                        linewidth=3, markersize=6, markeredgecolor='white', 
                        markeredgewidth=1, label='Gradient Descent Path')
                ax.scatter([theta_history[0, 0]], [theta_history[0, 1]], 
                          color='green', s=100, label='Start', edgecolors='white', linewidth=2)
                ax.scatter([theta_history[i, 0]], [theta_history[i, 1]], 
                          color='red', s=100, label='Current', edgecolors='white', linewidth=2)
                ax.set_xlabel('θ₀ (Intercept)', fontsize=12)
                ax.set_ylabel('θ₁ (Slope)', fontsize=12)
                ax.set_title(f'Gradient Descent Progress - Iteration {i+1}', fontsize=14)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                plt.draw()
                plt.pause(0.1)
        
        plt.tight_layout()
        plt.show()

    def experiment_learning_rates(self, X, y, learning_rates=[0.001, 0.025, 0.1]):
        """Experiment with different learning rates"""
        plt.figure(figsize=(15, 5))
        
        for idx, lr in enumerate(learning_rates):
            print(f"\nTesting learning rate: {lr}")
            theta, cost_history, theta_history = self.batch_gradient_descent(X, y, learning_rate=lr)
            plt.subplot(1, 3, idx + 1)
            plt.plot(cost_history)
            plt.xlabel('Iterations')
            plt.ylabel('Cost Function J(θ)')
            plt.title(f'Learning Rate = {lr}')
            plt.grid(True, alpha=0.3)
            print(f"Final parameters: θ₀ = {theta[0][0]:.4f}, θ₁ = {theta[1][0]:.4f}")
            print(f"Final cost: {cost_history[-1]:.6f}")
            print(f"Iterations to converge: {len(cost_history)}")
        
        plt.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, lr in enumerate(learning_rates):
            theta, cost_history, theta_history = self.batch_gradient_descent(X, y, learning_rate=lr)
            theta_history = np.array(theta_history)
            theta0_min = min(theta_history[:, 0].min(), 0) - 10
            theta0_max = max(theta_history[:, 0].max(), 0) + 10
            theta1_min = min(theta_history[:, 1].min(), 0) - 10
            theta1_max = max(theta_history[:, 1].max(), 0) + 10
            theta0_range = np.linspace(theta0_min, theta0_max, 50)
            theta1_range = np.linspace(theta1_min, theta1_max, 50)
            Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
            Z = np.zeros_like(Theta0)
            for i in range(Theta0.shape[0]):
                for j in range(Theta0.shape[1]):
                    theta_temp = np.array([[Theta0[i, j]], [Theta1[i, j]]])
                    Z[i, j] = self.compute_cost(X, y, theta_temp)
            
            # Non-linear contour levels with more density near minimum
            min_cost = min(cost_history) if cost_history else 0
            max_cost = max(cost_history) * 1.5 if cost_history else 100  # Reduced max_cost for tighter range
            levels = np.logspace(np.log10(min_cost + 1e-6), np.log10(max_cost), 30)
            
            axes[idx].contour(Theta0, Theta1, Z, levels=levels, cmap='viridis', alpha=0.7)
            axes[idx].contourf(Theta0, Theta1, Z, levels=levels, cmap='viridis', alpha=0.3)
            axes[idx].plot(theta_history[:, 0], theta_history[:, 1], 'r.-', 
                          linewidth=2, markersize=3, markeredgecolor='white', 
                          markeredgewidth=1, label='Path')
            axes[idx].scatter([theta_history[0, 0]], [theta_history[0, 1]], 
                             color='green', s=100, label='Start', edgecolors='white', linewidth=2)
            axes[idx].scatter([theta_history[-1, 0]], [theta_history[-1, 1]], 
                             color='red', s=100, label='End', edgecolors='white', linewidth=2)
            axes[idx].set_xlabel('θ₀')
            axes[idx].set_ylabel('θ₁')
            axes[idx].set_title(f'Learning Rate = {lr}')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    lr_model = LinearRegression()
    X, y = lr_model.load_data()
    print("Data loaded successfully!")
    print(f"Number of training examples: {len(y)}")
    
    print("\n" + "="*50)
    print("1. BATCH GRADIENT DESCENT")
    print("="*50)
    
    global learning_rate
    learning_rate = 0.01
    theta_final, cost_history, theta_history = lr_model.batch_gradient_descent(
        X, y, learning_rate=learning_rate
    )
    
    print(f"Learning rate used: {learning_rate}")
    print(f"Stopping criteria: Change in cost < 1e-8")
    print(f"Final parameters: θ₀ = {theta_final[0][0]:.6f}, θ₁ = {theta_final[1][0]:.6f}")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Number of iterations: {len(cost_history)}")
    
    print("\n2. Plotting data and fitted line...")
    lr_model.plot_data_and_hypothesis(X, y)
    
    print("\n3. Plotting 3D cost surface...")
    lr_model.plot_3d_cost_surface(X, y, animated=False)
    
    print("\n4. Plotting contour plots...")
    lr_model.plot_contour_cost_surface(X, y, animated=False)
    
    print("\n" + "="*50)
    print("5. LEARNING RATE EXPERIMENTS")
    print("="*50)
    lr_model.experiment_learning_rates(X, y, learning_rates=[0.001, 0.025, 0.1])

if __name__ == "__main__":
    main()
