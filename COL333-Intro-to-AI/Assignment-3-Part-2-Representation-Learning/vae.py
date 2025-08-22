import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
# from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import skimage.metrics as image_metrics  # For SSIM
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


# Set manual seed for reproducibility
torch.manual_seed(0)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset from .npz file
class NPZDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = data['data'] / 255.0  # Normalize to [0, 1]
        self.labels = data['labels'] if 'labels' in data else None  # Optional labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        return image

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=400, latent_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Loss function combining reconstruction and KL divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# VAE training function
def train_vae(model, train_loader, num_epochs=120, learning_rate=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset)}')
    torch.save(model.state_dict(), "vae.pth")

# Manually implement Gaussian Mixture Model (GMM)
class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        """
        Custom Gaussian Mixture Model implementation.

        Parameters:
        - n_components: Number of clusters (Gaussian components).
        - max_iter: Maximum number of iterations for the EM algorithm.
        - tol: Convergence tolerance for log-likelihood improvement.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        
    def initialize_parameters(self, X):
        """
        Initialize GMM parameters randomly.

        Parameters:
        - X: Input data (N x D), where N is the number of samples and D is the dimensionality.
        """
        N, D = X.shape
        self.means_ = X[np.random.choice(N, self.n_components, replace=False)]
        self.covariances_ = np.array([np.eye(D) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components
        
    def e_step(self, X):
        """
        Perform the Expectation step of the EM algorithm.

        Parameters:
        - X: Input data (N x D).

        Returns:
        - responsibilities: N x K matrix where each row sums to 1.
        """
        N, _ = X.shape
        self.responsibilities = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            self.responsibilities[:, k] = self.weights_[k] * multivariate_normal.pdf(X, mean=self.means_[k], cov=self.covariances_[k])

        total_responsibility = self.responsibilities.sum(axis=1, keepdims=True)
        self.responsibilities /= total_responsibility

    def m_step(self, X):
        """
        Perform the Maximization step of the EM algorithm.

        Parameters:
        - X: Input data (N x D).
        """
        N, D = X.shape
        Nk = self.responsibilities.sum(axis=0)

        # Update weights
        self.weights_ = Nk / N

        # Update means
        self.means_ = (self.responsibilities.T @ X) / Nk[:, None]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (self.responsibilities[:, k][:, None] * diff).T @ diff / Nk[k]
    
    
    def log_likelihood(self, X):
        """
        Compute the log-likelihood of the data under the current GMM.

        Parameters:
        - X: Input data (N x D).

        Returns:
        - log_likelihood: Scalar log-likelihood value.
        """
        N, _ = X.shape
        log_likelihood = 0

        for k in range(self.n_components):
            log_likelihood += self.weights_[k] * multivariate_normal.pdf(X, mean=self.means_[k], cov=self.covariances_[k])

        return np.log(log_likelihood).sum()
    

    # def fit(self, X):
    #     n_samples, n_features = X.shape
    #     np.random.seed(0)

    #     # Initialize means, covariances, and weights
    #     self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
    #     self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
    #     self.weights_ = np.ones(self.n_components) / self.n_components

    #     log_likelihood = 0
    #     for _ in range(self.max_iters):
    #         # E-step: Calculate responsibilities
    #         responsibilities = np.zeros((n_samples, self.n_components))
    #         for k in range(self.n_components):
    #             responsibilities[:, k] = self.weights_[k] * self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
    #         responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    #         # M-step: Update parameters
    #         Nk = responsibilities.sum(axis=0)
    #         self.weights_ = Nk / n_samples
    #         self.means_ = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
    #         for k in range(self.n_components):
    #             X_centered = X - self.means_[k]
    #             self.covariances_[k] = (responsibilities[:, k][:, np.newaxis] * X_centered).T @ X_centered / Nk[k]

    #         # Check convergence
    #         new_log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
    #         if np.abs(new_log_likelihood - log_likelihood) < self.tol:
    #             break
    #         log_likelihood = new_log_likelihood

    # def predict(self, X):
    #     responsibilities = np.zeros((X.shape[0], self.n_components))
    #     for k in range(self.n_components):
    #         responsibilities[:, k] = self.weights_[k] * self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
    #     return np.argmax(responsibilities, axis=1)
    
    def fit(self, X):
        """
        Fit the GMM to the data using the EM algorithm.

        Parameters:
        - X: Input data (N x D).
        """
        self.initialize_parameters(X)

        log_likelihood = -np.inf
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            new_log_likelihood = self.log_likelihood(X)

            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood
            
    def predict(self, X):
        """
        Predict the cluster assignments for the data.

        Parameters:
        - X: Input data (N x D).

        Returns:
        - labels: Array of cluster labels for each sample.
        """
        self.e_step(X)
        return np.argmax(self.responsibilities, axis=1)

    @staticmethod
    def _multivariate_gaussian(X, mean, cov):
        n_features = X.shape[1]
        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1 / ((2 * np.pi) ** (n_features / 2) * np.sqrt(det_cov))
        X_centered = X - mean
        return norm_const * np.exp(-0.5 * np.sum(X_centered @ cov_inv * X_centered, axis=1))



# GMM Evaluation Metrics Function
def evaluate_gmm_performance(labels_true, labels_pred):
    accuracy = accuracy_score(labels_true, labels_pred)
    precision_macro = precision_score(labels_true, labels_pred, average='macro', zero_division=1)
    recall_macro = recall_score(labels_true, labels_pred, average='macro', zero_division=1)
    f1_macro = f1_score(labels_true, labels_pred, average='macro', zero_division=1)
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

# Reconstruction and evaluation using MSE and SSIM
# Reconstruction and evaluation using MSE and SSIM
def generate_reconstructions(model, val_loader, save_path='vae_reconstructed.npz'):
    model.to(device)
    model.eval()
    reconstructed_images = []
    original_images = []
    mse_total = 0
    ssim_total = 0

    with torch.no_grad():
        for data in val_loader:
            # Check if data is a tuple or list and unpack accordingly
            if isinstance(data, (tuple, list)):
                data = data[0]  # Get the image part only

            # Move data to device
            data = data.to(device)

            # Forward pass through VAE to get reconstruction
            recon_batch, _, _ = model(data)
            reconstructed_images.append(recon_batch.view(-1, 28, 28).cpu().numpy())
            original_images.append(data.view(-1, 28, 28).cpu().numpy())

            # Compute MSE and SSIM
            for orig, recon in zip(original_images[-1], reconstructed_images[-1]):
                mse_total += np.mean((orig - recon) ** 2)
                ssim_total += image_metrics.structural_similarity(orig, recon, data_range=1.0)

    # Save reconstructed images
    np.savez(save_path, images=np.concatenate(reconstructed_images))

    # Calculate and print average MSE and SSIM
    num_images = len(original_images) * original_images[0].shape[0]
    print(f"Average MSE: {mse_total / num_images}")
    print(f"Average SSIM: {ssim_total / num_images}")
    
###

def plot_reconstructions(model, data_loader, n=10):
    model.eval()
    originals, reconstructions = [], []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, _, _ = model(data)
            originals.append(data.cpu())
            reconstructions.append(recon_batch.cpu())
            if len(originals) >= n:
                break

    # Flatten and concatenate images for easier plotting
    originals = torch.cat(originals)[:n].view(n, 28, 28)
    reconstructions = torch.cat(reconstructions)[:n].view(n, 28, 28)

    plt.figure(figsize=(15, 4))
    for i in range(n):
        # Original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(originals[i], cmap='gray')
        ax.axis('off')

        # Reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i], cmap='gray')
        ax.axis('off')

    plt.suptitle("Original (Top Row) vs. Reconstructed Images (Bottom Row)")
    plt.show()
    
    
    
def plot_latent_space_with_labels(model, data_loader):
    model.eval()
    latents, labels = [], []
    
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 28 * 28))
            latents.append(mu.cpu().numpy())
            labels.append(label.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, label="True Labels")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space with True Labels")
    plt.show()
    
def plot_image_manifold(model, latent_dim=2, n=20, grid_range=2.0):
    model.eval()
    grid_x = np.linspace(-grid_range, grid_range, n)
    grid_y = np.linspace(-grid_range, grid_range, n)

    figure = np.zeros((28 * n, 28 * n))
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                generated_img = model.decode(z).view(28, 28).cpu().numpy()
                figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = generated_img

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.title("Image Manifold in Latent Space")
    plt.axis('off')
    plt.show()
    

def plot_gmm_clusters_with_parameters(model, gmm, data_loader):
    model.eval()
    latents, true_labels = [], []

    # Collect latent representations of validation data
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 28 * 28))
            latents.append(mu.cpu().numpy())
            true_labels.append(label.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Plot data points in latent space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=true_labels, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, label="True Labels")

    # Plot GMM ellipses and print GMM parameters
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        eigvals, eigvecs = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals)
        
        ellip = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='blue', fc='cyan', lw=2, alpha=0.3)
        plt.gca().add_patch(ellip)
        
        # Plot mean point and display parameters
        plt.plot(mean[0], mean[1], 'ro')
        print(f"Cluster {i+1}:")
        print(f"  Mean = {mean}")
        print(f"  Covariance Matrix =\n{covar}")
        print(f"  Mixing Coefficient = {gmm.weights_[i]}\n")

    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("GMM Clusters with Means and Covariances in Latent Space")
    plt.show()


###


# # Classification with GMM and saving predictions to CSV
# def classify_with_gmm(model, gmm, data_loader, save_path='vae_predictions.csv'):
#     model.to(device)
#     model.eval()
#     latents = []
#     true_labels = []
#     with torch.no_grad():
#         for data, labels in data_loader:
#             data = data.to(device)
#             mu, _ = model.encode(data.view(-1, 28 * 28))
#             latents.append(mu.cpu().numpy())
#             true_labels.append(labels.cpu().numpy())
#     latents = np.concatenate(latents, axis=0)
#     true_labels = np.concatenate(true_labels, axis=0)
#     predictions = gmm.predict(latents)

#     # Save predictions to CSV
#     df = pd.DataFrame(predictions, columns=['Predicted_Label'])
#     df.to_csv(save_path, index=False)

#     # Calculate and print evaluation metrics
#     metrics = evaluate_gmm_performance(true_labels, predictions)
#     print(f"Accuracy: {metrics['accuracy']:.4f}")
#     print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
#     print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
#     print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    
# Classification with Custom GMM and Saving Predictions to CSV
def classify_with_custom_gmm(model, gmm, data_loader, save_path='vae_predictions.csv'):
    model.to(device)
    model.eval()
    latents = []
    true_labels = []

    # Extract latent features
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 28 * 28))
            latents.append(mu.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    
    # Predict labels using the custom GMM
    predictions = gmm.predict(latents)

    # Save predictions to CSV
    df = pd.DataFrame(predictions, columns=['Predicted_Label'])
    df.to_csv(save_path, index=False)

    # Calculate and print evaluation metrics
    metrics = evaluate_gmm_performance(true_labels, predictions)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")

# Main section for argument parsing and running different modes
if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None

    print(f"arg1:{arg1}, arg2:{arg2}, arg3:{arg3}, arg4:{arg4}, arg5:{arg5}")

    if len(sys.argv) == 4:
        # Running code for VAE reconstruction
        path_to_test_dataset_recon = arg1
        test_reconstruction = arg2
        vaePath = arg3

        # Load the model and reconstruct images
        model = VAE()
        model.load_state_dict(torch.load(vaePath))
        val_dataset = NPZDataset(path_to_test_dataset_recon)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        generate_reconstructions(model, val_loader, save_path=test_reconstruction)
        plot_latent_space_with_labels(model, val_loader)
        plot_reconstructions(model, val_loader)
        plot_image_manifold(model)
        
        
        
    elif len(sys.argv) == 5:
        # Running code for class prediction during testing
        path_to_test_dataset = arg1
        test_classifier = arg2
        vaePath = arg3
        gmmPath = arg4

        # Load the model and GMM
        model = VAE()
        model.load_state_dict(torch.load(vaePath))
        with open(gmmPath, 'rb') as f:
            gmm = pickle.load(f)

        # Perform classification and save predictions
        test_dataset = NPZDataset(path_to_test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        # classify_with_gmm(model, gmm, test_loader, save_path='vae_predictions.csv')
        classify_with_custom_gmm(model, gmm, test_loader, save_path='vae_predictions.csv')
        
    else:
        # Running code for training
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmPath = arg5

        # Instantiate and train the VAE model
        model = VAE()
        train_dataset = NPZDataset(path_to_train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        train_vae(model, train_loader)

        # Extract latent features and train GMM
        latents = []
        model.eval()
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(device)
                mu, _ = model.encode(data.view(-1, 28 * 28))
                latents.append(mu.cpu().numpy())
        latents = np.concatenate(latents, axis=0)
        # gmm = GaussianMixture(n_components=3, random_state=0)
        # Train custom GMM
        gmm = GMM(n_components=3, max_iter=100, tol=1e-3)
        gmm.fit(latents)

        # Save GMM parameters
        with open(gmmPath, 'wb') as f:
            pickle.dump(gmm, f)
