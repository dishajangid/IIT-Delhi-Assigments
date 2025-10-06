import cvxopt
import numpy as np

cvxopt.solvers.options['show_progress'] = False

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    Implementation using CVXOPT for solving the dual optimization problem
    '''
    def __init__(self):
        # pass
        self.alphas = None          # Lagrange multipliers
        self.support_vectors = None # Support vectors (X values)
        self.support_labels = None  # Support vector labels (y values)
        self.support_alphas = None  # Alpha values for support vectors
        self.w = None              # Weight vector (only for linear kernel)
        self.b = None              # Bias term
        self.kernel_type = None    # Type of kernel used
        self.C = None              # Regularization parameter
        self.gamma = None          # Gamma for Gaussian kernel
        self.X_train = None        # Training data (needed for Gaussian kernel)
    

    def _linear_kernel(self, X1, X2):
        """
        Compute linear kernel: K(x1, x2) = x1^T * x2
        
        Args:
            X1: np.array of shape (N1, D)
            X2: np.array of shape (N2, D)
            
        Returns:
            np.array of shape (N1, N2) containing kernel values
        """
        return np.dot(X1, X2.T)

    def _gaussian_kernel(self, X1, X2):
        """
        Compute Gaussian (RBF) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
        
        Args:
            X1: np.array of shape (N1, D)
            X2: np.array of shape (N2, D)
            
        Returns:
            np.array of shape (N1, N2) containing kernel values
        """
        # Efficient computation using broadcasting
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1^T*x2
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)  # (N1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)  # (1, N2)
        cross_term = np.dot(X1, X2.T)                  # (N1, N2)
        
        sq_distances = X1_sq + X2_sq - 2 * cross_term
        return np.exp(-self.gamma * sq_distances)
    

    def _compute_kernel(self, X1, X2):
        """
        Compute kernel matrix based on kernel type.
        
        Args:
            X1: np.array of shape (N1, D)
            X2: np.array of shape (N2, D)
            
        Returns:
            np.array of shape (N1, N2) containing kernel values
        """
        if self.kernel_type == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel_type == 'gaussian':
            return self._gaussian_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    

        
    def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel, ignored for linear kernel
        '''
        # pass
        # Store parameters
        self.kernel_type = kernel
        self.C = C
        self.gamma = gamma
        self.X_train = X.copy()
        
        # Convert labels from {0, 1} to {-1, 1} for SVM formulation
        N, D = X.shape
        y_svm = np.where(y == 0, -1, 1).astype(np.float64)
        
        # Compute kernel matrix K(x_i, x_j) for all training samples
        K = self._compute_kernel(X, X)
        
        # Construct matrices for CVXOPT quadratic programming
        # Standard form: minimize (1/2)x^T P x + q^T x
        # subject to: Gx <= h and Ax = b
        
        # For SVM dual, we want to maximize the dual objective, which is equivalent to
        # minimizing its negative. The dual objective in matrix form is:
        # maximize: alpha^T * 1 - (1/2) * alpha^T * (y*y^T ⊙ K) * alpha
        # This becomes: minimize (1/2) * alpha^T * P * alpha + q^T * alpha
        # where P = y*y^T ⊙ K (element-wise product), q = -1 (vector of -1s)
        
        # P matrix: (y_i * y_j * K(x_i, x_j))
        P = np.outer(y_svm, y_svm) * K
        P = P.astype(np.float64)
        
        # q vector: all -1s (we want to maximize sum of alphas)
        q = -np.ones(N)
        
        # Inequality constraints: -alpha_i <= 0 and alpha_i <= C
        # This is written as: G * alpha <= h
        # G = [-I; I] (negative identity on top, identity on bottom)
        # h = [0; C*1] (zeros on top, C on bottom)
        G = np.vstack([-np.eye(N), np.eye(N)])
        h = np.hstack([np.zeros(N), np.ones(N) * C])
        
        # Equality constraint: sum(alpha_i * y_i) = 0
        # This is written as: A * alpha = b
        A = y_svm.reshape(1, -1)
        b = np.array([0.0])
        
        # Convert to cvxopt matrices
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        
        # Solve the quadratic programming problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Extract alphas from solution
        alphas = np.array(solution['x']).flatten()
        self.alphas = alphas
        
        # Identify support vectors (alphas > threshold)
        # Use a small threshold to account for numerical errors
        sv_threshold = 1e-5
        sv_indices = alphas > sv_threshold
        
        self.support_vectors = X[sv_indices]
        self.support_labels = y_svm[sv_indices]
        self.support_alphas = alphas[sv_indices]
        
        # Compute bias term b
        # For samples with 0 < alpha < C (margin support vectors), we have:
        # y_i * (w^T * x_i + b) = 1, so b = y_i - w^T * x_i
        # For Gaussian kernel, b = y_i - sum(alpha_j * y_j * K(x_j, x_i))
        
        # Find margin support vectors (0 < alpha < C)
        margin_sv_indices = (alphas > sv_threshold) & (alphas < C - sv_threshold)
        
        if np.sum(margin_sv_indices) > 0:
            # Use margin support vectors to compute b
            margin_sv_X = X[margin_sv_indices]
            margin_sv_y = y_svm[margin_sv_indices]
            
            # Compute kernel between margin SVs and all support vectors
            K_margin = self._compute_kernel(margin_sv_X, self.support_vectors)
            
            # b = y_i - sum(alpha_j * y_j * K(x_j, x_i))
            b_values = margin_sv_y - np.sum(
                self.support_alphas * self.support_labels * K_margin, axis=1
            )
            self.b = np.mean(b_values)
        else:
            # Fallback: use all support vectors
            K_sv = self._compute_kernel(self.support_vectors, self.support_vectors)
            b_values = self.support_labels - np.sum(
                self.support_alphas * self.support_labels * K_sv, axis=1
            )
            self.b = np.mean(b_values)
        
        # For linear kernel, compute weight vector w explicitly
        if self.kernel_type == 'linear':
            self.w = np.sum(
                (self.support_alphas * self.support_labels)[:, np.newaxis] * self.support_vectors,
                axis=0
            )
    

    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''
        
        # pass
        if self.kernel_type == 'linear':
            # For linear kernel: y = sign(w^T * x + b)
            decision_values = np.dot(X, self.w) + self.b
        else:
            # For Gaussian kernel: y = sign(sum(alpha_i * y_i * K(x, x_i)) + b)
            K = self._compute_kernel(X, self.support_vectors)
            decision_values = np.sum(
                self.support_alphas * self.support_labels * K, axis=1
            ) + self.b
        
        # Convert from {-1, 1} back to {0, 1}
        predictions = np.where(decision_values >= 0, 1, 0)
        
        return predictions


    def get_num_support_vectors(self):
        """
        Get the number of support vectors.
        
        Returns:
            int: Number of support vectors
        """
        return len(self.support_vectors) if self.support_vectors is not None else 0

    
    def get_support_vector_indices(self, X_original):
        """
        Get indices of support vectors in the original training data.
        
        Args:
            X_original: Original training data
            
        Returns:
            np.array: Indices of support vectors
        """
        if self.support_vectors is None:
            return np.array([])
        
        # Find matching rows
        indices = []
        for sv in self.support_vectors:
            # Find rows in X_original that match this support vector
            matches = np.where(np.all(X_original == sv, axis=1))[0]
            if len(matches) > 0:
                indices.append(matches[0])
        
        return np.array(indices)

        
class MultiClassSVM:
    """
    Multi-class SVM using One-vs-One strategy.
    Trains k*(k-1)/2 binary classifiers for k classes.
    """
    
    def __init__(self):
        """Initialize multi-class SVM."""
        self.classifiers = {}  # Dictionary to store binary classifiers
        self.classes = None    # Unique class labels
        self.class_pairs = []  # List of class pairs
        
    def fit(self, X, y, kernel='linear', C=1.0, gamma=0.001):
        """
        Train one-vs-one multi-class SVM.
        
        Args:
            X: np.array of shape (N, D)
            y: np.array of shape (N,) with class labels
            kernel: str - 'linear' or 'gaussian'
            C: float - regularization parameter
            gamma: float - gamma for Gaussian kernel
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Train binary classifier for each pair of classes
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i = self.classes[i]
                class_j = self.classes[j]
                
                # Get samples for these two classes
                mask = (y == class_i) | (y == class_j)
                X_pair = X[mask]
                y_pair = y[mask]
                
                # Convert to binary labels: class_i -> 0, class_j -> 1
                y_binary = np.where(y_pair == class_i, 0, 1)
                
                # Train binary SVM
                clf = SupportVectorMachine()
                clf.fit(X_pair, y_binary, kernel=kernel, C=C, gamma=gamma)
                
                # Store classifier
                self.classifiers[(class_i, class_j)] = clf
                self.class_pairs.append((class_i, class_j))
        
    def predict(self, X):
        """
        Predict classes using majority voting.
        
        Args:
            X: np.array of shape (N, D)
            
        Returns:
            np.array of shape (N,) with predicted class labels
        """
        N = X.shape[0]
        votes = np.zeros((N, len(self.classes)))
        
        # Get votes from each binary classifier
        for (class_i, class_j), clf in self.classifiers.items():
            predictions = clf.predict(X)
            
            # prediction = 0 means class_i, prediction = 1 means class_j
            i_idx = np.where(self.classes == class_i)[0][0]
            j_idx = np.where(self.classes == class_j)[0][0]
            
            for idx in range(N):
                if predictions[idx] == 0:
                    votes[idx, i_idx] += 1
                else:
                    votes[idx, j_idx] += 1
        
        # Return class with most votes
        predicted_indices = np.argmax(votes, axis=1)
        return self.classes[predicted_indices]
    
    def get_total_support_vectors(self):
        """
        Get total number of support vectors across all binary classifiers.
        
        Returns:
            int: Total number of support vectors
        """
        total = 0
        for clf in self.classifiers.values():
            total += clf.get_num_support_vectors()
        return total
    
