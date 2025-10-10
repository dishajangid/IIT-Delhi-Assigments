# part3.py

import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def run(X_train_binary, y_train_binary, X_test_binary, y_test_binary, svm_linear, svm_gaussian, n_sv_linear, n_sv_gaussian, test_acc_linear, test_acc_gaussian, train_time_linear, train_time_gaussian):
    print("\n" + "=" * 80)
    print("PART 3: SCIKIT-LEARN SVM (LIBSVM)")
    print("=" * 80)

    # Linear kernel
    print("\nTraining sklearn SVM with linear kernel...")
    start_time = time.time()

    sklearn_linear = SVC(kernel='linear', C=1.0)
    sklearn_linear.fit(X_train_binary, y_train_binary)

    sklearn_linear_time = time.time() - start_time
    print(f"Training completed in {sklearn_linear_time:.2f} seconds")

    # Gaussian kernel
    print("\nTraining sklearn SVM with Gaussian kernel...")
    start_time = time.time()

    sklearn_gaussian = SVC(kernel='rbf', C=1.0, gamma=0.001)
    sklearn_gaussian.fit(X_train_binary, y_train_binary)

    sklearn_gaussian_time = time.time() - start_time
    print(f"Training completed in {sklearn_gaussian_time:.2f} seconds")

    # Part 3(a): Compare support vectors
    n_sv_sklearn_linear = len(sklearn_linear.support_)
    n_sv_sklearn_gaussian = len(sklearn_gaussian.support_)

    print(f"\nPart 3(a) - Support Vector Comparison:")
    print(f"  Linear kernel:")
    print(f"    CVXOPT: {n_sv_linear} SVs")
    print(f"    sklearn: {n_sv_sklearn_linear} SVs")
    print(f"    Difference: {abs(n_sv_linear - n_sv_sklearn_linear)} SVs")

    # Check matching SVs for linear
    sklearn_linear_sv_indices = set(sklearn_linear.support_)
    cvxopt_linear_sv_indices = set(svm_linear.get_support_vector_indices(X_train_binary))
    matching_linear = len(sklearn_linear_sv_indices & cvxopt_linear_sv_indices)
    print(f"    Matching SVs: {matching_linear}")

    print(f"\n  Gaussian kernel:")
    print(f"    CVXOPT: {n_sv_gaussian} SVs")
    print(f"    sklearn: {n_sv_sklearn_gaussian} SVs")
    print(f"    Difference: {abs(n_sv_gaussian - n_sv_sklearn_gaussian)} SVs")

    # Check matching SVs for Gaussian
    sklearn_gaussian_sv_indices = set(sklearn_gaussian.support_)
    cvxopt_gaussian_sv_indices = set(svm_gaussian.get_support_vector_indices(X_train_binary))
    matching_gaussian = len(sklearn_gaussian_sv_indices & cvxopt_gaussian_sv_indices)
    print(f"    Matching SVs: {matching_gaussian}")

    # Part 3(b): Compare weights and bias for linear kernel
    print(f"\nPart 3(b) - Linear Kernel Parameters:")
    print(f"  Weight vector (w):")
    print(f"    CVXOPT norm: {np.linalg.norm(svm_linear.w):.6f}")
    print(f"    sklearn norm: {np.linalg.norm(sklearn_linear.coef_[0]):.6f}")
    
    # Cosine similarity
    cosine_sim = np.dot(svm_linear.w, sklearn_linear.coef_[0]) / (np.linalg.norm(svm_linear.w) * np.linalg.norm(sklearn_linear.coef_[0]))
    print(f"    Cosine similarity: {cosine_sim:.6f}")

    print(f"  Bias (b):")
    print(f"    CVXOPT: {svm_linear.b:.6f}")
    print(f"    sklearn: {sklearn_linear.intercept_[0]:.6f}")
    print(f"    Difference: {abs(svm_linear.b - sklearn_linear.intercept_[0]):.6f}")

    # Part 3(c): Test accuracy
    y_pred_sklearn_linear = sklearn_linear.predict(X_test_binary)
    test_acc_sklearn_linear = accuracy_score(y_test_binary, y_pred_sklearn_linear)
    train_acc_sklearn_linear = accuracy_score(y_train_binary, sklearn_linear.predict(X_train_binary))

    y_pred_sklearn_gaussian = sklearn_gaussian.predict(X_test_binary)
    test_acc_sklearn_gaussian = accuracy_score(y_test_binary, y_pred_sklearn_gaussian)
    train_acc_sklearn_gaussian = accuracy_score(y_train_binary, sklearn_gaussian.predict(X_train_binary))

    print(f"\nPart 3(c) - Accuracy Comparison:")
    print(f"  Linear kernel:")
    print(f"    CVXOPT test accuracy: {test_acc_linear:.4f}")
    print(f"    sklearn test accuracy: {test_acc_sklearn_linear:.4f}")
    print(f"    CVXOPT training accuracy: {train_acc_sklearn_linear:.4f}")
    print(f"    sklearn training accuracy: {train_acc_sklearn_linear:.4f}")
    print(f"    Difference: {abs(test_acc_linear - test_acc_sklearn_linear):.4f}")
    
    print(f"\n  Gaussian kernel:")
    print(f"    CVXOPT test accuracy: {test_acc_gaussian:.4f}")
    print(f"    sklearn test accuracy: {test_acc_sklearn_gaussian:.4f}")
    print(f"    CVXOPT training accuracy: {train_acc_sklearn_gaussian:.4f}")
    print(f"    sklearn training accuracy: {train_acc_sklearn_gaussian:.4f}")
    print(f"    Difference: {abs(test_acc_gaussian - test_acc_sklearn_gaussian):.4f}")

    # Part 3(d): Training time comparison
    print(f"\nPart 3(d) - Computational Cost (Training Time):")
    print(f"  Linear kernel:")
    print(f"    CVXOPT: {train_time_linear:.2f} seconds")
    print(f"    sklearn: {sklearn_linear_time:.2f} seconds")
    print(f"    Speedup: {train_time_linear/sklearn_linear_time:.2f}x (sklearn is faster)" if train_time_linear > sklearn_linear_time else f"    Slowdown: {sklearn_linear_time/train_time_linear:.2f}x")

    print(f"\n  Gaussian kernel:")
    print(f"    CVXOPT: {train_time_gaussian:.2f} seconds")
    print(f"    sklearn: {sklearn_gaussian_time:.2f} seconds")
    print(f"    Speedup: {train_time_gaussian/sklearn_gaussian_time:.2f}x (sklearn is faster)" if train_time_gaussian > sklearn_gaussian_time else f"    Slowdown: {sklearn_gaussian_time/train_time_gaussian:.2f}x")

    # Summary table
    summary_data = {
        'Method': ['CVXOPT Linear', 'sklearn Linear', 'CVXOPT Gaussian', 'sklearn Gaussian'],
        'Train Time (s)': [f"{train_time_linear:.2f}", f"{sklearn_linear_time:.2f}", 
                          f"{train_time_gaussian:.2f}", f"{sklearn_gaussian_time:.2f}"],
        'Num SVs': [n_sv_linear, n_sv_sklearn_linear, n_sv_gaussian, n_sv_sklearn_gaussian],
        'Test Acc': [f"{test_acc_linear:.4f}", f"{test_acc_sklearn_linear:.4f}", 
                    f"{test_acc_gaussian:.4f}", f"{test_acc_sklearn_gaussian:.4f}"],
        'Train Acc': [f"{train_acc_sklearn_linear:.4f}", f"{train_acc_sklearn_linear:.4f}",
                     f"{train_acc_sklearn_gaussian:.4f}", f"{train_acc_sklearn_gaussian:.4f}"]
    }

    summary_df = pd.DataFrame(summary_data)
    print("\n" + "=" * 80)
    print("BINARY CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    return sklearn_linear_time, test_acc_sklearn_gaussian