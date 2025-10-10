# part1.py

import time
import numpy as np
from svm import SupportVectorMachine
from sklearn.metrics import accuracy_score
from helpers import plot_images
import matplotlib.pyplot as plt

def run(X_train_binary, y_train_binary, X_test_binary, y_test_binary):
    print("\n" + "=" * 80)
    print("PART 1: LINEAR KERNEL WITH CVXOPT")
    print("=" * 80)

    print("\nTraining SVM with linear kernel using CVXOPT...")
    start_time = time.time()

    svm_linear = SupportVectorMachine()
    svm_linear.fit(X_train_binary, y_train_binary, kernel='linear', C=1.0)

    train_time_linear = time.time() - start_time
    print(f"Training completed in {train_time_linear:.2f} seconds")

    # Part 1(a): Number of support vectors
    n_sv_linear = svm_linear.get_num_support_vectors()
    pct_sv_linear = (n_sv_linear / len(X_train_binary)) * 100

    print(f"\nPart 1(a) - Support Vectors:")
    print(f"  Number of support vectors: {n_sv_linear}")
    print(f"  Percentage of training samples: {pct_sv_linear:.2f}%")

    # Part 1(b): Weight vector and bias
    print(f"\nPart 1(b) - Model Parameters:")
    print(f"  Weight vector shape: {svm_linear.w.shape}")
    print(f"  Weight vector norm: {np.linalg.norm(svm_linear.w):.6f}")
    print(f"  Bias term (b): {svm_linear.b:.6f}")

    # Test accuracy
    y_pred_linear = svm_linear.predict(X_test_binary)
    test_acc_linear = accuracy_score(y_test_binary, y_pred_linear)
    print(f"  Test accuracy: {test_acc_linear:.4f}")

    # Training accuracy
    y_train_pred_linear = svm_linear.predict(X_train_binary)
    train_acc_linear = accuracy_score(y_train_binary, y_train_pred_linear)
    print(f"  Training accuracy: {train_acc_linear:.4f}")

    # Part 1(c): Visualize support vectors and weight vector
    print(f"\nPart 1(c) - Visualizing support vectors and weight vector...")

    # Get top 5 support vectors by alpha coefficient
    top_5_indices = np.argsort(svm_linear.support_alphas)[-5:][::-1]
    top_5_sv = svm_linear.support_vectors[top_5_indices]

    fig = plot_images(top_5_sv,
                      titles=[f"SV {i+1} (α={svm_linear.support_alphas[idx]:.4f})" 
                             for i, idx in enumerate(top_5_indices)],
                      n_cols=5, figsize=(15, 3))
    plt.suptitle('Top 5 Support Vectors (Linear Kernel)', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('output/part1c_top5_support_vectors_linear.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Top 5 support vectors saved to 'output/part1c_top5_support_vectors_linear.png'")

    # Visualize weight vector
    fig = plot_images([svm_linear.w], titles=['Weight Vector (w)'], n_cols=1, figsize=(4, 4))
    plt.savefig('output/part1c_weight_vector.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Weight vector visualization saved to 'output/part1c_weight_vector.png'")

    return svm_linear, test_acc_linear, train_acc_linear, n_sv_linear, train_time_linear