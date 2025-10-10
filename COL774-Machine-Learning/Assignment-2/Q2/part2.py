# part2.py

import time
import numpy as np
from svm import SupportVectorMachine
from sklearn.metrics import accuracy_score
from helpers import plot_images
import matplotlib.pyplot as plt

def run(X_train_binary, y_train_binary, X_test_binary, y_test_binary, svm_linear, test_acc_linear, n_sv_linear):
    print("\n" + "=" * 80)
    print("PART 2: GAUSSIAN KERNEL WITH CVXOPT")
    print("=" * 80)

    print("\nTraining SVM with Gaussian kernel using CVXOPT...")
    start_time = time.time()

    svm_gaussian = SupportVectorMachine()
    svm_gaussian.fit(X_train_binary, y_train_binary, kernel='gaussian', C=1.0, gamma=0.001)

    train_time_gaussian = time.time() - start_time
    print(f"Training completed in {train_time_gaussian:.2f} seconds")

    # Part 2(a): Number of support vectors
    n_sv_gaussian = svm_gaussian.get_num_support_vectors()
    pct_sv_gaussian = (n_sv_gaussian / len(X_train_binary)) * 100

    print(f"\nPart 2(a) - Support Vectors:")
    print(f"  Number of support vectors: {n_sv_gaussian}")
    print(f"  Percentage of training samples: {pct_sv_gaussian:.2f}%")
    print(f"  Comparison with linear kernel:")
    print(f"    Linear: {n_sv_linear} SVs ({(n_sv_linear/len(X_train_binary))*100:.2f}%)")
    print(f"    Gaussian: {n_sv_gaussian} SVs ({pct_sv_gaussian:.2f}%)")
    print(f"    Difference: {n_sv_gaussian - n_sv_linear} ({'more' if n_sv_gaussian > n_sv_linear else 'fewer'} SVs)")

    # Check how many support vectors match
    sv_indices_linear = svm_linear.get_support_vector_indices(X_train_binary)
    sv_indices_gaussian = svm_gaussian.get_support_vector_indices(X_train_binary)
    matching_sv = len(set(sv_indices_linear) & set(sv_indices_gaussian))
    print(f"    Matching support vectors: {matching_sv}")

    # Part 2(b): Test accuracy
    y_pred_gaussian = svm_gaussian.predict(X_test_binary)
    test_acc_gaussian = accuracy_score(y_test_binary, y_pred_gaussian)
    print(f"\nPart 2(b) - Test Accuracy:")
    print(f"  Test accuracy: {test_acc_gaussian:.4f}")

    # Training accuracy
    y_train_pred_gaussian = svm_gaussian.predict(X_train_binary)
    train_acc_gaussian = accuracy_score(y_train_binary, y_train_pred_gaussian)
    print(f"  Training accuracy: {train_acc_gaussian:.4f}")

    # Part 2(c): Visualize top 5 support vectors
    print(f"\nPart 2(c) - Visualizing support vectors...")
    top_5_indices_g = np.argsort(svm_gaussian.support_alphas)[-5:][::-1]
    top_5_sv_g = svm_gaussian.support_vectors[top_5_indices_g]

    fig = plot_images(top_5_sv_g,
                      titles=[f"SV {i+1} (α={svm_gaussian.support_alphas[idx]:.4f})" 
                             for i, idx in enumerate(top_5_indices_g)],
                      n_cols=5, figsize=(15, 3))
    plt.suptitle('Top 5 Support Vectors (Gaussian Kernel)', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('output/part2c_top5_support_vectors_gaussian.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Top 5 support vectors saved to 'output/part2c_top5_support_vectors_gaussian.png'")

    # Part 2(d): Compare accuracies
    print(f"\nPart 2(d) - Comparison:")
    print(f"  Linear kernel test accuracy: {test_acc_linear:.4f}")
    print(f"  Gaussian kernel test accuracy: {test_acc_gaussian:.4f}")
    print(f"  Improvement: {test_acc_gaussian - test_acc_linear:+.4f}")
    
    if test_acc_gaussian > test_acc_linear:
        print(f"  Gaussian kernel performs better by {(test_acc_gaussian - test_acc_linear)*100:.2f}%")
    elif test_acc_gaussian < test_acc_linear:
        print(f"  Linear kernel performs better by {(test_acc_linear - test_acc_gaussian)*100:.2f}%")
    else:
        print(f"  Both kernels achieve the same accuracy")

    return svm_gaussian, test_acc_gaussian, train_acc_gaussian, n_sv_gaussian, train_time_gaussian