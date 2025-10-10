# part8.py

import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def run(X_train_multi, y_train_multi, X_test_multi, y_test_multi, test_acc_sklearn_multi):
    print("\n" + "=" * 80)
    print("PART 8: CROSS-VALIDATION FOR HYPERPARAMETER TUNING")
    print("=" * 80)

    # Part 8(a): 5-fold cross-validation for different C values
    C_values = [1e-5, 1e-3, 1, 5, 10]
    gamma_fixed = 0.001

    cv_accuracies = []
    test_accuracies = []

    print("\nPerforming 5-fold cross-validation...")
    print(f"{'C':>10} {'CV Accuracy':>15} {'Test Accuracy':>15} {'Time (s)':>12}")
    print("-" * 55)

    for C in C_values:
        # Cross-validation
        start_time = time.time()
        svm_cv = SVC(kernel='rbf', C=C, gamma=gamma_fixed)
        cv_scores = cross_val_score(svm_cv, X_train_multi, y_train_multi, cv=5, n_jobs=-1)
        cv_acc = np.mean(cv_scores)
        cv_time = time.time() - start_time
        
        # Test accuracy
        svm_cv.fit(X_train_multi, y_train_multi)
        y_pred_cv = svm_cv.predict(X_test_multi)
        test_acc = accuracy_score(y_test_multi, y_pred_cv)
        
        cv_accuracies.append(cv_acc)
        test_accuracies.append(test_acc)
        
        print(f"{C:>10.5f} {cv_acc:>15.4f} {test_acc:>15.4f} {cv_time:>12.2f}")

    # Part 8(b): Plot CV and test accuracies
    print("\nPart 8(b) - Plotting accuracies vs C...")

    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, cv_accuracies, 'o-', label='5-Fold CV Accuracy', linewidth=2, markersize=8)
    plt.semilogx(C_values, test_accuracies, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    plt.xlabel('C (Regularization Parameter)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Performance vs C Value', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/part8_cv_accuracy_vs_c.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to 'output/part8_cv_accuracy_vs_c.png'")

    # Find best C
    best_c_idx = np.argmax(cv_accuracies)
    best_c = C_values[best_c_idx]
    best_cv_acc = cv_accuracies[best_c_idx]

    print(f"\nBest C value: {best_c}")
    print(f"Best CV accuracy: {best_cv_acc:.4f}")

    print("\nObservations:")
    print(f"  - Best C value based on cross-validation: {best_c}")
    print(f"  - CV accuracy tends to {'increase' if cv_accuracies[-1] > cv_accuracies[0] else 'vary'} with C")
    print(f"  - Test accuracy follows a similar trend to CV accuracy")

    # Part 8(c): Train with best C
    print(f"\nPart 8(c) - Training with best C={best_c}...")
    svm_best = SVC(kernel='rbf', C=best_c, gamma=gamma_fixed)
    svm_best.fit(X_train_multi, y_train_multi)

    y_pred_best = svm_best.predict(X_test_multi)
    test_acc_best = accuracy_score(y_test_multi, y_pred_best)

    print(f"  Test accuracy with best C: {test_acc_best:.4f}")
    print(f"  Previous test accuracy (C=1.0): {test_acc_sklearn_multi:.4f}")
    print(f"  Improvement: {test_acc_best - test_acc_sklearn_multi:+.4f}")

    if test_acc_best > test_acc_sklearn_multi:
        print("  The optimized C value improved the test accuracy!")
    else:
        print("  The default C=1.0 was already near-optimal.")
        
    train_acc_best = accuracy_score(y_train_multi, svm_best.predict(X_train_multi))
    print(f"  Training accuracy with best C: {train_acc_best:.4f}")

    return test_acc_best, train_acc_best, best_c, best_cv_acc