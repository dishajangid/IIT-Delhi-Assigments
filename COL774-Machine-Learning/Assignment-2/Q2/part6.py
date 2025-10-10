# part6.py

import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run(X_train_multi, y_train_multi, X_test_multi, y_test_multi, test_acc_cvxopt_multi, train_acc_cvxopt_multi, multiclass_cvxopt_time):
    print("\n" + "=" * 80)
    print("PART 6: MULTI-CLASS SVM (SKLEARN)")
    print("=" * 80)

    print("\nTraining multi-class SVM with sklearn...")
    print("Using one-vs-one strategy (decision_function_shape='ovo')...")
    start_time = time.time()

    sklearn_multi = SVC(kernel='rbf', C=1.0, gamma=0.001, decision_function_shape='ovo')
    sklearn_multi.fit(X_train_multi, y_train_multi)

    sklearn_multi_time = time.time() - start_time
    print(f"Training completed in {sklearn_multi_time:.2f} seconds")

    # Part 6(a): Test accuracy
    print("\nPart 6(a) - Test Accuracy:")
    y_pred_sklearn_multi = sklearn_multi.predict(X_test_multi)
    test_acc_sklearn_multi = accuracy_score(y_test_multi, y_pred_sklearn_multi)
    print(f"  Test accuracy: {test_acc_sklearn_multi:.4f} ({test_acc_sklearn_multi*100:.2f}%)")
    
    # Training accuracy
    train_acc_sklearn_multi = accuracy_score(y_train_multi, sklearn_multi.predict(X_train_multi))
    print(f"  Training accuracy: {train_acc_sklearn_multi:.4f} ({train_acc_sklearn_multi*100:.2f}%)")

    # Additional sklearn-specific metrics
    print(f"  Total number of support vectors: {len(sklearn_multi.support_)}")
    print(f"  Support vectors per class: {sklearn_multi.n_support_}")

    # Part 6(b): Comparison
    print(f"\nPart 6(b) - Comparison with CVXOPT:")
    
    print(f"  Test Accuracy:")
    print(f"    CVXOPT: {test_acc_cvxopt_multi:.4f} ({test_acc_cvxopt_multi*100:.2f}%)")
    print(f"    sklearn: {test_acc_sklearn_multi:.4f} ({test_acc_sklearn_multi*100:.2f}%)")
    acc_diff = test_acc_sklearn_multi - test_acc_cvxopt_multi
    print(f"    Difference: {abs(acc_diff):.4f} ({abs(acc_diff)*100:.2f}%)")
    
    if acc_diff > 0:
        print(f"    sklearn performs better by {acc_diff*100:.2f}%")
    elif acc_diff < 0:
        print(f"    CVXOPT performs better by {abs(acc_diff)*100:.2f}%")
    else:
        print(f"    Both methods achieve the same accuracy")

    print(f"\n  Training Accuracy:")
    print(f"    CVXOPT: {train_acc_cvxopt_multi:.4f} ({train_acc_cvxopt_multi*100:.2f}%)")
    print(f"    sklearn: {train_acc_sklearn_multi:.4f} ({train_acc_sklearn_multi*100:.2f}%)")

    print(f"\n  Training Time:")
    print(f"    CVXOPT: {multiclass_cvxopt_time:.2f} seconds")
    print(f"    sklearn: {sklearn_multi_time:.2f} seconds")
    speedup = multiclass_cvxopt_time / sklearn_multi_time
    print(f"    Speedup: {speedup:.2f}x")
    print(f"    sklearn is {speedup:.1f} times faster than CVXOPT")
    
    print(f"\n  Key Observations:")
    print(f"    - LIBSVM (sklearn) is highly optimized for speed")
    print(f"    - Both implementations use one-vs-one strategy")
    print(f"    - Accuracy differences are typically small (both well-implemented)")
    print(f"    - LIBSVM's optimization makes it practical for large datasets")
    
    return test_acc_sklearn_multi, train_acc_sklearn_multi, sklearn_multi_time, y_pred_sklearn_multi