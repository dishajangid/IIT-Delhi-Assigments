# part5.py

import time
from svm import MultiClassSVM
from sklearn.metrics import accuracy_score

def run(X_train_multi, y_train_multi, X_test_multi, y_test_multi):
    print("\n" + "=" * 80)
    print("PART 5: ONE-VS-ONE MULTI-CLASS SVM (CVXOPT)")
    print("=" * 80)

    print("\nTraining one-vs-one multi-class SVM with CVXOPT...")
    print("This will train 45 binary classifiers (10 choose 2)...")
    start_time = time.time()

    multiclass_cvxopt = MultiClassSVM()
    multiclass_cvxopt.fit(X_train_multi, y_train_multi, kernel='gaussian', C=1.0, gamma=0.001)

    multiclass_cvxopt_time = time.time() - start_time
    print(f"Training completed in {multiclass_cvxopt_time:.2f} seconds")

    # Part 5(a): Test accuracy
    print("\nPart 5(a) - Classification:")
    y_pred_cvxopt_multi = multiclass_cvxopt.predict(X_test_multi)
    test_acc_cvxopt_multi = accuracy_score(y_test_multi, y_pred_cvxopt_multi)

    print(f"  Test accuracy: {test_acc_cvxopt_multi:.4f} ({test_acc_cvxopt_multi*100:.2f}%)")
    
    total_svs = multiclass_cvxopt.get_total_support_vectors()
    print(f"  Total support vectors across all classifiers: {total_svs}")
    print(f"  Average SVs per binary classifier: {total_svs/45:.1f}")

    # Training accuracy
    y_train_pred_cvxopt_multi = multiclass_cvxopt.predict(X_train_multi)
    train_acc_cvxopt_multi = accuracy_score(y_train_multi, y_train_pred_cvxopt_multi)
    print(f"  Training accuracy: {train_acc_cvxopt_multi:.4f} ({train_acc_cvxopt_multi*100:.2f}%)")
    
    # Additional analysis
    print(f"\n  Performance Analysis:")
    print(f"    Number of binary classifiers: 45")
    print(f"    Average training time per classifier: {multiclass_cvxopt_time/45:.2f} seconds")
    
    return multiclass_cvxopt, test_acc_cvxopt_multi, train_acc_cvxopt_multi, multiclass_cvxopt_time