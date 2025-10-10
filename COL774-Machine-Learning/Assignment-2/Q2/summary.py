# summary.py

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import time

def run(X_train_multi, y_train_multi, X_test_multi, y_test_multi, test_acc_cvxopt_multi, train_acc_cvxopt_multi, multiclass_cvxopt_time, test_acc_sklearn_multi, train_acc_sklearn_multi, sklearn_multi_time, test_acc_best, train_acc_best, best_c, best_cv_acc, multiclass_cvxopt, test_acc_gaussian, sklearn_linear_time):
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nBinary Classification Results:")
    print(f"  Best method: sklearn Gaussian (Accuracy: {test_acc_gaussian:.4f})")
    print(f"  Fastest method: sklearn Linear (Time: {sklearn_linear_time:.2f}s)")

    print("\nMulti-class Classification Results:")
    print(f"  Best accuracy: {max(test_acc_cvxopt_multi, test_acc_sklearn_multi, test_acc_best):.4f}")
    print(f"  Optimal C value: {best_c}")
    print(f"  Training time comparison: sklearn is {multiclass_cvxopt_time/sklearn_multi_time:.1f}x faster")

    print("\nKey Findings:")
    print("  1. Gaussian kernel generally outperforms linear kernel")
    print("  2. sklearn (LIBSVM) is significantly faster than CVXOPT")
    print("  3. Cross-validation helps identify optimal hyperparameters")
    print("  4. One-vs-One strategy works well for multi-class problems")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print("\n### MULTI-CLASS CLASSIFICATION TABLE ###")
    print(f"CVXOPT Gaussian:")
    print(f"  Training Time: {multiclass_cvxopt_time:.2f} seconds")
    print(f"  Test Accuracy: {test_acc_cvxopt_multi*100:.2f}%")
    print(f"  Training Accuracy: {train_acc_cvxopt_multi*100:.2f}%")
    print(f"  No. of Support Vectors: {multiclass_cvxopt.get_total_support_vectors()}")

    print(f"\nsklearn Gaussian:")
    print(f"  Training Time: {sklearn_multi_time:.2f} seconds")
    print(f"  Test Accuracy: {test_acc_sklearn_multi*100:.2f}%")
    print(f"  Training Accuracy: {train_acc_sklearn_multi*100:.2f}%")

    print("\n### CROSS-VALIDATION RESULTS ###")
    print(f"Best C value: {best_c}")
    print(f"Best CV Accuracy: {best_cv_acc*100:.2f}%")
    print(f"Test Accuracy with best C: {test_acc_best*100:.2f}%")
    print(f"Training Accuracy with best C: {train_acc_best*100:.2f}%")

    print("\n### READY-TO-COPY LATEX TABLE ###")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Training Time (s)} & \\textbf{Test Accuracy (\\%)} & \\textbf{No. of Support Vectors} \\\\")
    print("\\midrule")
    print(f"\\textbf{{CVXOPT Gaussian}} & {multiclass_cvxopt_time:.2f} & {test_acc_cvxopt_multi*100:.2f} & {multiclass_cvxopt.get_total_support_vectors()} \\\\")
    print(f"\\textbf{{sklearn Gaussian}} & {sklearn_multi_time:.2f} & {test_acc_sklearn_multi*100:.2f} & N/A \\\\")
    print(f"\\textbf{{sklearn Best C={best_c}}} & N/A & {test_acc_best*100:.2f} & N/A \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # ============================================================================
    # SGD PARAMETER COMPARISON
    # ============================================================================

    print("\n" + "=" * 80)
    print("SGD PARAMETER COMPARISON")
    print("=" * 80)

    # Alpha comparison
    alpha_values = [0.0001, 0.001, 0.01, 0.1]
    print("\nComparing alpha values (max_iter=1000):")
    print("\\textbf{Comparing alpha values:}")
    print("\\begin{itemize}")
    for alpha in alpha_values:
        start_time = time.time()
        sgd_clf = SGDClassifier(loss='hinge', alpha=alpha, max_iter=1000, random_state=42, tol=1e-3)
        sgd_clf.fit(X_train_multi, y_train_multi)
        train_time = time.time() - start_time
        
        y_pred = sgd_clf.predict(X_test_multi)
        acc = accuracy_score(y_test_multi, y_pred)
        
        emphasis = "\\textbf{" if alpha == 0.001 else ""
        emphasis_end = "}" if alpha == 0.001 else ""
        print(f"  {emphasis}Alpha={alpha}: Accuracy = {acc*100:.2f}\\%, Training time = {train_time:.2f}s{emphasis_end}")
        print(f"  \\item {emphasis}Alpha={alpha}: Accuracy = {acc*100:.2f}\\%, Training time = {train_time:.2f}s{emphasis_end}")

    print("\\end{itemize}")

    # Max iterations comparison
    max_iter_values = [100, 500, 1000, 5000]
    print("\n\\textbf{Comparing max iterations:}")
    print("\\begin{itemize}")
    for max_iter in max_iter_values:
        start_time = time.time()
        sgd_clf = SGDClassifier(loss='hinge', alpha=0.001, max_iter=max_iter, random_state=42, tol=1e-3)
        sgd_clf.fit(X_train_multi, y_train_multi)
        train_time = time.time() - start_time
        
        y_pred = sgd_clf.predict(X_test_multi)
        acc = accuracy_score(y_test_multi, y_pred)
        
        emphasis = "\\textbf{" if max_iter == 1000 else ""
        emphasis_end = "}" if max_iter == 1000 else ""
        print(f"  {emphasis}Max iterations={max_iter}: Accuracy = {acc*100:.2f}\\%, Training time = {train_time:.2f}s{emphasis_end}")
        print(f"  \\item {emphasis}Max iterations={max_iter}: Accuracy = {acc*100:.2f}\\%, Training time = {train_time:.2f}s{emphasis_end}")

    print("\\end{itemize}")

    print("\n" + "=" * 80)
    print("END OF REPORT VALUES")
    print("=" * 80)