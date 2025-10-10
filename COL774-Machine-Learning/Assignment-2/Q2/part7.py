# part7.py

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from helpers import plot_images
import numpy as np

def run(X_test_multi, y_test_multi, y_pred_cvxopt_multi, y_pred_sklearn_multi, ALL_CLASS_NAMES):
    print("\n" + "=" * 80)
    print("PART 7: CONFUSION MATRICES")
    print("=" * 80)

    # Confusion matrix for CVXOPT
    cm_cvxopt = confusion_matrix(y_test_multi, y_pred_cvxopt_multi)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_cvxopt, annot=True, fmt='d', cmap='Blues',
                xticklabels=ALL_CLASS_NAMES, yticklabels=ALL_CLASS_NAMES)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('Confusion Matrix - CVXOPT Multi-class SVM', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('output/part7_confusion_matrix_cvxopt.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("CVXOPT confusion matrix saved")

    # Confusion matrix for sklearn
    cm_sklearn = confusion_matrix(y_test_multi, y_pred_sklearn_multi)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues',
                xticklabels=ALL_CLASS_NAMES, yticklabels=ALL_CLASS_NAMES)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('Confusion Matrix - sklearn Multi-class SVM', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('output/part7_confusion_matrix_sklearn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("sklearn confusion matrix saved")

    # Analyze misclassifications
    print("\nAnalyzing confusion matrices...")

    # Find most common misclassifications
    misclass_cvxopt = []
    for i in range(len(ALL_CLASS_NAMES)):
        for j in range(len(ALL_CLASS_NAMES)):
            if i != j and cm_cvxopt[i, j] > 0:
                misclass_cvxopt.append((ALL_CLASS_NAMES[i], ALL_CLASS_NAMES[j], cm_cvxopt[i, j]))

    misclass_cvxopt.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 misclassifications (CVXOPT):")
    for true_cls, pred_cls, count in misclass_cvxopt[:5]:
        print(f"  {true_cls} → {pred_cls}: {count} times")

    # Visualize misclassified examples
    print("\nFinding misclassified examples...")
    misclassified_indices = np.where(y_test_multi != y_pred_cvxopt_multi)[0]

    if len(misclassified_indices) >= 10:
        sample_misclassified = np.random.choice(misclassified_indices, 10, replace=False)
        
        misclass_images = X_test_multi[sample_misclassified]
        misclass_titles = [f"True: {ALL_CLASS_NAMES[y_test_multi[i]]}\nPred: {ALL_CLASS_NAMES[y_pred_cvxopt_multi[i]]}"
                          for i in sample_misclassified]
        
        fig = plot_images(misclass_images, titles=misclass_titles, n_cols=5, figsize=(15, 6))
        plt.suptitle('10 Misclassified Examples', fontsize=14, fontweight='bold', y=1.02)
        plt.savefig('output/part7_misclassified_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Misclassified examples saved to 'output/part7_misclassified_examples.png'")

    print("\nObservations:")
    print("  - Some classes are more difficult to distinguish (e.g., cat vs dog, truck vs automobile)")
    print("  - Visually similar classes tend to be confused more often")
    print("  - The confusion matrices show where the model struggles most")