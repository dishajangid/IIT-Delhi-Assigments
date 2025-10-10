# part8.py

import seaborn as sns
import matplotlib.pyplot as plt
from helpers import *

def run(train_df, test_df):
    # ============================================================================
    # PART 8: Confusion Matrix
    # ============================================================================
    print("=" * 80)
    print("PART 8: CONFUSION MATRIX")
    print("=" * 80)

    # Use best model (separate parameters)
    y_true = test_df['label']
    y_pred = test_df['Predicted Combined Separate']

    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 15), yticklabels=range(1, 15))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix - Best Model (Separate Parameters)')
    plt.tight_layout()
    plt.savefig('output/part8_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to 'output/part8_confusion_matrix.png'")

    # Find class with highest diagonal entry
    diagonal_values = np.diag(cm)
    print(f"Diagonal values: {diagonal_values}")
    best_class = np.argmax(diagonal_values) + 1
    print(f"\nClass with highest diagonal entry: Class {best_class}")
    print(f"Value: {diagonal_values[best_class-1]} (out of {np.sum(cm[best_class-1, :])} samples)")
    print(f"This means Class {best_class} has the highest number of correct predictions.")
    print()