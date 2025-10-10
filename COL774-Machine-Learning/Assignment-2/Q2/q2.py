# q2.py (Main file)

"""
Assignment 2 - Question 2: Image Classification using SVM
"""

import numpy as np
import pandas as pd
import time
import os
from helpers import *
from part1 import run as run_part1
from part2 import run as run_part2
from part3 import run as run_part3
from part5 import run as run_part5
from part6 import run as run_part6
from part7 import run as run_part7
from part8 import run as run_part8
from summary import run as run_summary

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# ============================================================================
# DETERMINE CLASSES BASED ON ENTRY NUMBER
# ============================================================================

print("=" * 80)
print("PART 2: IMAGE CLASSIFICATION USING SVM")
print("=" * 80)

# Ask user for entry number
entry_number_input = input("\nEnter the last 2 digits of your entry number: ")
d = int(entry_number_input)

class1_idx = d % 10
class2_idx = (d + 1) % 10

# Class names (alphabetically ordered)
ALL_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

selected_classes = [ALL_CLASS_NAMES[class1_idx], ALL_CLASS_NAMES[class2_idx]]

print(f"\nEntry number last 2 digits: {d}")
print(f"Class 1 (index {class1_idx}): {selected_classes[0]}")
print(f"Class 2 (index {class2_idx}): {selected_classes[1]}")

# ============================================================================
# LOAD BINARY CLASSIFICATION DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING BINARY CLASSIFICATION DATA")
print("=" * 80)

# Data directory structure: data/train and data/test
DATA_DIR = 'data'

# Load training and test data
X_train_binary, y_train_binary = load_dataset(DATA_DIR, selected_classes, 'train')
X_test_binary, y_test_binary = load_dataset(DATA_DIR, selected_classes, 'test')

print(f"\nBinary classification dataset:")
print(f"  Training samples: {len(X_train_binary)}")
print(f"  Test samples: {len(X_test_binary)}")
print(f"  Feature dimension: {X_train_binary.shape[1]}")
print(f"  Class distribution (train): {np.bincount(y_train_binary)}")
print(f"  Class distribution (test): {np.bincount(y_test_binary)}")

# Visualize sample images
print("\nVisualizing sample images...")
fig = plot_images(X_train_binary[:10], 
                  titles=[f"Class {y}" for y in y_train_binary[:10]],
                  n_cols=5, figsize=(15, 6))
plt.savefig('output/sample_images.png', dpi=200, bbox_inches='tight')
plt.close()
print("Sample images saved to 'output/sample_images.png'")

# ============================================================================
# RUN BINARY CLASSIFICATION PARTS
# ============================================================================

# Part 1: Linear Kernel with CVXOPT
svm_linear, test_acc_linear, train_acc_linear, n_sv_linear, train_time_linear = run_part1(
    X_train_binary, y_train_binary, X_test_binary, y_test_binary
)

# Part 2: Gaussian Kernel with CVXOPT
svm_gaussian, test_acc_gaussian, train_acc_gaussian, n_sv_gaussian, train_time_gaussian = run_part2(
    X_train_binary, y_train_binary, X_test_binary, y_test_binary,
    svm_linear, test_acc_linear, n_sv_linear
)

# Part 3: Scikit-learn SVM comparison
run_part3(
    X_train_binary, y_train_binary, X_test_binary, y_test_binary,
    svm_linear, svm_gaussian, n_sv_linear, n_sv_gaussian,
    test_acc_linear, test_acc_gaussian, train_time_linear, train_time_gaussian
)

# ============================================================================
# LOAD MULTI-CLASS DATA
# ============================================================================

print("\n\n" + "=" * 80)
print("MULTI-CLASS CLASSIFICATION")
print("=" * 80)

# Load full dataset (all 10 classes)
print("\nLoading multi-class dataset...")
X_train_multi, y_train_multi = load_dataset(DATA_DIR, ALL_CLASS_NAMES, 'train')
X_test_multi, y_test_multi = load_dataset(DATA_DIR, ALL_CLASS_NAMES, 'test')

print(f"\nMulti-class dataset:")
print(f"  Training samples: {len(X_train_multi)}")
print(f"  Test samples: {len(X_test_multi)}")
print(f"  Number of classes: {len(ALL_CLASS_NAMES)}")
print(f"  Class distribution (train): {np.bincount(y_train_multi)}")

# ============================================================================
# RUN MULTI-CLASS CLASSIFICATION PARTS
# ============================================================================

# Part 5: One-vs-One Multi-class SVM (CVXOPT)
multiclass_cvxopt, test_acc_cvxopt_multi, train_acc_cvxopt_multi, multiclass_cvxopt_time = run_part5(
    X_train_multi, y_train_multi, X_test_multi, y_test_multi
)

# Part 6: Multi-class SVM (sklearn)
test_acc_sklearn_multi, train_acc_sklearn_multi, sklearn_multi_time, y_pred_sklearn_multi = run_part6(
    X_train_multi, y_train_multi, X_test_multi, y_test_multi,
    test_acc_cvxopt_multi, train_acc_cvxopt_multi, multiclass_cvxopt_time
)

# Part 7: Confusion Matrices
y_pred_cvxopt_multi = multiclass_cvxopt.predict(X_test_multi)
run_part7(
    X_test_multi, y_test_multi, y_pred_cvxopt_multi, y_pred_sklearn_multi, ALL_CLASS_NAMES
)

# Part 8: Cross-validation for hyperparameter tuning
test_acc_best, train_acc_best, best_c, best_cv_acc = run_part8(
    X_train_multi, y_train_multi, X_test_multi, y_test_multi, test_acc_sklearn_multi
)

# Summary
run_summary(
    X_train_multi, y_train_multi, X_test_multi, y_test_multi,
    test_acc_cvxopt_multi, train_acc_cvxopt_multi, multiclass_cvxopt_time,
    test_acc_sklearn_multi, train_acc_sklearn_multi, sklearn_multi_time,
    test_acc_best, train_acc_best, best_c, best_cv_acc,
    multiclass_cvxopt, test_acc_gaussian, train_time_linear
)

print("\n" + "=" * 80)
print("ALL PARTS COMPLETED SUCCESSFULLY")
print("=" * 80)