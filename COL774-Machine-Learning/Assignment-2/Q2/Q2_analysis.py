"""
Assignment 2 - Question 2: Image Classification using SVM
Complete analysis script covering all parts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import time
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from svm import SupportVectorMachine, MultiClassSVM

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_images_from_folder(folder_path, target_size=(32, 32)):
    """
    Load and preprocess images from a folder.
    
    Args:
        folder_path: Path to folder containing images
        target_size: Tuple (height, width) to resize images
        
    Returns:
        np.array: Array of flattened and normalized images
    """
    images = []
    filenames = []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            try:
                # Load image
                img = Image.open(img_path)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # # Resize to target size
                # img = img.resize(target_size)
                
                # # Convert to numpy array and normalize to [0, 1]
                # img_array = np.array(img) / 255.0

                # Use high-quality resampling for resize
                img = img.resize(target_size, resample=Image.LANCZOS)
                img_array = np.array(img).astype(np.float32) / 255.0  # H x W x 3 in [0,1]
                
                # Flatten to 1D vector
                img_flat = img_array.flatten()
                
                images.append(img_flat)
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if len(images) == 0:
        return np.empty((0, target_size[0]*target_size[1]*3)), filenames
    
    return np.array(images), filenames


def load_dataset(data_dir, class_names, split='train'):
    """
    Load dataset for given classes.
    
    Args:
        data_dir: Base directory containing data
        class_names: List of class names (folder names)
        split: 'train' or 'test'
        
    Returns:
        X: np.array of shape (N, D) - flattened images
        y: np.array of shape (N,) - class labels
    """
    X_list = []
    y_list = []
    
    for class_idx, class_name in enumerate(class_names):
        folder_path = os.path.join(data_dir, split, class_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist!")
            continue
        
        print(f"Loading {split} images for class {class_idx} ({class_name})...")
        images, _ = load_images_from_folder(folder_path)
        
        X_list.append(images)
        y_list.append(np.full(len(images), class_idx))
        
        print(f"  Loaded {len(images)} images")
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    return X, y


def reshape_to_image(flat_array, shape=(32, 32, 3)):
    """
    Reshape flattened array back to image.
    
    Args:
        flat_array: 1D array of length D
        shape: Target shape (height, width, channels)
        
    Returns:
        np.array: Reshaped image
    """
    return flat_array.reshape(shape)


# def plot_images(images, titles=None, n_cols=5, figsize=(15, 3)):
#     """
#     Plot multiple images in a grid.
    
#     Args:
#         images: List of images (can be flattened or shaped)
#         titles: List of titles for each image
#         n_cols: Number of columns in grid
#         figsize: Figure size
#     """
#     n_images = len(images)
#     n_rows = (n_images + n_cols - 1) // n_cols
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
#     axes = axes.flatten() if n_images > 1 else [axes]
    
#     for i in range(n_images):
#         img = images[i]
        
#         # Reshape if flattened
#         if len(img.shape) == 1:
#             img = reshape_to_image(img)

#         # Normalize to [0, 1] for display
#         img_min = img.min()
#         img_max = img.max()
#         if img_max > img_min:
#             img = (img - img_min) / (img_max - img_min)
#         else:
#             img = np.zeros_like(img)
        
#         axes[i].imshow(img)
#         axes[i].axis('off')
        
#         if titles and i < len(titles):
#             axes[i].set_title(titles[i], fontsize=10)
    
#     # Hide remaining subplots
#     for i in range(n_images, len(axes)):
#         axes[i].axis('off')
    
#     plt.tight_layout()
#     return fig

def plot_images(images, titles=None, n_cols=5, figsize=(15, 3), normalize='global'):
    """
    Plot multiple images in a grid with pixel-accurate rendering.
    images: list/array of flattened images or shaped (H,W,3)
    normalize: 'global' -> scale by global min/max across all images (recommended)
               'per_image' -> scale each image individually (your earlier behavior)
               None -> assume values are already in [0,1] or uint8
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]

    # Convert list to numpy array for global normalization (if needed)
    imgs = []
    for img in images:
        arr = img
        if arr.ndim == 1:
            arr = arr.reshape((32, 32, 3))
        imgs.append(arr)
    # imgs = np.array(imgs)
    imgs = np.stack(imgs, axis=0)


    # Global normalization across all shown images (better for comparison)
    if normalize == 'global':
        global_min = imgs.min()
        global_max = imgs.max()
        if global_max > global_min:
            imgs = (imgs - global_min) / (global_max - global_min)
    elif normalize == 'per_image':
        for i in range(len(imgs)):
            mn = imgs[i].min()
            mx = imgs[i].max()
            if mx > mn:
                imgs[i] = (imgs[i] - mn) / (mx - mn)
    # else assume in [0,1] already

    # Convert to uint8 for crisper saving/display (optional)
    imgs_to_show = (imgs * 255).clip(0, 255).astype(np.uint8)

    for i in range(n_images):
        img = imgs_to_show[i]
        axes[i].imshow(img, interpolation='nearest')   # IMPORTANT: nearest keeps pixels sharp
        axes[i].axis('off')
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=10)

    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


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
# PART 1: LINEAR KERNEL WITH CVXOPT
# ============================================================================

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
print("Top 5 support vectors saved")

# Visualize weight vector
fig = plot_images([svm_linear.w], titles=['Weight Vector (w)'], n_cols=1, figsize=(4, 4))
plt.savefig('output/part1c_weight_vector.png', dpi=150, bbox_inches='tight')
plt.close()
print("Weight vector visualization saved")

# ============================================================================
# PART 2: GAUSSIAN KERNEL WITH CVXOPT
# ============================================================================

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
print(f"    Linear: {n_sv_linear} SVs")
print(f"    Gaussian: {n_sv_gaussian} SVs")
print(f"    Difference: {n_sv_gaussian - n_sv_linear} ({n_sv_gaussian - n_sv_linear > 0 and 'more' or 'fewer'} SVs)")

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
print("Top 5 support vectors saved")

# Part 2(d): Compare accuracies
print(f"\nPart 2(d) - Comparison:")
print(f"  Linear kernel test accuracy: {test_acc_linear:.4f}")
print(f"  Gaussian kernel test accuracy: {test_acc_gaussian:.4f}")
print(f"  Improvement: {test_acc_gaussian - test_acc_linear:+.4f}")

# ============================================================================
# PART 3: SCIKIT-LEARN SVM (LIBSVM)
# ============================================================================

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

# Check matching SVs for linear
sklearn_linear_sv_indices = set(sklearn_linear.support_)
cvxopt_linear_sv_indices = set(sv_indices_linear)
matching_linear = len(sklearn_linear_sv_indices & cvxopt_linear_sv_indices)
print(f"    Matching SVs: {matching_linear}")

print(f"\n  Gaussian kernel:")
print(f"    CVXOPT: {n_sv_gaussian} SVs")
print(f"    sklearn: {n_sv_sklearn_gaussian} SVs")

# Check matching SVs for Gaussian
sklearn_gaussian_sv_indices = set(sklearn_gaussian.support_)
cvxopt_gaussian_sv_indices = set(sv_indices_gaussian)
matching_gaussian = len(sklearn_gaussian_sv_indices & cvxopt_gaussian_sv_indices)
print(f"    Matching SVs: {matching_gaussian}")

# Part 3(b): Compare weights and bias for linear kernel
print(f"\nPart 3(b) - Linear Kernel Parameters:")
print(f"  Weight vector (w):")
print(f"    CVXOPT norm: {np.linalg.norm(svm_linear.w):.6f}")
print(f"    sklearn norm: {np.linalg.norm(sklearn_linear.coef_[0]):.6f}")
print(f"    Cosine similarity: {np.dot(svm_linear.w, sklearn_linear.coef_[0]) / (np.linalg.norm(svm_linear.w) * np.linalg.norm(sklearn_linear.coef_[0])):.6f}")

print(f"  Bias (b):")
print(f"    CVXOPT: {svm_linear.b:.6f}")
print(f"    sklearn: {sklearn_linear.intercept_[0]:.6f}")
print(f"    Difference: {abs(svm_linear.b - sklearn_linear.intercept_[0]):.6f}")

# Part 3(c): Test accuracy
y_pred_sklearn_linear = sklearn_linear.predict(X_test_binary)
test_acc_sklearn_linear = accuracy_score(y_test_binary, y_pred_sklearn_linear)
# Training accuracy
train_acc_sklearn_linear = accuracy_score(y_train_binary, sklearn_linear.predict(X_train_binary))


y_pred_sklearn_gaussian = sklearn_gaussian.predict(X_test_binary)
test_acc_sklearn_gaussian = accuracy_score(y_test_binary, y_pred_sklearn_gaussian)
# Training accuracy
train_acc_sklearn_gaussian = accuracy_score(y_train_binary, sklearn_gaussian.predict(X_train_binary))


print(f"\nPart 3(c) - Test Accuracy:")
print(f"  Linear kernel:")
print(f"    CVXOPT: {test_acc_linear:.4f}")
print(f"    sklearn test accuracy: {test_acc_sklearn_linear:.4f}")
print(f"    sklearn training accuracy: {train_acc_sklearn_linear:.4f}")
print(f"\n  Gaussian kernel:")
print(f"    CVXOPT: {test_acc_gaussian:.4f}")
print(f"    sklearn test accuracy: {test_acc_sklearn_gaussian:.4f}")
print(f"    sklearn training accuracy: {train_acc_sklearn_gaussian:.4f}")


# Part 3(d): Training time comparison
print(f"\nPart 3(d) - Computational Cost (Training Time):")
print(f"  Linear kernel:")
print(f"    CVXOPT: {train_time_linear:.2f} seconds")
print(f"    sklearn: {sklearn_linear_time:.2f} seconds")
print(f"    Speedup: {train_time_linear/sklearn_linear_time:.2f}x")

print(f"\n  Gaussian kernel:")
print(f"    CVXOPT: {train_time_gaussian:.2f} seconds")
print(f"    sklearn: {sklearn_gaussian_time:.2f} seconds")
print(f"    Speedup: {train_time_gaussian/sklearn_gaussian_time:.2f}x")

# Summary table
summary_data = {
    'Method': ['CVXOPT Linear', 'sklearn Linear', 'CVXOPT Gaussian', 'sklearn Gaussian'],
    'Train Time (s)': [train_time_linear, sklearn_linear_time, train_time_gaussian, sklearn_gaussian_time],
    'Num SVs': [n_sv_linear, n_sv_sklearn_linear, n_sv_gaussian, n_sv_sklearn_gaussian],
    'Test Acc': [test_acc_linear, test_acc_sklearn_linear, test_acc_gaussian, test_acc_sklearn_gaussian]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + "=" * 80)
print("BINARY CLASSIFICATION SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))

# ============================================================================
# MULTI-CLASS CLASSIFICATION
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
# PART 5: ONE-VS-ONE MULTI-CLASS SVM WITH CVXOPT
# ============================================================================

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

print(f"  Test accuracy: {test_acc_cvxopt_multi:.4f}")
print(f"  Total support vectors across all classifiers: {multiclass_cvxopt.get_total_support_vectors()}")

# Training accuracy
y_train_pred_cvxopt_multi = multiclass_cvxopt.predict(X_train_multi)
train_acc_cvxopt_multi = accuracy_score(y_train_multi, y_train_pred_cvxopt_multi)
print(f"  Training accuracy: {train_acc_cvxopt_multi:.4f}")


# ============================================================================
# PART 6: MULTI-CLASS SVM WITH SKLEARN
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: MULTI-CLASS SVM (SKLEARN)")
print("=" * 80)

print("\nTraining multi-class SVM with sklearn...")
start_time = time.time()

sklearn_multi = SVC(kernel='rbf', C=1.0, gamma=0.001, decision_function_shape='ovo')
sklearn_multi.fit(X_train_multi, y_train_multi)

sklearn_multi_time = time.time() - start_time
print(f"Training completed in {sklearn_multi_time:.2f} seconds")

# Part 6(a): Test accuracy
y_pred_sklearn_multi = sklearn_multi.predict(X_test_multi)
test_acc_sklearn_multi = accuracy_score(y_test_multi, y_pred_sklearn_multi)

print(f"\nPart 6(a) - Test Accuracy:")
print(f"  Test accuracy: {test_acc_sklearn_multi:.4f}")
# Training accuracy
train_acc_sklearn_multi = accuracy_score(y_train_multi, sklearn_multi.predict(X_train_multi))
print(f"  Training accuracy: {train_acc_sklearn_multi:.4f}")


# Part 6(b): Comparison
print(f"\nPart 6(b) - Comparison:")
print(f"  Test Accuracy:")
print(f"    CVXOPT: {test_acc_cvxopt_multi:.4f}")
print(f"    sklearn: {test_acc_sklearn_multi:.4f}")
print(f"    Difference: {abs(test_acc_cvxopt_multi - test_acc_sklearn_multi):.4f}")

print(f"\n  Training Time:")
print(f"    CVXOPT: {multiclass_cvxopt_time:.2f} seconds")
print(f"    sklearn: {sklearn_multi_time:.2f} seconds")
print(f"    Speedup: {multiclass_cvxopt_time/sklearn_multi_time:.2f}x")

# ============================================================================
# PART 7: CONFUSION MATRICES
# ============================================================================

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

# ============================================================================
# PART 8: CROSS-VALIDATION FOR HYPERPARAMETER TUNING
# ============================================================================

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


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nBinary Classification Results:")
print(f"  Best method: sklearn Gaussian (Accuracy: {test_acc_sklearn_gaussian:.4f})")
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