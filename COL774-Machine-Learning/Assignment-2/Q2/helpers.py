# helpers.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

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

def plot_images(images, titles=None, n_cols=5, figsize=(15, 3), normalize='global'):
    """
    Plot multiple images in a grid with pixel-accurate rendering.
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