"""
Assignment 2 - Question 1: Text Classification using Naive Bayes
"""

import pandas as pd
import numpy as np
import os
from naive_bayes import NaiveBayes
from helpers import *
import part1
import part2
import part3
import part4
import part5
import part6
import part7
import part8
import part9

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Load data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Number of classes: {len(train_df['label'].unique())}")
print(f"Classes: {sorted(train_df['label'].unique())}")
print(f"Columns: {train_df.columns.tolist()}")
print()

# Check class distribution
print("Class Distribution in Training Set:")
print(train_df['label'].value_counts().sort_index())
print()

# Run each part
print("\n" + "=" * 80)
print("STARTING ANALYSIS")
print("=" * 80 + "\n")

part1.run(train_df, test_df)
part2.run(train_df, test_df)
part3.run(train_df, test_df)
part4.run(train_df, test_df)
part5.run(train_df, test_df)
part6.run(train_df, test_df)
part7.run(train_df, test_df)
part8.run(train_df, test_df)
part9.run(train_df, test_df)

# Summary
print("\n" + "=" * 80)
print("FINAL SUMMARY OF ALL MODELS")
print("=" * 80 + "\n")

all_models = {}

# Collect all test accuracies
if hasattr(part1, 'test_accuracy') and part1.test_accuracy is not None:
    all_models['Part 1: Raw Content (Simple Tokenizer)'] = part1.test_accuracy

if hasattr(part2, 'test_accuracy_processed') and part2.test_accuracy_processed is not None:
    all_models['Part 2: Preprocessed Content (Stemming + Stopwords)'] = part2.test_accuracy_processed

if hasattr(part3, 'test_accuracy_unigrams') and part3.test_accuracy_unigrams is not None:
    all_models['Part 3: Content Unigrams'] = part3.test_accuracy_unigrams

if hasattr(part3, 'test_accuracy_bigrams') and part3.test_accuracy_bigrams is not None:
    all_models['Part 3: Content Bigrams Only'] = part3.test_accuracy_bigrams

# Check for combined unigrams+bigrams from test_df
if 'Predicted Combined' in test_df.columns:
    acc = np.mean(test_df['Predicted Combined'] == test_df['label'])
    all_models['Part 3: Content (Unigrams + Bigrams)'] = acc

if hasattr(part5, 'test_accuracy_title_processed') and part5.test_accuracy_title_processed is not None:
    all_models['Part 5: Title Unigrams'] = part5.test_accuracy_title_processed

if hasattr(part5, 'test_accuracy_title_bigrams') and part5.test_accuracy_title_bigrams is not None:
    all_models['Part 5: Title (Unigrams + Bigrams)'] = part5.test_accuracy_title_bigrams

if hasattr(part6, 'test_accuracy_concat') and part6.test_accuracy_concat is not None:
    all_models['Part 6a: Combined (Same Parameters)'] = part6.test_accuracy_concat

if hasattr(part6, 'test_accuracy_separate') and part6.test_accuracy_separate is not None:
    all_models['Part 6b: Combined (Separate Parameters)'] = part6.test_accuracy_separate

if hasattr(part9, 'test_accuracy_enhanced') and part9.test_accuracy_enhanced is not None:
    all_models['Part 9: Enhanced (Length Features)'] = part9.test_accuracy_enhanced

# Print summary table
if all_models:
    print("Test Accuracies Summary:")
    print("-" * 80)
    print(f"{'Model':<50} {'Test Accuracy':<15}")
    print("-" * 80)
    
    # Sort by accuracy (descending)
    for model_name, accuracy in sorted(all_models.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<50} {accuracy:<15.4f}")
    
    print("-" * 80)
    
    # Identify best model
    best_model = max(all_models.items(), key=lambda x: x[1])
    print(f"\n✓ Best Overall Model: {best_model[0]}")
    print(f"  Test Accuracy: {best_model[1]:.4f}")
    print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nAll outputs have been saved to the 'output/' directory:")
print("  - Word clouds: output/part_wordclouds_*.png")
print("  - Confusion matrix: output/part8_confusion_matrix.png")
print("\nFor detailed analysis, refer to the individual part outputs above.")
print()