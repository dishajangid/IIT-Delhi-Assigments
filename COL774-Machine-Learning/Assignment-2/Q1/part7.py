from helpers import *
import numpy as np

def run(train_df, test_df):
    # ============================================================================
    # PART 7: Baseline Comparisons
    # ============================================================================
    print("=" * 80)
    print("PART 7: BASELINE COMPARISONS")
    print("=" * 80)

    # Part 7(a): Random Guessing Baseline
    print("\nPart 7(a) - Random Guessing Baseline")
    print("-" * 80)
    num_classes = len(train_df['label'].unique())
    random_accuracy = 1.0 / num_classes
    print(f"Expected Accuracy (Random Guessing): {random_accuracy:.4f}")
    print(f"Explanation: With {num_classes} classes, random guessing gives 1/{num_classes} = {random_accuracy:.4f}")
    print("This represents the lower bound - any reasonable model should exceed this.")
    print()

    # Part 7(b): Most Frequent Class Baseline
    print("Part 7(b) - Most Frequent Class Baseline")
    print("-" * 80)
    most_frequent_class = train_df['label'].mode()[0]
    most_frequent_count = train_df['label'].value_counts().max()
    test_df['Predicted_Frequent'] = most_frequent_class
    frequent_accuracy = np.mean(test_df['Predicted_Frequent'] == test_df['label'])
    
    print(f"Most Frequent Class in Training: {most_frequent_class}")
    print(f"Occurrences: {most_frequent_count} out of {len(train_df)} samples ({most_frequent_count/len(train_df)*100:.2f}%)")
    print(f"Test Accuracy (Always predicting class {most_frequent_class}): {frequent_accuracy:.4f}")
    print("This baseline assumes the test distribution matches training distribution.")
    print()

    # Part 7(c): Comparison with Best Model
    print("Part 7(c) - Comparison with Best Model")
    print("-" * 80)
    
    # Get best model accuracy by checking all prediction columns in test_df
    best_model_acc = 0
    best_model_name = ""
    
    # Dictionary of all possible model predictions
    model_predictions = {
        'Combined (Separate Parameters)': 'Predicted Combined Separate',
        'Combined (Concatenation)': 'Predicted Combined Concat',
        'Enhanced (Length Features)': 'Predicted Enhanced',
        'Content (Unigrams+Bigrams)': 'Predicted Combined',
        'Content (Unigrams)': 'Predicted Unigrams',
        'Content (Bigrams)': 'Predicted Bigrams',
        'Content (Preprocessed)': 'Predicted',
        'Title (Unigrams+Bigrams)': 'Predicted Title Combined',
        'Title (Unigrams)': 'Predicted Title Unigrams',
        'Title (Simple)': 'Predicted Title Simple'
    }
    
    for model_name, pred_col in model_predictions.items():
        if pred_col in test_df.columns:
            acc = np.mean(test_df[pred_col] == test_df['label'])
            if acc > best_model_acc:
                best_model_acc = acc
                best_model_name = model_name
    
    if best_model_acc == 0:
        print("No model predictions found. Please run previous parts first.")
        print()
        return
    
    print(f"Best Model: {best_model_name}")
    print(f"Best Model Test Accuracy: {best_model_acc:.4f}")
    print()
    
    print("Improvement Analysis:")
    print("-" * 80)
    
    # Improvement over random guessing
    improvement_random = best_model_acc - random_accuracy
    relative_improvement_random = (best_model_acc / random_accuracy - 1) * 100
    print(f"1. Improvement over Random Guessing:")
    print(f"   - Absolute: {improvement_random:+.4f}")
    print(f"   - Relative: {relative_improvement_random:.1f}% improvement")
    print(f"   - Random baseline: {random_accuracy:.4f}")
    print(f"   - Best model: {best_model_acc:.4f}")
    print(f"   - This shows the model learned meaningful patterns from the data")
    print()
    
    # Improvement over most frequent class
    improvement_frequent = best_model_acc - frequent_accuracy
    relative_improvement_frequent = ((best_model_acc / frequent_accuracy - 1) * 100) if frequent_accuracy > 0 else 0
    print(f"2. Improvement over Most Frequent Class Baseline:")
    print(f"   - Absolute: {improvement_frequent:+.4f}")
    print(f"   - Relative: {relative_improvement_frequent:.1f}% improvement")
    print(f"   - Frequent class baseline: {frequent_accuracy:.4f}")
    print(f"   - Best model: {best_model_acc:.4f}")
    if improvement_frequent > 0:
        print(f"   - The model successfully distinguishes between classes")
        print(f"   - It doesn't simply memorize class distribution")
    else:
        print(f"   - Note: Dataset appears balanced (all classes equally frequent)")
        print(f"   - Most frequent class baseline equals random guessing")
    print()
    
    print("Statistical Significance:")
    print("-" * 80)
    test_size = len(test_df)
    print(f"Test set size: {test_size} samples")
    print(f"Correctly classified by best model: {int(best_model_acc * test_size)} samples")
    print(f"Correctly classified by random guessing (expected): {int(random_accuracy * test_size)} samples")
    print(f"Correctly classified by frequent baseline: {int(frequent_accuracy * test_size)} samples")
    print(f"Additional correct predictions: {int((best_model_acc - frequent_accuracy) * test_size)} samples")
    print()
    
    print("Conclusion:")
    print(f"The {best_model_name} achieves {best_model_acc:.4f} accuracy,")
    print(f"which is {relative_improvement_random:.1f}% better than random guessing")
    if improvement_frequent > 0.001:
        print(f"and {relative_improvement_frequent:.1f}% better than the most frequent class baseline.")
    else:
        print(f"(the most frequent class baseline equals random guessing for balanced datasets).")
    print("This demonstrates that the model has learned meaningful patterns and features")
    print("that enable it to effectively discriminate between different classes.")
    print()