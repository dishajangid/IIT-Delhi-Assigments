from helpers import *
from naive_bayes import NaiveBayes

# Declare global variable at module level
test_accuracy_enhanced = None

def run(train_df, test_df):
    global test_accuracy_enhanced
    
    # ============================================================================
    # PART 9: Additional Feature Engineering
    # ============================================================================
    print("=" * 80)
    print("PART 9: ADDITIONAL FEATURE ENGINEERING")
    print("=" * 80)

    print("\nFeature Engineering Strategy:")
    print("-" * 80)
    print("Adding length-based features to capture document structure:")
    print("  1. Title length indicators:")
    print("     - SHORT_TITLE: < 5 tokens")
    print("     - LONG_TITLE: > 15 tokens")
    print("  2. Content length indicators:")
    print("     - SHORT_CONTENT: < 50 tokens")
    print("     - LONG_CONTENT: > 200 tokens")
    print()
    print("Rationale:")
    print("  - Different document categories may have characteristic lengths")
    print("  - News articles vs. blog posts vs. research papers have different structures")
    print("  - Length features can capture these structural patterns")
    print()

    def create_features_with_length(title, content):
        """Add length-based features to unigrams and bigrams."""
        title_tokens = create_unigrams_bigrams(title)
        content_tokens = create_unigrams_bigrams(content)
        features = title_tokens + content_tokens
        
        # Add length-based features
        title_length = len(title_tokens)
        content_length = len(content_tokens)
        
        # Title length features
        if title_length < 5:
            features.append('_SHORT_TITLE_')
        elif title_length > 15:
            features.append('_LONG_TITLE_')
        else:
            features.append('_MEDIUM_TITLE_')
        
        # Content length features
        if content_length < 50:
            features.append('_SHORT_CONTENT_')
        elif content_length > 200:
            features.append('_LONG_CONTENT_')
        else:
            features.append('_MEDIUM_CONTENT_')
        
        # Add combined length indicator
        if title_length < 5 and content_length < 50:
            features.append('_SHORT_DOCUMENT_')
        elif title_length > 15 and content_length > 200:
            features.append('_LONG_DOCUMENT_')
        
        return features

    print("Applying feature engineering...")
    train_df['Tokenized Enhanced'] = train_df.apply(
        lambda row: create_features_with_length(row['title'], row['content']), axis=1)
    test_df['Tokenized Enhanced'] = test_df.apply(
        lambda row: create_features_with_length(row['title'], row['content']), axis=1)

    print("Training enhanced model...")
    nb_enhanced = NaiveBayes()
    nb_enhanced.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Enhanced')
    
    nb_enhanced.predict(train_df, text_col='Tokenized Enhanced', predicted_col='Predicted Enhanced')
    train_accuracy_enhanced = np.mean(train_df['Predicted Enhanced'] == train_df['label'])
    
    nb_enhanced.predict(test_df, text_col='Tokenized Enhanced', predicted_col='Predicted Enhanced')
    test_accuracy_enhanced = np.mean(test_df['Predicted Enhanced'] == test_df['label'])

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Get previous best model accuracy
    best_previous_acc = 0
    best_previous_name = ""
    
    if 'Predicted Combined Separate' in test_df.columns:
        acc = np.mean(test_df['Predicted Combined Separate'] == test_df['label'])
        if acc > best_previous_acc:
            best_previous_acc = acc
            best_previous_name = "Combined (Separate Parameters)"
    
    if 'Predicted Combined Concat' in test_df.columns:
        acc = np.mean(test_df['Predicted Combined Concat'] == test_df['label'])
        if acc > best_previous_acc:
            best_previous_acc = acc
            best_previous_name = "Combined (Concatenation)"
    
    print(f"\nPrevious Best Model ({best_previous_name}):")
    print(f"  - Test Accuracy: {best_previous_acc:.4f}")
    print()
    print(f"Enhanced Model (with length features):")
    print(f"  - Training Accuracy: {train_accuracy_enhanced:.4f}")
    print(f"  - Test Accuracy: {test_accuracy_enhanced:.4f}")
    print()
    
    improvement = test_accuracy_enhanced - best_previous_acc
    print(f"Change in Test Accuracy: {improvement:+.4f}")
    
    if improvement > 0:
        print(f"✓ Improvement of {improvement:.4f} ({(improvement/best_previous_acc)*100:.2f}%)")
        print("\nAnalysis:")
        print("  - The length-based features successfully captured additional patterns")
        print("  - Different document categories exhibit characteristic length distributions")
        print("  - This structural information complements the textual content")
    elif improvement < -0.001:
        print(f"✗ Decrease of {abs(improvement):.4f}")
        print("\nAnalysis:")
        print("  - The length-based features may have introduced noise")
        print("  - The features might not be discriminative enough for this dataset")
        print("  - The model might be overfitting to length patterns in training data")
    else:
        print("≈ Negligible change")
        print("\nAnalysis:")
        print("  - Length features don't significantly impact performance")
        print("  - The textual content already captures most discriminative information")
        print("  - Length may not be a strong indicator for these document categories")
    
    print()
    print("Feature Statistics:")
    print("-" * 80)
    
    # Analyze length distributions
    title_lengths = train_df['title'].apply(lambda x: len(create_unigrams_bigrams(x)))
    content_lengths = train_df['content'].apply(lambda x: len(create_unigrams_bigrams(x)))
    
    print(f"Title Length Statistics:")
    print(f"  - Mean: {title_lengths.mean():.2f} tokens")
    print(f"  - Median: {title_lengths.median():.2f} tokens")
    print(f"  - Std Dev: {title_lengths.std():.2f} tokens")
    print(f"  - Min: {title_lengths.min()}, Max: {title_lengths.max()}")
    print()
    print(f"Content Length Statistics:")
    print(f"  - Mean: {content_lengths.mean():.2f} tokens")
    print(f"  - Median: {content_lengths.median():.2f} tokens")
    print(f"  - Std Dev: {content_lengths.std():.2f} tokens")
    print(f"  - Min: {content_lengths.min()}, Max: {content_lengths.max()}")
    print()
    
    print("Conclusion:")
    print("Length-based features can provide additional structural information about")
    print("documents. Their effectiveness depends on whether document categories have")
    print("characteristic length patterns. In this experiment, we observe")
    if improvement > 0:
        print("that these features provide meaningful improvements to classification accuracy.")
    else:
        print("that textual content remains the primary discriminative factor.")
    print()