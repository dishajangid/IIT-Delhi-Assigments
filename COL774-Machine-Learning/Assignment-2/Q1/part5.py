from helpers import *
from naive_bayes import NaiveBayes
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os

# Declare global variables at module level
test_accuracy_title_processed = None
test_accuracy_title_bigrams = None
test_accuracy_title_combined = None

def run(train_df, test_df):
    global test_accuracy_title_processed, test_accuracy_title_bigrams, test_accuracy_title_combined
    
    # ============================================================================
    # PART 5: Comprehensive Title Model Analysis (Following Parts 1-4)
    # ============================================================================
    print("=" * 80)
    print("PART 5: COMPREHENSIVE TITLE MODEL ANALYSIS")
    print("=" * 80)
    print("\nFollowing the same pipeline as Parts 1-4, but for TITLE features")
    print()

    # ============================================================================
    # Step 1: Basic Naive Bayes with Title (like Part 1)
    # ============================================================================
    print("Step 1: Basic Naive Bayes with Title")
    print("-" * 80)
    
    train_df['Tokenized Title Simple'] = train_df['title'].apply(simple_tokenizer)
    test_df['Tokenized Title Simple'] = test_df['title'].apply(simple_tokenizer)

    nb_title_simple = NaiveBayes()
    nb_title_simple.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Title Simple')
    nb_title_simple.predict(train_df, text_col='Tokenized Title Simple', predicted_col='Predicted Title Simple')
    train_acc_simple = np.mean(train_df['Predicted Title Simple'] == train_df['label'])
    nb_title_simple.predict(test_df, text_col='Tokenized Title Simple', predicted_col='Predicted Title Simple')
    test_acc_simple = np.mean(test_df['Predicted Title Simple'] == test_df['label'])

    print(f"Training Accuracy: {train_acc_simple:.4f}")
    print(f"Test Accuracy: {test_acc_simple:.4f}")
    print()

    # Generate word clouds
    print("Generating word clouds for title...")
    generate_word_cloud(train_df, data_type="train_title", col='Tokenized Title Simple')
    generate_word_cloud(test_df, data_type="test_title", col='Tokenized Title Simple')
    print()

    # ============================================================================
    # Step 2-4: Comprehensive Analysis (like Part 2-4)
    # ============================================================================
    print("=" * 80)
    print("COMPREHENSIVE TITLE MODEL COMPARISON")
    print("=" * 80)
    print("\nAnalyzing all combinations of:")
    print("  - Features: Unigrams, Bigrams, Unigrams+Bigrams")
    print("  - Preprocessing: No preprocessing, Stemming only, Stopwords only, Both")
    print()

    # Define all configurations to test
    configurations = [
        # Unigrams
        ("Unigrams - Raw", create_unigrams, False, False),
        ("Unigrams - Stemming Only", create_unigrams, False, True),
        ("Unigrams - Stopwords Only", create_unigrams, True, False),
        ("Unigrams - Both", create_unigrams, True, True),
        
        # Bigrams
        ("Bigrams - Raw", create_bigrams, False, False),
        ("Bigrams - Stemming Only", create_bigrams, False, True),
        ("Bigrams - Stopwords Only", create_bigrams, True, False),
        ("Bigrams - Both", create_bigrams, True, True),
        
        # Unigrams + Bigrams
        ("Unigrams+Bigrams - Raw", create_unigrams_bigrams, False, False),
        ("Unigrams+Bigrams - Stemming Only", create_unigrams_bigrams, False, True),
        ("Unigrams+Bigrams - Stopwords Only", create_unigrams_bigrams, True, False),
        ("Unigrams+Bigrams - Both", create_unigrams_bigrams, True, True),
    ]

    results = []
    y_true = test_df['label']
    
    print("Training and evaluating models...")
    print("-" * 80)

    for config_name, feature_func, remove_stop, apply_stem in configurations:
        print(f"Processing: {config_name}...")
        
        # Create tokenized columns with preprocessing
        col_name = f'Title_{config_name.replace(" ", "_").replace("-", "")}'
        
        # Apply preprocessing first
        if remove_stop or apply_stem:
            train_df[f'{col_name}_preprocessed'] = train_df['title'].apply(
                preprocess_tokens, remove_stopwords=remove_stop, apply_stemming=apply_stem)
            test_df[f'{col_name}_preprocessed'] = test_df['title'].apply(
                preprocess_tokens, remove_stopwords=remove_stop, apply_stemming=apply_stem)
            
            # Use preprocessed tokens directly
            train_df[col_name] = train_df[f'{col_name}_preprocessed'].apply(lambda x: x)
            test_df[col_name] = test_df[f'{col_name}_preprocessed'].apply(lambda x: x)
        else:
            # Apply feature function directly
            train_df[col_name] = train_df['title'].apply(feature_func)
            test_df[col_name] = test_df['title'].apply(feature_func)

        # Train model
        nb = NaiveBayes()
        nb.fit(train_df, smoothening=1.0, class_col='label', text_col=col_name)
        
        # Predict on train set
        nb.predict(train_df, text_col=col_name, predicted_col=f'Pred_{col_name}')
        train_acc = np.mean(train_df[f'Pred_{col_name}'] == train_df['label'])
        
        # Predict on test set
        nb.predict(test_df, text_col=col_name, predicted_col=f'Pred_{col_name}')
        y_pred = test_df[f'Pred_{col_name}']
        
        # Calculate metrics
        test_acc = np.mean(y_pred == y_true)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': config_name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Store for global variables
        if 'Unigrams - Both' in config_name:
            test_accuracy_title_processed = test_acc
        elif 'Unigrams+Bigrams - Both' in config_name:
            test_accuracy_title_bigrams = test_acc
            test_accuracy_title_combined = test_acc

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by F1-Score descending
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS TABLE - TITLE MODELS")
    print("=" * 80)
    print()
    
    # Pretty print the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print()
    
    # Save to CSV
    os.makedirs('output', exist_ok=True)
    results_df.to_csv('output/part5_title_models_comparison.csv', index=False)
    print("✓ Results saved to: output/part5_title_models_comparison.csv")
    print()

    # Identify best model
    best_title_model = results_df.iloc[0]
    print("=" * 80)
    print("BEST MODEL IDENTIFICATION")
    print("=" * 80)
    print(f"\n✓ Best Title Model: {best_title_model['Model']}")
    print(f"  - Train Accuracy: {best_title_model['Train Accuracy']:.4f}")
    print(f"  - Test Accuracy:  {best_title_model['Test Accuracy']:.4f}")
    print(f"  - Precision:      {best_title_model['Precision']:.4f}")
    print(f"  - Recall:         {best_title_model['Recall']:.4f}")
    print(f"  - F1-Score:       {best_title_model['F1-Score']:.4f}")
    print()
    
    # Analysis
    print("=" * 80)
    print("JUSTIFICATION AND ANALYSIS")
    print("=" * 80)
    print()
    
    print("Selection Criteria:")
    print("  The best model is selected based on F1-Score for balanced performance.")
    print("  For titles, which are shorter and more concise, capturing the right")
    print("  features is crucial for effective classification.")
    print()
    
    print("Key Observations:")
    print()
    
    # Compare feature types
    unigrams_best = results_df[results_df['Model'].str.contains('Unigrams - ')].iloc[0]
    bigrams_best = results_df[results_df['Model'].str.contains('Bigrams - ')].iloc[0]
    combined_best = results_df[results_df['Model'].str.contains('Unigrams\\+Bigrams')].iloc[0]
    
    print("1. Feature Type Comparison:")
    print(f"   - Best Unigrams model:          F1 = {unigrams_best['F1-Score']:.4f}")
    print(f"   - Best Bigrams model:           F1 = {bigrams_best['F1-Score']:.4f}")
    print(f"   - Best Unigrams+Bigrams model:  F1 = {combined_best['F1-Score']:.4f}")
    
    if 'Unigrams+Bigrams' in best_title_model['Model']:
        print("\n   ✓ Combined features perform best for titles because:")
        print("     • Every word in a title is carefully chosen and meaningful")
        print("     • Bigrams capture key phrases that are common in titles")
        print("     • Combined approach maximizes information extraction from limited text")
    elif 'Unigrams' in best_title_model['Model'] and 'Bigrams' not in best_title_model['Model']:
        print("\n   ✓ Unigrams perform best for titles because:")
        print("     • Individual keywords in titles are highly discriminative")
        print("     • Short titles may not have enough word pairs for meaningful bigrams")
    else:
        print("\n   ✓ Bigrams perform best for titles because:")
        print("     • Titles often contain key phrases (e.g., 'machine learning', 'climate change')")
        print("     • Word pairs capture the essence of the document topic")
    print()
    
    # Compare preprocessing
    print("2. Preprocessing Impact on Titles:")
    raw_models = results_df[results_df['Model'].str.contains('Raw')]
    stem_models = results_df[results_df['Model'].str.contains('Stemming Only')]
    stop_models = results_df[results_df['Model'].str.contains('Stopwords Only')]
    both_models = results_df[results_df['Model'].str.contains('Both')]
    
    print(f"   - Raw (no preprocessing):   Avg F1 = {raw_models['F1-Score'].mean():.4f}")
    print(f"   - Stemming only:            Avg F1 = {stem_models['F1-Score'].mean():.4f}")
    print(f"   - Stopwords removal only:   Avg F1 = {stop_models['F1-Score'].mean():.4f}")
    print(f"   - Both (Stem + Stopwords):  Avg F1 = {both_models['F1-Score'].mean():.4f}")
    
    if 'Both' in best_title_model['Model']:
        print("\n   ✓ Both preprocessing steps help for titles because:")
        print("     • Stemming groups word variations (even in short text)")
        print("     • Stopword removal focuses on content words")
        print("     • Critical for maximizing signal in limited text")
    print()
    
    # Top 5 models
    print("3. Top 5 Performing Title Models:")
    for idx, row in results_df.head(5).iterrows():
        print(f"   {idx+1}. {row['Model']:<40} F1 = {row['F1-Score']:.4f}")
    print()

    # ============================================================================
    # Comparison: Title vs Content Models
    # ============================================================================
    print("=" * 80)
    print("COMPARISON: TITLE vs CONTENT MODELS")
    print("=" * 80)
    print()

    # Load content results if available
    content_results_path = 'output/part4_content_models_comparison.csv'
    if os.path.exists(content_results_path):
        content_results_df = pd.read_csv(content_results_path)
        best_content_model = content_results_df.iloc[0]
        
        print("Best Content Model:")
        print(f"  Model: {best_content_model['Model']}")
        print(f"  Test Accuracy:  {best_content_model['Test Accuracy']:.4f}")
        print(f"  F1-Score:       {best_content_model['F1-Score']:.4f}")
        print()
        
        print("Best Title Model:")
        print(f"  Model: {best_title_model['Model']}")
        print(f"  Test Accuracy:  {best_title_model['Test Accuracy']:.4f}")
        print(f"  F1-Score:       {best_title_model['F1-Score']:.4f}")
        print()
        
        # Comparison
        acc_diff = best_title_model['Test Accuracy'] - best_content_model['Test Accuracy']
        f1_diff = best_title_model['F1-Score'] - best_content_model['F1-Score']
        
        print(f"Difference (Title - Content):")
        print(f"  Accuracy:  {acc_diff:+.4f}")
        print(f"  F1-Score:  {f1_diff:+.4f}")
        print()
        
        print("-" * 80)
        print("OBSERVATIONS")
        print("-" * 80)
        print()
        
        if best_content_model['Test Accuracy'] > best_title_model['Test Accuracy']:
            improvement_pct = ((best_content_model['Test Accuracy'] - best_title_model['Test Accuracy']) / best_title_model['Test Accuracy']) * 100
            print(f"Content-based models outperform title-based models by {abs(acc_diff):.4f} ({improvement_pct:.1f}%)")
            print()
            print("Reasons for Content superiority:")
            print("  • More textual information: Content has significantly more words")
            print("  • Richer context: Full sentences provide better contextual understanding")
            print("  • More features: Larger vocabulary for learning discriminative patterns")
            print("  • Better statistics: More word occurrences improve probability estimates")
            print()
            print("Title performance analysis:")
            print(f"  • Despite being shorter, titles achieve {best_title_model['Test Accuracy']:.4f} accuracy")
            print("  • Titles contain carefully chosen keywords that are highly discriminative")
            print("  • This explains reasonable performance with minimal text")
        else:
            improvement_pct = ((best_title_model['Test Accuracy'] - best_content_model['Test Accuracy']) / best_content_model['Test Accuracy']) * 100
            print(f"Title-based models surprisingly outperform content by {abs(acc_diff):.4f} ({improvement_pct:.1f}%)")
            print()
            print("Possible reasons:")
            print("  • Titles are carefully curated to be maximally informative")
            print("  • Less noise in titles compared to full content")
            print("  • Content may contain generic text that reduces discriminative power")
            print("  • Higher information density in titles")
        
        print()
        print("Implications:")
        if best_content_model['Test Accuracy'] > best_title_model['Test Accuracy']:
            print("  • Use full content when available for best accuracy")
            print("  • Titles can serve as a fast approximation when content is unavailable")
            print("  • Combined approach (Part 6) should leverage both for optimal results")
        else:
            print("  • Titles may be sufficient for many classification tasks")
            print("  • Consider computational efficiency: titles are much faster to process")
            print("  • Full content processing may not always justify the cost")
        print()
        
    else:
        print("Note: Content model results not found. Run Part 4 first for comparison.")
        print()
        print(f"Title model achieves {best_title_model['Test Accuracy']:.4f} accuracy")
        print("Will be compared with content models after Part 4 execution.")
        print()
    
    print("Conclusion:")
    print(f"  The {best_title_model['Model']} achieves the best performance")
    print("  for title-based classification, demonstrating that even with limited text,")
    print("  proper feature engineering and preprocessing can achieve strong results.")
    print()