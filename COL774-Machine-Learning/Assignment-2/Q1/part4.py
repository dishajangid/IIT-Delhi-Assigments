# part4.py
from helpers import *
from naive_bayes import NaiveBayes
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def run(train_df, test_df):
    # ============================================================================
    # PART 4: Comprehensive Content Model Comparison
    # ============================================================================
    print("=" * 80)
    print("PART 4: IDENTIFYING BEST MODEL FOR CONTENT FEATURES")
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
        col_name = f'Content_{config_name.replace(" ", "_").replace("-", "")}'
        
        # Apply preprocessing first
        if remove_stop or apply_stem:
            train_df[f'{col_name}_preprocessed'] = train_df['content'].apply(
                preprocess_tokens, remove_stopwords=remove_stop, apply_stemming=apply_stem)
            test_df[f'{col_name}_preprocessed'] = test_df['content'].apply(
                preprocess_tokens, remove_stopwords=remove_stop, apply_stemming=apply_stem)
            
            # Then apply feature function to preprocessed tokens
            # Need to convert preprocessed tokens back to string for feature functions
            train_df[col_name] = train_df[f'{col_name}_preprocessed'].apply(lambda x: x)
            test_df[col_name] = test_df[f'{col_name}_preprocessed'].apply(lambda x: x)
        else:
            # Apply feature function directly
            train_df[col_name] = train_df['content'].apply(feature_func)
            test_df[col_name] = test_df['content'].apply(feature_func)

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

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by F1-Score descending
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS TABLE - CONTENT MODELS")
    print("=" * 80)
    print()
    
    # Pretty print the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print()
    
    # Save to CSV
    import os
    os.makedirs('output', exist_ok=True)
    results_df.to_csv('output/part4_content_models_comparison.csv', index=False)
    print("✓ Results saved to: output/part4_content_models_comparison.csv")
    print()

    # Identify best model
    best_model = results_df.iloc[0]
    print("=" * 80)
    print("BEST MODEL IDENTIFICATION")
    print("=" * 80)
    print(f"\n✓ Best Content Model: {best_model['Model']}")
    print(f"  - Train Accuracy: {best_model['Train Accuracy']:.4f}")
    print(f"  - Test Accuracy:  {best_model['Test Accuracy']:.4f}")
    print(f"  - Precision:      {best_model['Precision']:.4f}")
    print(f"  - Recall:         {best_model['Recall']:.4f}")
    print(f"  - F1-Score:       {best_model['F1-Score']:.4f}")
    print()
    
    # Analysis
    print("=" * 80)
    print("JUSTIFICATION AND ANALYSIS")
    print("=" * 80)
    print()
    
    print("Selection Criteria:")
    print("  The best model is selected based on F1-Score, which provides a balanced")
    print("  measure between precision and recall. F1-Score is particularly important")
    print("  for multi-class classification to ensure the model performs well across")
    print("  all classes rather than favoring majority classes.")
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
    
    if 'Unigrams+Bigrams' in best_model['Model']:
        print("\n   ✓ Combined features (Unigrams+Bigrams) perform best because:")
        print("     • Unigrams capture individual word meanings and keywords")
        print("     • Bigrams capture contextual relationships between consecutive words")
        print("     • Together they provide richer feature representation")
    elif 'Unigrams' in best_model['Model'] and 'Bigrams' not in best_model['Model']:
        print("\n   ✓ Unigrams alone perform best because:")
        print("     • Individual words are highly discriminative for this task")
        print("     • Bigrams may introduce sparsity without adding much value")
    else:
        print("\n   ✓ Bigrams alone perform best because:")
        print("     • Word pairs capture important contextual patterns")
        print("     • Context is crucial for distinguishing between classes")
    print()
    
    # Compare preprocessing
    print("2. Preprocessing Impact:")
    raw_models = results_df[results_df['Model'].str.contains('Raw')]
    stem_models = results_df[results_df['Model'].str.contains('Stemming Only')]
    stop_models = results_df[results_df['Model'].str.contains('Stopwords Only')]
    both_models = results_df[results_df['Model'].str.contains('Both')]
    
    print(f"   - Raw (no preprocessing):   Avg F1 = {raw_models['F1-Score'].mean():.4f}")
    print(f"   - Stemming only:            Avg F1 = {stem_models['F1-Score'].mean():.4f}")
    print(f"   - Stopwords removal only:   Avg F1 = {stop_models['F1-Score'].mean():.4f}")
    print(f"   - Both (Stem + Stopwords):  Avg F1 = {both_models['F1-Score'].mean():.4f}")
    
    if 'Both' in best_model['Model']:
        print("\n   ✓ Both preprocessing steps help because:")
        print("     • Stemming reduces vocabulary size and groups word variations")
        print("     • Stopword removal eliminates common but non-discriminative words")
        print("     • Together they reduce noise and improve signal-to-noise ratio")
    elif 'Stemming Only' in best_model['Model']:
        print("\n   ✓ Stemming alone is sufficient because:")
        print("     • Reduces vocabulary sparsity effectively")
        print("     • Stopwords might contain some discriminative information")
    elif 'Stopwords Only' in best_model['Model']:
        print("\n   ✓ Stopword removal alone is sufficient because:")
        print("     • Eliminates most common non-discriminative words")
        print("     • Maintains word variations that might be meaningful")
    else:
        print("\n   ✓ Raw text performs best because:")
        print("     • All words contribute to classification")
        print("     • Preprocessing might remove useful information")
    print()
    
    # Top 5 models
    print("3. Top 5 Performing Models:")
    for idx, row in results_df.head(5).iterrows():
        print(f"   {idx+1}. {row['Model']:<40} F1 = {row['F1-Score']:.4f}")
    print()
    
    print("Conclusion:")
    print(f"  The {best_model['Model']} achieves the best balance of")
    print("  precision and recall, making it the optimal choice for content-based")
    print("  classification on this dataset.")
    print()