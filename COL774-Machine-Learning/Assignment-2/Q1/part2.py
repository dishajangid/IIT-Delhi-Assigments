from helpers import *
from naive_bayes import NaiveBayes

test_accuracy_processed = None

def run(train_df, test_df):
    global test_accuracy_processed
    # ============================================================================
    # PART 2: Preprocessing (Stemming + Stopword Removal)
    # ============================================================================
    print("=" * 80)
    print("PART 2: PREPROCESSING (STEMMING + STOPWORD REMOVAL)")
    print("=" * 80)

    # Apply preprocessing and evaluate models
    train_df['Tokenized Content Stemming Only'] = train_df['content'].apply(preprocess_tokens, remove_stopwords=False, apply_stemming=True)
    test_df['Tokenized Content Stemming Only'] = test_df['content'].apply(preprocess_tokens, remove_stopwords=False, apply_stemming=True)
    train_accuracy_stemming_only, test_accuracy_stemming_only, _ = train_and_evaluate_model(train_df, test_df, 'Tokenized Content Stemming Only', 'Tokenized Content Stemming Only')

    train_df['Tokenized Content Stopwords Only'] = train_df['content'].apply(preprocess_tokens, remove_stopwords=True, apply_stemming=False)
    test_df['Tokenized Content Stopwords Only'] = test_df['content'].apply(preprocess_tokens, remove_stopwords=True, apply_stemming=False)
    train_accuracy_stopwords_only, test_accuracy_stopwords_only, _ = train_and_evaluate_model(train_df, test_df, 'Tokenized Content Stopwords Only', 'Tokenized Content Stopwords Only')

    train_df['Tokenized Content Processed'] = train_df['content'].apply(preprocess_tokens, remove_stopwords=True, apply_stemming=True)
    test_df['Tokenized Content Processed'] = test_df['content'].apply(preprocess_tokens, remove_stopwords=True, apply_stemming=True)

    train_accuracy_processed, test_accuracy_processed, _ = train_and_evaluate_model(train_df, test_df, 'Tokenized Content Processed', 'Tokenized Content Processed')

    # Print the results
    print(f"Train Accuracy (with stemming only): {train_accuracy_stemming_only:.4f}")
    print(f"Test Accuracy (with stemming only): {test_accuracy_stemming_only:.4f}")
    print(f"Train Accuracy (with stopwords only): {train_accuracy_stopwords_only:.4f}")
    print(f"Test Accuracy (with stopwords only): {test_accuracy_stopwords_only:.4f}")
    print(f"Train Accuracy (with both stemming and stopwords): {train_accuracy_processed:.4f}")
    print(f"Test Accuracy (with both stemming and stopwords): {test_accuracy_processed:.4f}")

    print(f"Train Accuracy change (stemming only): {train_accuracy_stemming_only - train_accuracy_processed:.4f}")
    print(f"Test Accuracy change (stemming only): {test_accuracy_stemming_only - test_accuracy_processed:.4f}")
    print(f"Train Accuracy change (stopwords only): {train_accuracy_stopwords_only - train_accuracy_processed:.4f}")
    print(f"Test Accuracy change (stopwords only): {test_accuracy_stopwords_only - test_accuracy_processed:.4f}")

    # ============================================================================
    # PART 2(b): Word clouds after preprocessing
    # ============================================================================
    generate_word_cloud(train_df, data_type="train")
    generate_word_cloud(test_df, data_type="test")
