from helpers import *
from naive_bayes import NaiveBayes

test_accuracy_bigrams = None

def run(train_df, test_df):
    global test_accuracy_bigrams  
    # ============================================================================
    # PART 3: Bigrams vs Unigrams
    # ============================================================================
    print("=" * 80)
    print("PART 3: UNIGRAMS vs BIGRAMS")
    print("=" * 80)

    # Apply unigram and bigram tokenization
    train_df['Tokenized Content Unigrams'] = train_df['content'].apply(create_unigrams)
    test_df['Tokenized Content Unigrams'] = test_df['content'].apply(create_unigrams)

    train_df['Tokenized Content Bigrams'] = train_df['content'].apply(create_bigrams)
    test_df['Tokenized Content Bigrams'] = test_df['content'].apply(create_bigrams)

    # Train and evaluate unigrams
    nb_content_unigrams = NaiveBayes()
    nb_content_unigrams.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Content Unigrams')
    nb_content_unigrams.predict(train_df, text_col='Tokenized Content Unigrams', predicted_col='Predicted Unigrams')
    train_accuracy_unigrams = np.mean(train_df['Predicted Unigrams'] == train_df['label'])
    nb_content_unigrams.predict(test_df, text_col='Tokenized Content Unigrams', predicted_col='Predicted Unigrams')
    test_accuracy_unigrams = np.mean(test_df['Predicted Unigrams'] == test_df['label'])

    print(f"Training Accuracy (Unigrams): {train_accuracy_unigrams:.4f}")
    print(f"Test Accuracy (Unigrams): {test_accuracy_unigrams:.4f}")
    print()

    # Train and evaluate bigrams
    nb_content_bigrams = NaiveBayes()
    nb_content_bigrams.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Content Bigrams')
    nb_content_bigrams.predict(train_df, text_col='Tokenized Content Bigrams', predicted_col='Predicted Bigrams')
    train_accuracy_bigrams = np.mean(train_df['Predicted Bigrams'] == train_df['label'])
    nb_content_bigrams.predict(test_df, text_col='Tokenized Content Bigrams', predicted_col='Predicted Bigrams')
    test_accuracy_bigrams = np.mean(test_df['Predicted Bigrams'] == test_df['label'])

    print(f"Training Accuracy (Bigrams): {train_accuracy_bigrams:.4f}")
    print(f"Test Accuracy (Bigrams): {test_accuracy_bigrams:.4f}")
    print()
