from helpers import *
from naive_bayes import NaiveBayes

# Declare global variable at module level
test_accuracy = None

def run(train_df, test_df):
    global test_accuracy
    
    # ============================================================================
    # PART 1: Basic Naive Bayes with Content
    # ============================================================================
    print("=" * 80)
    print("PART 1: BASIC NAIVE BAYES (CONTENT ONLY)")
    print("=" * 80)

    # Tokenize content
    train_df['Tokenized Content'] = train_df['content'].apply(simple_tokenizer)
    test_df['Tokenized Content'] = test_df['content'].apply(simple_tokenizer)

    # Train model
    nb_content = NaiveBayes()
    nb_content.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Content')

    # Predictions
    nb_content.predict(train_df, text_col='Tokenized Content', predicted_col='Predicted')
    train_accuracy = np.mean(train_df['Predicted'] == train_df['label'])
    
    nb_content.predict(test_df, text_col='Tokenized Content', predicted_col='Predicted')
    test_accuracy = np.mean(test_df['Predicted'] == test_df['label'])

    # Print the accuracies
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print()

    # Part 1(b): Word clouds
    print("Generating word clouds for each class...")
    generate_word_cloud(train_df, data_type="train", col='Tokenized Content')
    generate_word_cloud(test_df, data_type="test", col='Tokenized Content')