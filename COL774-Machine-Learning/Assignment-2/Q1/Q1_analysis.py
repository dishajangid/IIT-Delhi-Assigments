"""
Assignment 2 - Question 1: Text Classification using Naive Bayes
Complete analysis script covering all parts (1-9)
"""

import pandas as pd
import numpy as np
from naive_bayes import NaiveBayes
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print("Available columns in train_df:", train_df.columns.tolist())

print("=" * 80)
print("DATA LOADING")
print("=" * 80)
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Classes: {sorted(train_df['label'].unique())}")
print()

# ============================================================================
# PART 1: Basic Naive Bayes with Content
# ============================================================================
print("=" * 80)
print("PART 1: BASIC NAIVE BAYES (CONTENT ONLY)")
print("=" * 80)


def simple_tokenizer(text):
    """Simple tokenizer: convert to lowercase and split by whitespace."""
    if pd.isna(text):
        return []
    return str(text).lower().split()

# Tokenize content
train_df['Tokenized Content'] = train_df['content'].apply(simple_tokenizer)
test_df['Tokenized Content'] = test_df['content'].apply(simple_tokenizer)

# Train modelmetrics
nb_content = NaiveBayes()
nb_content.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Content')

# Predictions
nb_content.predict(train_df, text_col='Tokenized Content', predicted_col='Predicted')
train_accuracy = np.mean(train_df['Predicted'] == train_df['label'])

nb_content.predict(test_df, text_col='Tokenized Content', predicted_col='Predicted')
test_accuracy = np.mean(test_df['Predicted'] == test_df['label'])

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print()

# Part 1(b): Word clouds for each class
print("Generating word clouds for each class...")
class_names = {
    0: 'Class_0', 1: 'Class_1', 2: 'Class_2', 3: 'Class_3',
    4: 'Class_4', 5: 'Class_5', 6: 'Class_6', 7: 'Class_7',
    8: 'Class_8', 9: 'Class_9', 10: 'Class_10', 11: 'Class_11',
    12: 'Class_12', 13: 'Class_13'
}
# Part 1(b): Word clouds for both training and test data (raw content)
def generate_wordclouds_for_data(df, data_type="train"):
    print(f"Generating word clouds for {data_type} data (Raw Content)...")
    # Determine number of classes
    num_classes = len(train_df['label'].unique())

    # Calculate rows and columns for subplots dynamically
    rows = (num_classes // 4) + (num_classes % 4 > 0)  # Calculate the number of rows based on the number of classes
    cols = min(4, num_classes)  # Max 4 columns per row

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.ravel()

    for i, cls in enumerate(sorted(train_df['label'].unique())):
        class_docs = train_df[train_df['label'] == cls]['Tokenized Content']
        all_words = []
        
        # Collect all tokens for this class
        for tokens in class_docs:
            all_words.extend(tokens)
        
        word_freq = Counter(all_words)
        
        if word_freq:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{class_names[cls]} (Class {cls})', fontsize=12)
        else:
            axes[i].axis('off')  # Hide the axis if no words

    # Turn off remaining axes that are empty
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    output_file = f'output/part1_wordclouds_raw_{data_type}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Word clouds saved to '{output_file}'")
    print()

# Generate word clouds for both train and test datasets using raw content
generate_wordclouds_for_data(train_df, data_type="train")
generate_wordclouds_for_data(test_df, data_type="test")

# ============================================================================
# PART 2: Preprocessing (Stemming + Stopword Removal)
# ============================================================================
print("=" * 80)
print("PART 2: PREPROCESSING (STEMMING + STOPWORD REMOVAL)")
print("=" * 80)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_tokens(text):
    """Tokenize, remove stopwords, and apply stemming."""
    if pd.isna(text):
        return []
    tokens = str(text).lower().split()
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

# Apply preprocessing
train_df['Tokenized Content Processed'] = train_df['content'].apply(preprocess_tokens)
test_df['Tokenized Content Processed'] = test_df['content'].apply(preprocess_tokens)

# Train preprocessed model
nb_content_processed = NaiveBayes()
nb_content_processed.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Content Processed')

# Predictions
nb_content_processed.predict(test_df, text_col='Tokenized Content Processed', predicted_col='Predicted Processed')
test_accuracy_processed = np.mean(test_df['Predicted Processed'] == test_df['label'])

print(f"Test Accuracy (with preprocessing): {test_accuracy_processed:.4f}")
print(f"Accuracy change: {test_accuracy_processed - test_accuracy:+.4f}")
print()


# ============================================================================
# PART 2(b): Word clouds after preprocessing for both Train and Test data
# ============================================================================

# Function to generate and save word clouds for a given dataset
def generate_word_clouds(df, data_type="train"):
    """Generate word clouds for a given dataset (train or test)"""
    print(f"Generating word clouds for {data_type} data...")

    # Create a figure with a dynamic number of rows and columns based on the number of classes
    num_classes = len(df['label'].unique())
    rows = (num_classes // 4) + (num_classes % 4 > 0)  # Calculate number of rows dynamically
    cols = min(4, num_classes)  # Max 4 columns per row

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.ravel()  # Flatten the axes array

    # Iterate over all classes to create word clouds for each class
    for i, cls in enumerate(sorted(df['label'].unique())):
        # Get the tokenized content for this class
        class_docs = df[df['label'] == cls]['Tokenized Content Processed']
        all_words = []
        
        # Collect all tokens for this class
        for tokens in class_docs:
            all_words.extend(tokens)

        word_freq = Counter(all_words)

        # Generate word cloud only if there are words for this class
        if word_freq:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{class_names[cls]} (Class {cls})', fontsize=12)
        else:
            axes[i].axis('off')  # Hide the axis if no words

    # Turn off any remaining axes that are empty
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_file = f'output/part2_wordclouds_{data_type}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Word clouds saved to '{output_file}'")
    print()


# Generate word clouds for training data
generate_word_clouds(train_df, data_type="train")

# Generate word clouds for test data
generate_word_clouds(test_df, data_type="test")


# ============================================================================
# PART 3: Bigrams
# ============================================================================
print("=" * 80)
print("PART 3: BIGRAMS (UNIGRAMS + BIGRAMS)")
print("=" * 80)

def create_unigrams_bigrams(text):
    """Create both unigrams and bigrams from preprocessed text."""
    if pd.isna(text):
        return []
    tokens = str(text).lower().split()
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    # Create unigrams and bigrams
    features = tokens.copy()  # Unigrams
    
    # Add bigrams
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i+1]}"
        features.append(bigram)
    
    return features

# Apply unigram + bigram tokenization
train_df['Tokenized Content Bigrams'] = train_df['content'].apply(create_unigrams_bigrams)
test_df['Tokenized Content Bigrams'] = test_df['content'].apply(create_unigrams_bigrams)

# Train bigram model
nb_content_bigrams = NaiveBayes()
nb_content_bigrams.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Content Bigrams')

# Predictions
nb_content_bigrams.predict(train_df, text_col='Tokenized Content Bigrams', predicted_col='Predicted Bigrams')
train_accuracy_bigrams = np.mean(train_df['Predicted Bigrams'] == train_df['label'])

nb_content_bigrams.predict(test_df, text_col='Tokenized Content Bigrams', predicted_col='Predicted Bigrams')
test_accuracy_bigrams = np.mean(test_df['Predicted Bigrams'] == test_df['label'])

print(f"Training Accuracy (Unigrams + Bigrams): {train_accuracy_bigrams:.4f}")
print(f"Test Accuracy (Unigrams + Bigrams): {test_accuracy_bigrams:.4f}")
print()

def generate_wordclouds_for_bigrams(df, data_type="train"):
    print(f"Generating word clouds for {data_type} data (Unigrams + Bigrams)...")
    
    # Get the number of classes
    num_classes = len(df['label'].unique())

    # Calculate rows and columns dynamically based on the number of classes
    rows = (num_classes // 4) + (num_classes % 4 > 0)  # Calculate the number of rows
    cols = min(4, num_classes)  # Max 4 columns per row

    # Create subplots for word clouds (unigrams and bigrams)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.ravel()  # Flatten the axes array

    # Iterate over each class and generate word clouds for unigrams and bigrams
    for i, cls in enumerate(sorted(df['label'].unique())):
        # Unigrams
        class_docs = df[df['label'] == cls]['Tokenized Content Bigrams']  # Use bigrams here
        all_words = []

        # Collect all unigrams and bigrams for this class
        for doc in class_docs:
            all_words.extend(doc)

        word_freq = Counter(all_words)

        # Generate word cloud if there are words for this class
        if word_freq:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{class_names[cls]} (Class {cls})', fontsize=12)
        else:
            axes[i].axis('off')  # Hide the axis if no words

    # Turn off remaining axes that are empty
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    output_file = f'output/part3_wordclouds_unigrams_bigrams_{data_type}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Word clouds saved to '{output_file}'")
    print()


generate_wordclouds_for_bigrams(train_df, data_type="train")
generate_wordclouds_for_bigrams(test_df, data_type="test")

# ============================================================================
# PART 4: Model Comparison (Content)
# ============================================================================
print("=" * 80)
print("PART 4: MODEL COMPARISON (CONTENT)")
print("=" * 80)

# Compare all content models
models_content = {
    'Raw Content': test_accuracy,
    'Preprocessed Content': test_accuracy_processed,
    'Preprocessed + Bigrams': test_accuracy_bigrams
}

print("Model Comparison:")
for model_name, accuracy in models_content.items():
    print(f"  {model_name:30s}: {accuracy:.4f}")

best_content_model = max(models_content, key=models_content.get)
best_content_accuracy = models_content[best_content_model]
print(f"\nBest model for Content: {best_content_model} (Accuracy: {best_content_accuracy:.4f})")
print()

# Calculate additional metrics for best model
y_true = test_df['label']
y_pred = test_df['Predicted Bigrams']  # Assuming bigrams is best

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Additional metrics for best content model:")
print(f"  Precision (weighted): {precision:.4f}")
print(f"  Recall (weighted): {recall:.4f}")
print(f"  F1-Score (weighted): {f1:.4f}")
print()

# ============================================================================
# PART 5: Title Features
# ============================================================================
print("=" * 80)
print("PART 5: TITLE FEATURES")
print("=" * 80)

# Apply same preprocessing to titles
train_df['Tokenized Title Processed'] = train_df['title'].apply(preprocess_tokens)
test_df['Tokenized Title Processed'] = test_df['title'].apply(preprocess_tokens)

train_df['Tokenized Title Bigrams'] = train_df['title'].apply(create_unigrams_bigrams)
test_df['Tokenized Title Bigrams'] = test_df['title'].apply(create_unigrams_bigrams)

# Train models on title
nb_title_processed = NaiveBayes()
nb_title_processed.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Title Processed')
nb_title_processed.predict(test_df, text_col='Tokenized Title Processed', predicted_col='Predicted Title Processed')
test_accuracy_title_processed = np.mean(test_df['Predicted Title Processed'] == test_df['label'])

nb_title_bigrams = NaiveBayes()
nb_title_bigrams.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Title Bigrams')
nb_title_bigrams.predict(test_df, text_col='Tokenized Title Bigrams', predicted_col='Predicted Title Bigrams')
test_accuracy_title_bigrams = np.mean(test_df['Predicted Title Bigrams'] == test_df['label'])

print(f"Title - Preprocessed: {test_accuracy_title_processed:.4f}")
print(f"Title - Preprocessed + Bigrams: {test_accuracy_title_bigrams:.4f}")
print()
print(f"Best Content Model: {best_content_accuracy:.4f}")
print(f"Best Title Model: {max(test_accuracy_title_processed, test_accuracy_title_bigrams):.4f}")
print(f"Difference: {max(test_accuracy_title_processed, test_accuracy_title_bigrams) - best_content_accuracy:+.4f}")
print()

# ============================================================================
# PART 6: Combined Features
# ============================================================================
print("=" * 80)
print("PART 6: COMBINED FEATURES (TITLE + CONTENT)")
print("=" * 80)

# Part 6(a): Concatenation approach
def concatenate_features(row):
    """Concatenate title and content features."""
    return row['Tokenized Title Bigrams'] + row['Tokenized Content Bigrams']

train_df['Tokenized Combined Concat'] = train_df.apply(concatenate_features, axis=1)
test_df['Tokenized Combined Concat'] = test_df.apply(concatenate_features, axis=1)

nb_combined_concat = NaiveBayes()
nb_combined_concat.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Combined Concat')
nb_combined_concat.predict(test_df, text_col='Tokenized Combined Concat', predicted_col='Predicted Combined Concat')
test_accuracy_concat = np.mean(test_df['Predicted Combined Concat'] == test_df['label'])

print(f"Part 6(a) - Concatenation approach: {test_accuracy_concat:.4f}")
print()

# Part 6(b): Separate parameters for title and content
print("Part 6(b) - Separate parameters approach")
print("Mathematical formulation:")
print("  P(class|title, content) ∝ P(class) * P(title|class) * P(content|class)")
print("  where P(title|class) and P(content|class) are modeled separately")
print()

# Implementation: Train separate models and combine predictions
# For prediction, we use: log P(class|doc) = log P(class) + log P(title|class) + log P(content|class)

class NaiveBayesSeparate:
    """Naive Bayes with separate parameters for different feature sets."""
    
    def __init__(self):
        self.model_title = NaiveBayes()
        self.model_content = NaiveBayes()
        self.classes = None
        
    def fit(self, df, smoothening, class_col, title_col, content_col):
        """Train separate models for title and content."""
        self.model_title.fit(df, smoothening, class_col, title_col)
        self.model_content.fit(df, smoothening, class_col, content_col)
        self.classes = self.model_title.classes
        
    def predict(self, df, title_col, content_col, predicted_col):
        """Predict by combining log posteriors from both models."""
        predictions = []
        
        for idx, row in df.iterrows():
            title_tokens = row[title_col]
            content_tokens = row[content_col]
            
            # Calculate log posteriors for title
            log_posteriors_title = np.copy(self.model_title.log_class_priors)
            for word in title_tokens:
                if word in self.model_title.word_to_idx:
                    word_idx = self.model_title.word_to_idx[word]
                    log_posteriors_title += self.model_title.word_probs[:, word_idx]
            
            # Calculate log posteriors for content
            log_posteriors_content = np.copy(self.model_content.log_class_priors)
            for word in content_tokens:
                if word in self.model_content.word_to_idx:
                    word_idx = self.model_content.word_to_idx[word]
                    log_posteriors_content += self.model_content.word_probs[:, word_idx]
            
            # Combine: log P(class|title,content) = log P(class) + log P(title|class) + log P(content|class)
            # Note: log P(class) is already included in both, so we need to add it only once more
            log_posteriors_combined = log_posteriors_title + log_posteriors_content - self.model_title.log_class_priors
            
            predicted_class_idx = np.argmax(log_posteriors_combined)
            predicted_class = self.classes[predicted_class_idx]
            predictions.append(predicted_class)
        
        df[predicted_col] = predictions

nb_combined_separate = NaiveBayesSeparate()
nb_combined_separate.fit(train_df, smoothening=1.0, class_col='label', 
                         title_col='Tokenized Title Bigrams', content_col='Tokenized Content Bigrams')
nb_combined_separate.predict(test_df, title_col='Tokenized Title Bigrams', 
                             content_col='Tokenized Content Bigrams', predicted_col='Predicted Combined Separate')
test_accuracy_separate = np.mean(test_df['Predicted Combined Separate'] == test_df['label'])

print(f"Part 6(b) - Separate parameters: {test_accuracy_separate:.4f}")
print(f"Improvement over concatenation: {test_accuracy_separate - test_accuracy_concat:+.4f}")
print()

# ============================================================================
# PART 7: Baseline Comparisons
# ============================================================================
print("=" * 80)
print("PART 7: BASELINE COMPARISONS")
print("=" * 80)

# Random prediction
random_accuracy = 1.0 / len(train_df['label'].unique())
print(f"Random baseline accuracy: {random_accuracy:.4f}")

# Most frequent class prediction
most_frequent_class = train_df['label'].mode()[0]
test_df['Predicted Frequent'] = most_frequent_class
frequent_accuracy = np.mean(test_df['Predicted Frequent'] == test_df['label'])
print(f"Most frequent class baseline: {frequent_accuracy:.4f}")

best_model_accuracy = max(test_accuracy_concat, test_accuracy_separate)
print(f"\nBest model accuracy: {best_model_accuracy:.4f}")
print(f"Improvement over random: {best_model_accuracy - random_accuracy:.4f} ({(best_model_accuracy/random_accuracy - 1)*100:.1f}% improvement)")
print(f"Improvement over frequent: {best_model_accuracy - frequent_accuracy:.4f}")
print()

# ============================================================================
# PART 8: Confusion Matrix
# ============================================================================
print("=" * 80)
print("PART 8: CONFUSION MATRIX")
print("=" * 80)

# Use best model (separate parameters)
y_true = test_df['label']
y_pred = test_df['Predicted Combined Separate']

cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(1, 15), yticklabels=range(1, 15))
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix - Best Model (Separate Parameters)')
plt.tight_layout()
plt.savefig('output/part8_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix saved to 'output/part8_confusion_matrix.png'")

# Find class with highest diagonal entry
diagonal_values = np.diag(cm)
best_class = np.argmax(diagonal_values) + 1
print(f"\nClass with highest diagonal entry: Class {best_class}")
print(f"Value: {diagonal_values[best_class-1]} (out of {np.sum(cm[best_class-1, :])} samples)")
print(f"This means Class {best_class} has the highest number of correct predictions.")
print()

# ============================================================================
# PART 9: Additional Feature Engineering
# ============================================================================
print("=" * 80)
print("PART 9: ADDITIONAL FEATURE ENGINEERING")
print("=" * 80)

print("Additional feature: Length-based features")
print("  - Short title/content vs long title/content")
print("  - This can capture structural patterns in different categories")
print()

def create_features_with_length(title, content):
    """Add length-based features to unigrams and bigrams."""
    # Get base features
    title_tokens = create_unigrams_bigrams(title)
    content_tokens = create_unigrams_bigrams(content)
    
    # Add length indicators
    features = title_tokens + content_tokens
    
    # Add length-based features
    if len(title_tokens) < 5:
        features.append('_SHORT_TITLE_')
    elif len(title_tokens) > 15:
        features.append('_LONG_TITLE_')
    
    if len(content_tokens) < 50:
        features.append('_SHORT_CONTENT_')
    elif len(content_tokens) > 200:
        features.append('_LONG_CONTENT_')
    
    return features

train_df['Tokenized Enhanced'] = train_df.apply(
    lambda row: create_features_with_length(row['title'], row['content']), axis=1)
test_df['Tokenized Enhanced'] = test_df.apply(
    lambda row: create_features_with_length(row['title'], row['content']), axis=1)

nb_enhanced = NaiveBayes()
nb_enhanced.fit(train_df, smoothening=1.0, class_col='label', text_col='Tokenized Enhanced')
nb_enhanced.predict(test_df, text_col='Tokenized Enhanced', predicted_col='Predicted Enhanced')
test_accuracy_enhanced = np.mean(test_df['Predicted Enhanced'] == test_df['label'])

print(f"Enhanced model (with length features): {test_accuracy_enhanced:.4f}")
print(f"Previous best model: {best_model_accuracy:.4f}")
print(f"Change: {test_accuracy_enhanced - best_model_accuracy:+.4f}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("SUMMARY OF ALL MODELS")
print("=" * 80)

all_models = {
    'Raw Content': test_accuracy,
    'Preprocessed Content': test_accuracy_processed,
    'Content + Bigrams': test_accuracy_bigrams,
    'Title (Preprocessed)': test_accuracy_title_processed,
    'Title + Bigrams': test_accuracy_title_bigrams,
    'Combined (Concatenation)': test_accuracy_concat,
    'Combined (Separate Params)': test_accuracy_separate,
    'Enhanced (Length Features)': test_accuracy_enhanced
}

print("\nTest Accuracies:")
for model_name, accuracy in sorted(all_models.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name:30s}: {accuracy:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)