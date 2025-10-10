# helpers.py

import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from naive_bayes import NaiveBayes

# Download NLTK data
nltk.download('stopwords', quiet=True)

# Global variables
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

class_names = {
    0: 'Class_0', 1: 'Class_1', 2: 'Class_2', 3: 'Class_3',
    4: 'Class_4', 5: 'Class_5', 6: 'Class_6', 7: 'Class_7',
    8: 'Class_8', 9: 'Class_9', 10: 'Class_10', 11: 'Class_11',
    12: 'Class_12', 13: 'Class_13'
}

def simple_tokenizer(text):
    """Simple tokenizer: convert to lowercase and split by whitespace."""
    if pd.isna(text):
        return []
    return str(text).lower().split()

def preprocess_tokens(text, remove_stopwords=True, apply_stemming=True):
    """Tokenize, optionally remove stopwords, and optionally apply stemming."""
    if pd.isna(text):
        return []
    tokens = str(text).lower().split()
    
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    
    if apply_stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

def create_unigrams(text):
    """Create unigrams (individual words) from preprocessed text."""
    if pd.isna(text):
        return []
    tokens = str(text).lower().split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

def create_bigrams(text):
    """Create bigrams (pairs of consecutive words) from preprocessed text."""
    if pd.isna(text):
        return []
    tokens = str(text).lower().split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return bigrams

def create_unigrams_bigrams(text):
    """Create both unigrams and bigrams from preprocessed text."""
    if pd.isna(text):
        return []
    tokens = str(text).lower().split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    features = tokens.copy()
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i+1]}"
        features.append(bigram)
    return features

def train_and_evaluate_model(train_data, test_data, train_col, test_col, nb_model=None):
    """Train NaiveBayes model and evaluate accuracy."""
    if nb_model is None:
        nb_model = NaiveBayes()
    
    nb_model.fit(train_data, smoothening=1.0, class_col='label', text_col=train_col)
    nb_model.predict(test_data, text_col=test_col, predicted_col='Predicted')
    test_accuracy = np.mean(test_data['Predicted'] == test_data['label'])
    nb_model.predict(train_data, text_col=train_col, predicted_col='Predicted')
    train_accuracy = np.mean(train_data['Predicted'] == train_data['label'])
    return train_accuracy, test_accuracy, nb_model

def generate_word_cloud(df, data_type="train", col='Tokenized Content'):
    """Generate word clouds for a given dataset."""
    print(f"Generating word clouds for {data_type} data ({col})...")
    num_classes = len(df['label'].unique())
    rows = (num_classes // 4) + (num_classes % 4 > 0)
    cols = min(4, num_classes)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.ravel()

    for i, cls in enumerate(sorted(df['label'].unique())):
        class_docs = df[df['label'] == cls][col]
        all_words = []
        for tokens in class_docs:
            all_words.extend(tokens)
        word_freq = Counter(all_words)
        if word_freq:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{class_names[cls]} (Class {cls})', fontsize=12)
        else:
            axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_file = f'output/part_wordclouds_{col}_{data_type}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Word clouds saved to '{output_file}'")
    print()