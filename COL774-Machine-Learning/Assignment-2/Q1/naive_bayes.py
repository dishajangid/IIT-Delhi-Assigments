import numpy as np
import pandas as pd

class NaiveBayes:
    """
    Naive Bayes Classifier for multiclass text classification.
    Uses Multinomial Naive Bayes with Laplace smoothing.
    """
    
    def __init__(self):
        """Initialize the Naive Bayes classifier."""
        self.class_priors = None  # P(class)
        self.word_probs = None    # P(word|class)
        self.vocabulary = None    # Unique words in training data
        self.classes = None       # Unique classes
        self.vocab_size = 0       # Size of vocabulary
        
    def fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        """
        Learn the parameters of the model from the training data.
        Classes are 1-indexed.

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                Each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter (alpha).
            class_col (str): Name of the column containing class labels.
            text_col (str): Name of the column containing tokenized text.
        """
        # Extract classes (1-indexed)
        self.classes = np.sort(df[class_col].unique())
        num_classes = len(self.classes)
        
        # Build vocabulary from all documents
        all_words = []
        for tokens in df[text_col]:
            all_words.extend(tokens)
        self.vocabulary = list(set(all_words))
        self.vocab_size = len(self.vocabulary)
        
        # Create word to index mapping for efficient lookup
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        # Calculate class priors: P(class) = count(class) / total_documents
        total_docs = len(df)
        self.class_priors = np.zeros(num_classes)
        
        for i, cls in enumerate(self.classes):
            class_count = np.sum(df[class_col] == cls)
            self.class_priors[i] = class_count / total_docs
        
        # Calculate word probabilities: P(word|class)
        # Using log probabilities to avoid underflow
        self.word_probs = np.zeros((num_classes, self.vocab_size))
        
        for i, cls in enumerate(self.classes):
            # Get all documents belonging to this class
            class_docs = df[df[class_col] == cls][text_col]
            
            # Count word occurrences in this class
            word_counts = np.zeros(self.vocab_size)
            total_words = 0
            
            for tokens in class_docs:
                for word in tokens:
                    if word in self.word_to_idx:
                        word_counts[self.word_to_idx[word]] += 1
                        total_words += 1
            
            # Apply Laplace smoothing and calculate log probabilities
            # P(word|class) = (count(word, class) + alpha) / (total_words_in_class + alpha * vocab_size)
            denominator = total_words + smoothening * self.vocab_size
            
            for j in range(self.vocab_size):
                numerator = word_counts[j] + smoothening
                self.word_probs[i, j] = np.log(numerator / denominator)
        
        # Store log of class priors for efficiency
        self.log_class_priors = np.log(self.class_priors)
    
    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                Each entry of text_col is a list of tokens.
            text_col (str): Name of the column containing tokenized text.
            predicted_col (str): Name of the column to store predictions.
        """
        predictions = []
        
        for tokens in df[text_col]:
            # Calculate log posterior for each class
            # log P(class|document) = log P(class) + sum(log P(word|class))
            log_posteriors = np.copy(self.log_class_priors)
            
            for word in tokens:
                if word in self.word_to_idx:
                    word_idx = self.word_to_idx[word]
                    # Add log probability of this word for each class
                    log_posteriors += self.word_probs[:, word_idx]
            
            # Predict the class with maximum log posterior
            predicted_class_idx = np.argmax(log_posteriors)
            predicted_class = self.classes[predicted_class_idx]
            predictions.append(predicted_class)
        
        # Add predictions to dataframe
        df[predicted_col] = predictions
