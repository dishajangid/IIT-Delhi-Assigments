import random

class Agent:
    def __init__(self, phoneme_table, vocabulary) -> None:
        """
        Initializes the agent with a phoneme table and vocabulary.
        """
        self.phoneme_table = phoneme_table
        self.vocabulary = vocabulary
        self.best_state = None

    def generate_variations(self, word):
        """
        Generate all possible variations of a word based on phoneme substitutions.
        Handles both single and double character phonemes.
        """
        variations = set()
        length = len(word)
        
        # Iterate over possible phoneme lengths (1 or 2)
        for phoneme_length in (1, 2):
            for i in range(length - phoneme_length + 1):
                substring = word[i:i+phoneme_length]
                for correct_phoneme, incorrect_phonemes in self.phoneme_table.items():
                    if substring in incorrect_phonemes:
                        new_word = word[:i] + correct_phoneme + word[i + phoneme_length:]
                        variations.add(new_word)
        
        variations.add(word)  # include the original word
        return variations

    def add_vocabulary_words(self, text, environment, best_cost):
        """
        Add vocabulary words to the start or end of the text and generate all combinations.
        Check if adding each vocabulary word improves the cost and keep only improved states.
        """
        words = text.split()
        improved_texts = []
        
        # Check for vocabulary words added to the start
        for vocab_word in self.vocabulary:
            # Add vocabulary word to the start
            new_state_start = vocab_word + ' ' + text
            cost_start = environment.compute_cost(new_state_start)
            if cost_start < best_cost:
                best_cost = cost_start
                improved_texts.append(new_state_start)
        
        # Check for vocabulary words added to the end
        for vocab_word in self.vocabulary:
            # Add vocabulary word to the end
            new_state_end = text + ' ' + vocab_word
            cost_end = environment.compute_cost(new_state_end)
            if cost_end < best_cost:
                best_cost = cost_end
                improved_texts.append(new_state_end)
        
        return improved_texts, best_cost

    def asr_corrector(self, environment):
        """
        Corrects the ASR output using a systematic improvement approach with dynamic iterations.
        """
        init_state = environment.init_state
        self.best_state = init_state
        best_cost = environment.compute_cost(init_state)
        num_iterations = 2  # Start with 2 iterations
        previous_best_states = set()
        
        print(f"Initial state: {init_state}")
        print(f"Initial cost: {best_cost}")

        iteration = 0
        while True:
            iteration += 1
            print(f"Iteration {iteration}")

            # Shuffle word order
            words = self.best_state.split()
            word_indices = list(range(len(words)))
            random.shuffle(word_indices)
            
            current_state = self.best_state
            improved = True
            
            while improved:
                improved = False
                words = current_state.split()
                
                for idx in word_indices:
                    original_word = words[idx]
                    variations = self.generate_variations(original_word)
                    
                    best_word_cost = best_cost
                    best_word_variation = original_word
                    
                    for variation in variations:
                        words[idx] = variation
                        new_state = ' '.join(words)
                        new_cost = environment.compute_cost(new_state)
                        
                        if new_cost < best_word_cost:
                            best_word_cost = new_cost
                            best_word_variation = variation
                    
                    # Update the word with the best variation
                    words[idx] = best_word_variation
                
                new_state = ' '.join(words)
                new_cost = environment.compute_cost(new_state)
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    self.best_state = new_state
                    current_state = new_state
                    improved = True
            
            # Incorporate vocabulary words at the end of the iterations
            improved_states, final_best_cost = self.add_vocabulary_words(self.best_state, environment, best_cost)
            
            # Update the best state with any improved states from vocabulary additions
            for state in improved_states:
                state_cost = environment.compute_cost(state)
                if state_cost < best_cost:
                    best_cost = state_cost
                    self.best_state = state
            
            # Print the best state found in this iteration
            print(f"Best state in iteration {iteration}: {self.best_state} with cost: {best_cost}")
            # Check if the best state has been seen before or if it's time to stop
            if self.best_state in previous_best_states:
                break
            previous_best_states.add(self.best_state)
            
            # Dynamically adjust the number of iterations
            if iteration == num_iterations:
                num_iterations += 1  # Increase the number of iterations for the next cycle

        # Update environment with the overall best state found
        environment.best_state = self.best_state
        print(f"Overall best state: {self.best_state} with cost: {best_cost}")

