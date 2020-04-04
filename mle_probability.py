'''
Luke Zambella
CSC470 -- Natural Languuage Processessing

Class for calculating MLE probabilities using Bigrams and Unigrams
'''
from collections import Counter
import multiprocessing


class MLEProbability:
    '''
    Class for calculating probability estimates for types in a corpora using MLE
    '''
    def __init__(self):
        '''
        Constructor method
        '''
        self.unigrams = {}        # List of unigrams as a dictionary
        self.bigrams_dict = {}    # Bigrams as a dict

        self.unigram_count = 0
        self.bigram_count = 0

        self.__token_count = 0    # Number of tokens loaded

        self.__bigrams = []       # List of bigram objects
        self.keys = None
        

    def load_line(self, input_string = ""):
        '''
        Load the unigrams of a given string
        Iterative in that the count list is updated rather than replaced

        input_string: The string to parse from
        '''

        # Sanitize the input
        # We don't count punctuation or symbols as a form of a word except for possessive nouns.
        input_sanitized = self.__sanitize_input(input_string)

        # Count the words
        word_list = input_sanitized.split()
        self.__token_count = self.__token_count + len(word_list)

        # Iterate through each word in the split list and add as type if it doesnt exist
        for word in word_list:
            try:
                # Attempt to update unigram frequency
                self.unigrams[word] = self.unigrams[word] + 1
            except KeyError:
                # Add as new unigram
                self.unigrams[word] = 1

        # Generate bigrams for the input
        
        self.generate_bigram_frequencies(word_list)
        self.unigram_count = len(self.unigrams)
        self.bigram_count = len(self.bigrams_dict)


    def get_unigram_probabilities(self):
        '''
        Method that computes the unigram probabilities
        This is simply the frequency of a type over the total amount of tokens

        Returns a probability distribution in the form of a dictionary: {word : frequency}
        '''
        prob_dist = {}
        for unigram in self.unigrams:
            prob_dist[unigram.word] = (unigram.frequency/self.__token_count)

        return prob_dist


    def generate_bigram_frequencies(self, tokens):
        '''
        Generates bigrams given a tokenized string (list of tokens in order)
        '''
        for i in range(0, len(tokens) - 1):
            try:
                # Attempt to update bigram frequency
                self.bigrams_dict[f"{tokens[i]} {tokens[i+1]}"] = self.bigrams_dict[f"{tokens[i]} {tokens[i+1]}"] + 1
            except KeyError:
                # Add as new bigram
                self.bigrams_dict[f"{tokens[i]} {tokens[i+1]}"] = 1

    def generate_bigram_probabilities(self):
        '''
        Generate the probabilities of a word occurring given another word precedes it
        A bigram tells us (word_n-1, word_n) so using word_n-1 as the current word, we can calculate the probabilities of possible of w_n+1 
        it uses the frequency of a type as the first word and the frequencies of all found bigrams with that type as the first word in the loaded text

        probabilities are store in a list of tuples (current word, next word, probability)

        Uses dictionaries and is faster
        '''
        # iterate all bigram objects with that type as the current word
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
        self.keys = list(self.bigrams_dict.keys())
        probabilities = pool.map(self.multicore_test, range(0, len(self.bigrams_dict)-1 ))
        return probabilities

    def multicore_test(self, i):
        '''
        Helper function that actually computes bigram probabilities via multiprocessing
        '''
        bigram_words = self.keys[i].split()
        return ( bigram_words[0], bigram_words[1] , (self.bigrams_dict[self.keys[i]] / self.unigram_count) )


    def smooth_unigrams(self):
        '''
        Smooth unigrams 
        '''
        for key in self.unigrams.keys():
            self.unigrams[key] += 1
            self.unigram_count += len(self.unigrams)

    def smooth_bigrams(self):
        '''
        Function to perform laplace smoothing on every single combination of possible bigrams
        That is, we consider every possible configuration of bigrams from our history and add one to the total token count and frequency of that bigram ocurring.
        No return value, but updates our bigram distribution and as such, we have to recalculate the probabilities again

        This is very slow
        '''
        # Create every single bigram combination from unigrams
        bigrams = {}
        for type_a in self.unigrams.keys():
            for type_b in self.unigrams.keys():
                bigrams[f"{type_a} {type_b}"] = 1
        print("Smoothed bigrams completed")
        # Append our dictionary to the master one 
        self.bigrams_dict.update(bigrams)
        # Add the new token count to the total amount of tokens
        # This is the amount of unigrams squared
        self.__token_count += len(self.unigrams) * len(self.unigrams)

    ### Helper Functions ###


    def get_types(self):
        return self.unigrams

    def get_type_count(self):
        return len(self.unigrams)
        
    def get_token_count(self):
        return self.__token_count

    def get_bigrams(self):
        return self.__bigrams

    def __sanitize_input(self, input_string):
        invalid_chars = ",./;[]\\-=<>?:\"{}|_+!@#$%^&*()~`\t"
        input_sanitized = input_string.strip()
        input_sanitized = input_sanitized.rstrip()
        input_sanitized = input_sanitized.lower()
        input_sanitized = input_sanitized.translate({ord(i): None for i in invalid_chars})
        return input_sanitized
