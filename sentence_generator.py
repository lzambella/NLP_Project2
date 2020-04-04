from random import randint


class SentenceGenerator:
    '''
    Generates sentences using probabilistic data
    '''

    def __init__(self, probabilistic_data):
        '''
        Constructor

        probabilistic_data -- list of 3-tuples in the form (current_word, next_word, probability of next_word given current word)
        '''
        self.__probabilistic_data = probabilistic_data

    def generate_sentence(self, starting_word=None, length=10):
        '''
        Generates a sentence using bigram probabilities using a starting word, if any, or a random word from the set
        '''
        out_str = ""
        current_word = ""
        # Get the starting word
        if starting_word != None:
            # Find starting all occurances of starting word in the probabilistic data
            out_str = starting_word
            bigrams = self.__find_word(starting_word)
            if len(bigrams) == 0:
                # If no match, choose a random starting word
                rand_bigram = randint(0, len(self.__probabilistic_data) - 1)
                out_str = out_str + " " + rand_bigram[0][0]
                current_word = rand_bigram[0][0]
            else:
                # Sort by probabilitiy and choose the most likely next word
                bigrams.sort(key = lambda x: x[2]) 
                bigrams.reverse()

                # Start the sentence
                out_str = out_str + " " + bigrams[0][1]
                current_word = bigrams[0][1]
        else:
            # Choose a random word if no word was specified
            rand_bigram = randint(0, len(self.__probabilistic_data) - 1)
            out_str = self.__probabilistic_data[rand_bigram][0]
            current_word = self.__probabilistic_data[rand_bigram][1]
        
        # Rest of the sentence
        for i in range (1, length):
            bigrams = self.__find_word(current_word)
            if len(bigrams) == 0:
                # If no match, choose a random word
                rand_bigram = randint(0, len(self.__probabilistic_data) - 1)
                out_str = out_str + " " + self.__probabilistic_data[rand_bigram][1]
                current_word = self.__probabilistic_data[rand_bigram][1]
            else:
                # Sort by probabilitiy and choose the most likely next word
                bigrams.sort(key = lambda x: x[2]) 
                bigrams.reverse()
                # if there are more than a few to choose from, have a chance of getting a different than the most likely (choose top 5 most likely)
                if len(bigrams) > 1:            # Only if there is another 
                    if randint(0,100) < 45:     # Roll d100
                        other_choice = randint(0, len(bigrams) - 1) % 10
                        out_str = out_str + " " + bigrams[other_choice][1]
                    else:
                        out_str = out_str + " " + bigrams[0][1]
                        current_word = bigrams[0][1]                       
                        
                else:                           # Choose the only one
                    out_str = out_str + " " + bigrams[0][1]
                    current_word = bigrams[0][1]

        return out_str


    ### Helper Functions ###

    def __find_word(self, word = ""):
        filtered_words = []
        for prob_tuple in self.__probabilistic_data:
            if prob_tuple[0] == word:
                filtered_words.append(prob_tuple)
        return filtered_words