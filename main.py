import pickle
from mle_probability import MLEProbability as ngram_prob
from sentence_generator import SentenceGenerator
from os import path
import re

if __name__ == "__main__":
    test_bigram = ngram_prob()

    # File paths for corpus and pickle
    corpus_path = "corpus_a.txt"
    pickle_file = "web_distributions_subset_smoothed.p"
    # Whether to smooth or not
    laplace_smooth = False

    # Check if distribution has been determined beforehand and load it to save time
    #if (path.exists(pickle_file)):
    #    probabilities = pickle.load(open(pickle_file, "r"))
    if True:
        corpus = open(corpus_path, encoding="utf8")
        print("Loading corpus and uni/bigram frequencies")
        for line in corpus:
            # This corpus numbers each line, we want to remove that number
            line_filtered = re.split("^[0-9]+", line)
            test_bigram.load_frequency(line_filtered[1])
        corpus.close()

        if (laplace_smooth == True):
            print("Smoothing dataset")
            test_bigram.smooth_unigrams()
            
        print("Generating probabilities")
        probabilities = test_bigram.generate_bigram_probabilities()
        #print("Generating them again")
    # Sort by probabilities ascending
    probabilities.sort(key = lambda x: x[2])
    # We want the largest probabilites first
    probabilities.reverse()
    print("Number of bigram probabilites:", len(probabilities))
    print("Number of types:", test_bigram.get_type_count())
    print("Total number of tokens: ", test_bigram.get_token_count())
    print("Most probable bigrams:")
    for i in range(0, 10):
        print(probabilities[i])

    print("Dumping distribution to binary file\n")
    pickle.dump(probabilities, open(pickle_file, "wb"))

    print("Procedurally generating nonsense...")
    generator = SentenceGenerator(probabilities)
    for i in range(0, 10):
        output = generator.generate_sentence(None, 20)
        print(f"{i}: {output}")