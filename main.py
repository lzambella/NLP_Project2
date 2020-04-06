import pickle
from mle_probability import MLEProbability as ngram_prob
from sentence_generator import SentenceGenerator
from os import path
import sys
import re

if __name__ == "__main__":
    test_bigram = ngram_prob()

    # File paths for corpus and pickle
    corpus_path = sys.argv[1]   # Corpus location
    pickle_file = "web_distributions_subset_smoothed.p"
    # Whether to smooth or not
    laplace_smooth = True

    # Check if distribution has been determined beforehand and load it to save time (This doesn't work)
    #if (path.exists(pickle_file)):
    #    probabilities = pickle.load(open(pickle_file, "r"))
    if True:
        corpus = open(corpus_path, encoding="utf8")
        print(f"Loading corpus and uni/bigram frequencies from {corpus_path}")
        for line in corpus:
            # This corpus numbers each line, we want to remove that number
            line_filtered = re.split("^[0-9]+", line)
            test_bigram.load_line(line_filtered[1] if len(line_filtered) > 1 else line)
        corpus.close()

        if (laplace_smooth == True):
            print("Smoothing dataset")
            
        print("Generating probabilities")
        probabilities = test_bigram.generate_bigram_probabilities(smooth=laplace_smooth)
        probabilities_unigram = test_bigram.get_unigram_probabilities(smooth=laplace_smooth)
        print("Normalizing probabilities to the token count for analysis")
        probabilities_normalized = test_bigram.generate_bigram_probabilities(smooth=laplace_smooth, normal=True)
    # Sort by probabilities ascending
    probabilities_normalized.sort(key = lambda x: x[2])
    probabilities_unigram.sort(key = lambda x: x[1])
    print(len(probabilities_unigram))
    # We want the largest probabilites first
    probabilities_normalized.reverse()
    probabilities_unigram.reverse()
    print("Number of bigram probabilites:", len(probabilities))
    print("Number of types:", test_bigram.get_type_count())
    print("Total number of tokens: ", test_bigram.get_token_count())
    print("Most probable bigrams:")
    for i in range(0, 10):
        print(probabilities_normalized[i])
    print()
    print("Most probable unigrams:")
    for i in range(0, 10):
        print(probabilities_unigram[i])
    print()
    print("Dumping distribution to binary file\n")
    pickle.dump(probabilities, open(pickle_file, "wb"))

    print("Procedurally generating nonsense...")
    generator = SentenceGenerator(probabilities)
    for i in range(0, 10):
        output = generator.generate_sentence(None, 20)
        print(f"{i}: {output}")