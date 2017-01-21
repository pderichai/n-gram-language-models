import math
from collections import Counter

'''This module contains a main method that will train an interpolated n-gram language model on each specified dataset
then evaluate each language model on every test set from each corpus.'''

START = '<s>'
'''Start symbol---two are prepended to the start of ever sentence.'''

STOP = '</s>'
'''Stop symbol---one is appended to the end of every sentence.'''

UNK = '<UNK>'
'''Unknown word symbol used to describe any word that is out of vocabulary.'''

DATASETS = ['reuters', 'brown', 'gutenberg']
'''The names of the different datasets we want to model.'''

DEV = False
'''Whether or not to evaluate the models on the dev datasets or test datasets.'''

LAMBDA_1 = 0.2
LAMBDA_2 = 0.3
LAMBDA_3 = 0.5
'''Linear interpolation hyper-parameters.'''


def main():
    def get_log_prob(n_gram, model, caches):
        '''This is a nested method that will be passed to eval_mode().

        n_gram is a tuple of strings that represents an n-gram
        model is a tuple of Counter objects (unigrams, bigrams, trigrams) that map from n-grams to counts
        caches is a tuple of dicts (unigrams_to_probs, bigrams_to_probs, n_grams_to_interpolated_probs) that memoize
            the return values of various function calls
        '''

        if n_gram in caches[2]:
            return caches[2][n_gram]

        # uni-gram part
        if n_gram[2:] in caches[0]:
            unigram_part = caches[0][n_gram[2:]]
        else:
            uni_numer = model[0][n_gram[2:]]
            uni_denom = sum(model[0].values()) - model[0][(START,)]
            unigram_part = 0
            if uni_denom != 0:
                unigram_part = LAMBDA_3 * uni_numer / uni_denom
            caches[0][n_gram[2:]] = unigram_part

        # bi-gram part
        if n_gram[1:] in caches[1]:
            bigram_part = caches[1][n_gram[1:]]
        else:
            bi_numer = model[1][n_gram[1:]]
            bi_denom = model[0][n_gram[2:]]
            bigram_part = 0
            if bi_denom != 0:
                bigram_part = LAMBDA_2 * bi_numer / bi_denom
            caches[1][n_gram[1:]] = bigram_part

        # tri-gram part
        tri_numer = model[2][n_gram]
        tri_denom = model[1][n_gram[1:]]
        trigram_part = 0
        if tri_denom != 0:
            trigram_part = LAMBDA_1 * tri_numer / tri_denom

        prob = trigram_part + bigram_part + unigram_part
        log_prob = math.log(prob, 2)
        caches[2][n_gram] = log_prob
        return log_prob

    for train_dataset in DATASETS:
        print('training on ' + train_dataset + '...')
        unigrams, bigrams, trigrams = train('data/' + train_dataset + '_train.txt')
        model = (unigrams, bigrams, trigrams)

        # (unigrams_to_probs, bigrams_to_probs, n_grams_to_interpolated_probs)
        caches = (dict(), dict(), dict(), dict())

        for test_dataset in DATASETS:
            print('evaluating ' + train_dataset + ' on ' + test_dataset + ' test set...')
            if DEV:
                perplexity = eval_model('data/' + test_dataset + '_dev.txt', model, get_log_prob, caches)
            else:
                perplexity = eval_model('data/' + test_dataset + '_test.txt', model, get_log_prob, caches)

            print('trained on: ' + train_dataset + '; tested on: ' + test_dataset + '; perplexity: ' + str(perplexity))


def train(filename):
    '''Trains an n-gram model using the specified corpus.

    filename is the filename of the corpus'''

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()

    lines = list()
    with open(filename, encoding='iso-8859-1') as f:
        for line in f:
            lines.append(line)

    # creating the uni-gram counts
    for line in lines:
        tokens = line.split()
        tokens.insert(0, START)
        tokens.insert(0, START)
        tokens.append(STOP)
        add_n_gram_counts(1, unigrams, tokens)

    # the set of all uni-grams that have a count of 1
    unks = set()
    for unigram, count in unigrams.items():
        if count == 1:
            unks.add(unigram[0])

    for word in unks:
        del unigrams[(word,)]

    unigrams[(UNK,)] = len(unks)

    # creating the bigram and trigram counts
    for line in lines:
        tokens = [token if token not in unks else UNK for token in line.split()]
        tokens.insert(0, START)
        tokens.insert(0, START)
        tokens.append(STOP)
        add_n_gram_counts(2, bigrams, tokens)
        add_n_gram_counts(3, trigrams, tokens)

    return unigrams, bigrams, trigrams


def add_n_gram_counts(n, n_grams, tokens):
    '''Adds the n-grams to the specified Counter from the specified tokens.'''
    for i in range(len(tokens) - (n - 1)):
        n_grams[tuple(tokens[i:i+n])] += 1

    return n_grams


def eval_model(filename, model, log_prob_func, caches):
    '''Returns the perplexity of the model on a specified test set.'''

    log_prob_sum = 0
    file_word_count = 0

    with open(filename, encoding='iso-8859-1') as f:
        for line in f:
            prob, num_tokens = eval_sentence(line, model, log_prob_func, caches)
            log_prob_sum += prob
            file_word_count += num_tokens
        f.close()

    average_log_prob = log_prob_sum / file_word_count
    perplexity = 2**(-average_log_prob)
    return perplexity


def eval_sentence(sentence, model, log_prob_func, caches):
    '''Returns log probability of a sentence and how many tokens were in the sentence.'''

    tokens = [token if (token,) in model[0] else UNK for token in sentence.split()]
    num_tokens = len(tokens) + 1
    tokens.insert(0, START)
    tokens.insert(0, START)
    tokens.append(STOP)

    log_prob_sum = 0
    for i in range(len(tokens) - 2):
        n_gram = tuple(tokens[i:i+3])
        next_prob = log_prob_func(n_gram, model, caches)
        log_prob_sum += next_prob

    return log_prob_sum, num_tokens


if __name__ == '__main__':
    main()
