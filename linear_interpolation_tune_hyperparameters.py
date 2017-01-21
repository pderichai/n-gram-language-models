import math
from collections import Counter


# start symbol
START = '<s>'
# stop symbol
STOP = '</s>'
# unk symbol
UNK = '<UNK>'
# the names of the different datasets we want to model
DATASETS = ['reuters', 'brown', 'gutenberg']
# whether or not to evaluate on the dev datasets
DEV = True

# linear interpolation hyper-parameters
LAMBDA_1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LAMBDA_2s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def main():
    # returns the log probability of a specified n-gram
    def get_log_prob(n_gram, model, caches):
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

    for i in LAMBDA_1s:
        for j in LAMBDA_2s:
            LAMBDA_1 = i
            LAMBDA_2 = j
            LAMBDA_3 = 1 - LAMBDA_1 - LAMBDA_2
            print('LAMBDA1: ' + str(LAMBDA_1) + ' LAMBDA2: ' + str(LAMBDA_2) + ' LAMBDA3: ' + str(LAMBDA_3))
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
    # initializing empty Counter objects to store the n-grams
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


# adds the n-grams to the specified Counter from the specified tokens
def add_n_gram_counts(n, n_grams, tokens):
    for i in range(len(tokens) - (n - 1)):
        n_grams[tuple(tokens[i:i+n])] += 1

    return n_grams


# returns the perplexity of the model
def eval_model(filename, model, log_prob_func, caches):
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


# returns log probability of a sentence and how many tokens were in the sentence
def eval_sentence(sentence, model, log_prob_func, caches):
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